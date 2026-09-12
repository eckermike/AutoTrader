"""
Cryptocurrency Paper Trading Bot - Main Daemon Entry Point.
Engineered for macOS (Apple Silicon).

Orchestrates:
- alpaca-py Paper Trading spot crypto execution (BTC/USD)
- Tri-Factor Decision Engine (Technical + Volume/Regime + Ollama NLP Sentiment)
- Virtual Tax Escrow Engine with persistent JSON state and hard capital gate
- Trailing stop-loss execution
- Scheduled daemon loop with clean POSIX signal shutdown (SIGINT / SIGTERM)
"""

import argparse
import logging
import signal
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from config import BotConfig, get_config
from execution.alpaca_client import AlpacaPaperClient, PositionInfo
from execution.liquidity_manager import LiquidityManager
from factors.sentiment import create_sentiment_analyzer

from factors.technical import TechnicalFactor
from factors.volume_regime import VolumeRegimeFactor
from notifier import TradeNotifier
from options.options_client import AlpacaOptionsClient
from options.spread_engine import SpreadPortfolioManager
from options.wheel_engine import WheelEngine, WheelPortfolioManager
from tax_engine import InsufficientTradableCashError, TaxEngine

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("autotrader.daemon")


class TradingDaemon:
    """
    Continuous paper trading execution daemon with tri-factor scoring,
    virtual tax escrow management, and clean signal shutdown.
    """

    def __init__(self, config: BotConfig, dry_run: bool = False):
        self.config = config
        self.dry_run = dry_run
        self.running = True
        self.cycle_count = 0

        # Multi-asset crypto position and risk tracking state
        self.crypto_positions: Dict[str, Dict[str, Any]] = {}

        # 1. Initialize Virtual Tax Escrow Engine
        self.tax_engine = TaxEngine(
            filepath=self.config.TAX_RESERVE_FILE,
            tax_rate=self.config.TAX_RATE,
        )

        # 2. Initialize Broker Client (strict paper mode)
        self.client = AlpacaPaperClient(
            api_key=self.config.ALPACA_API_KEY,
            secret_key=self.config.ALPACA_SECRET_KEY,
            paper=self.config.ALPACA_PAPER,
            base_url=self.config.ALPACA_BASE_URL,
            mock_mode=self.dry_run,
        )

        # 3. Initialize Decision Engine Factors
        self.technical_factor = TechnicalFactor()
        self.volume_factor = VolumeRegimeFactor()
        self.sentiment_analyzer = create_sentiment_analyzer(
            provider="mock" if self.dry_run else self.config.SENTIMENT_PROVIDER,
            ollama_base_url=self.config.OLLAMA_BASE_URL,
            ollama_model=self.config.OLLAMA_MODEL,
            gemini_api_key=self.config.GEMINI_API_KEY,
        )

        # 4. Initialize Notification Engine (ntfy iOS PWA Push + iMessage + macOS banners)
        self.notifier = TradeNotifier(
            recipient=self.config.ALERT_RECIPIENT,
            ntfy_topic=self.config.NTFY_TOPIC,
            enabled=self.config.ALERT_ENABLED,
            macos_banner=self.config.ALERT_MACOS_BANNER,
        )

        # 5. Initialize Options Broker Client
        self.options_client = AlpacaOptionsClient(
            api_key=self.config.ALPACA_API_KEY,
            secret_key=self.config.ALPACA_SECRET_KEY,
            paper=self.config.ALPACA_PAPER,
            base_url=self.config.ALPACA_BASE_URL,
            mock_mode=self.dry_run,
        )

        # 6. Initialize Autonomous Liquidity & Cash Yield Manager
        trading_client = (
            getattr(self.options_client, "trading_client", None)
            or getattr(self.client, "trading_client", None)
        )
        self.liquidity_manager = LiquidityManager(
            config=self.config,
            trading_client=trading_client,
            notifier=self.notifier,
            tax_engine=self.tax_engine,
        )

        # 7. Initialize Multi-Asset Option Wheel Portfolio Engine
        self.wheel_engine = WheelPortfolioManager(
            config=self.config,
            options_client=self.options_client,
            tax_engine=self.tax_engine,
            notifier=self.notifier,
            liquidity_manager=self.liquidity_manager,
        )

        # 8. Initialize Defined-Risk Option Spreads Engine (SPY, QQQ, IWM)
        self.spread_engine = SpreadPortfolioManager(
            config=self.config,
            options_client=self.options_client,
            tax_engine=self.tax_engine,
            notifier=self.notifier,
            liquidity_manager=self.liquidity_manager,
        )


        # Daily tracking and briefing state

        self.current_day_str: Optional[str] = None
        self.trades_executed_today: int = 0
        self.daily_max_composite_scores: Dict[str, float] = {}
        self.last_daily_recap_date: Optional[str] = None

        # Register POSIX signal handlers for graceful termination
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    # --- Backward compatibility properties for single-symbol access & tests ---
    @property
    def position_qty(self) -> float:
        return self.crypto_positions.get(self.config.TARGET_SYMBOL, {}).get("qty", 0.0)

    @position_qty.setter
    def position_qty(self, val: float):
        if self.config.TARGET_SYMBOL not in self.crypto_positions:
            self.crypto_positions[self.config.TARGET_SYMBOL] = {"qty": 0.0, "entry_price": None, "peak_price": None}
        self.crypto_positions[self.config.TARGET_SYMBOL]["qty"] = val

    @property
    def position_entry_price(self) -> Optional[float]:
        return self.crypto_positions.get(self.config.TARGET_SYMBOL, {}).get("entry_price")

    @position_entry_price.setter
    def position_entry_price(self, val: Optional[float]):
        if self.config.TARGET_SYMBOL not in self.crypto_positions:
            self.crypto_positions[self.config.TARGET_SYMBOL] = {"qty": 0.0, "entry_price": None, "peak_price": None}
        self.crypto_positions[self.config.TARGET_SYMBOL]["entry_price"] = val

    @property
    def position_peak_price(self) -> Optional[float]:
        return self.crypto_positions.get(self.config.TARGET_SYMBOL, {}).get("peak_price")

    @position_peak_price.setter
    def position_peak_price(self, val: Optional[float]):
        if self.config.TARGET_SYMBOL not in self.crypto_positions:
            self.crypto_positions[self.config.TARGET_SYMBOL] = {"qty": 0.0, "entry_price": None, "peak_price": None}
        self.crypto_positions[self.config.TARGET_SYMBOL]["peak_price"] = val

    def _handle_signal(self, signum, frame):
        """Intercepts termination signals and flags the daemon for clean shutdown."""
        sig_name = signal.Signals(signum).name
        logger.warning(
            "Received signal %s. Initiating graceful shutdown after current cycle...",
            sig_name,
        )
        self.running = False

    def sync_position_state(self, symbol: Optional[str] = None) -> Optional[PositionInfo]:
        """Synchronizes local position tracking with Alpaca broker state for a given symbol."""
        sym = symbol or self.config.TARGET_SYMBOL
        if sym not in self.crypto_positions:
            self.crypto_positions[sym] = {"qty": 0.0, "entry_price": None, "peak_price": None}

        position = self.client.get_crypto_position(sym)
        if position and position.qty > 0:
            self.crypto_positions[sym]["qty"] = position.qty
            if self.crypto_positions[sym]["entry_price"] is None:
                self.crypto_positions[sym]["entry_price"] = position.avg_entry_price
                self.crypto_positions[sym]["peak_price"] = max(
                    position.avg_entry_price, position.current_price
                )
            else:
                self.crypto_positions[sym]["peak_price"] = max(
                    self.crypto_positions[sym]["peak_price"] or position.current_price,
                    position.current_price,
                )
        else:
            self.crypto_positions[sym]["qty"] = 0.0
            self.crypto_positions[sym]["entry_price"] = None
            self.crypto_positions[sym]["peak_price"] = None
        return position

    def evaluate_signals(
        self,
        current_price: float,
        symbol: Optional[str] = None,
        bars_df: Optional[Any] = None,
    ) -> tuple[float, float, float, float, str, Optional[str]]:
        """
        Executes the Tri-Factor Decision Engine:
        - Technical Factor (40%)
        - Volume/Regime Factor (30%)
        - Sentiment Factor (30%)
        Returns (tech_score, vol_score, sent_score, composite_score, signal_type, exit_reason).
        """
        sym = symbol or self.config.TARGET_SYMBOL

        # Fetch OHLCV historical bars if not passed
        if bars_df is None:
            bars_df = self.client.get_crypto_bars(
                symbol=sym,
                limit=250,
            )

        # 1. Technical Factor
        tech_res = self.technical_factor.evaluate(bars_df)
        tech_score = tech_res.score

        # 2. Volume/Regime Factor
        vol_res = self.volume_factor.evaluate(bars_df)
        vol_score = vol_res.score

        # 3. Sentiment Factor
        coin_name = sym.split("/")[0]
        market_headline_sample = (
            f"{coin_name} trading around ${current_price:,.2f}. Market participants evaluate "
            f"on-chain liquidity, macroeconomic indicators, and institutional ETF inflows."
        )
        sent_res = self.sentiment_analyzer.analyze(market_headline_sample)
        sent_score = sent_res.sentiment_score

        # Weighted Ensemble Calculation
        composite_score = (
            (self.config.WEIGHT_TECHNICAL * tech_score)
            + (self.config.WEIGHT_VOLUME * vol_score)
            + (self.config.WEIGHT_SENTIMENT * sent_score)
        )
        composite_score = round(composite_score, 4)

        # Trailing Stop-Loss Check for this symbol
        trailing_stop_triggered = False
        exit_reason = None
        pos_info = self.crypto_positions.get(sym, {})
        pos_qty = pos_info.get("qty", 0.0)
        pos_peak = pos_info.get("peak_price")

        if pos_qty > 0 and pos_peak:
            drawdown = (pos_peak - current_price) / pos_peak
            if drawdown >= self.config.TRAILING_STOP_LOSS_PCT:
                trailing_stop_triggered = True
                exit_reason = f"Trailing Stop-Loss Triggered (-{drawdown:.1%} from peak ${pos_peak:,.2f})"

        # Signal Determination
        if trailing_stop_triggered:
            signal_type = "SELL"
        elif composite_score <= self.config.SELL_TRIGGER_SCORE:
            signal_type = "SELL"
            exit_reason = f"Composite Score ({composite_score:+.3f}) <= Sell Threshold ({self.config.SELL_TRIGGER_SCORE:+.3f})"
        elif pos_qty > 0:
            # Active position already open; ride momentum and manage risk via trailing stop
            signal_type = "HOLD"
        elif composite_score >= self.config.BUY_TRIGGER_SCORE:
            signal_type = "BUY"
        else:
            signal_type = "HOLD"

        return tech_score, vol_score, sent_score, composite_score, signal_type, exit_reason

    def run_cycle(self) -> None:
        """Executes a single evaluation and order placement cycle across the multi-crypto portfolio."""
        self.cycle_count += 1
        account = self.client.get_account()
        tradable_cash = self.tax_engine.calculate_tradable_cash(account.cash)
        remaining_cash = tradable_cash

        # Track daily rollover
        now = datetime.now()
        today_str = now.strftime("%Y-%m-%d")
        if self.current_day_str != today_str:
            self.current_day_str = today_str
            self.trades_executed_today = 0
            self.daily_max_composite_scores = {}

        # Determine target crypto symbols list
        raw_symbols = getattr(self.config, "TARGET_SYMBOLS", [self.config.TARGET_SYMBOL])
        if isinstance(raw_symbols, str):
            symbols = [s.strip().upper() for s in raw_symbols.split(",") if s.strip()]
        else:
            symbols = list(raw_symbols)
        if not symbols:
            symbols = [self.config.TARGET_SYMBOL]

        for symbol in symbols:
            position = self.sync_position_state(symbol)

            # Approximate current price from bars
            try:
                bars_df = self.client.get_crypto_bars(symbol, limit=250)
                if bars_df.empty:
                    logger.warning("Empty bars returned for %s, skipping.", symbol)
                    continue
                current_price = float(bars_df["close"].iloc[-1])
            except Exception as e:
                logger.error("Failed to fetch bars for %s: %s", symbol, e)
                continue

            # Evaluate Tri-Factor Engine
            tech_score, vol_score, sent_score, composite_score, signal_type, exit_reason = (
                self.evaluate_signals(current_price, symbol=symbol, bars_df=bars_df)
            )

            # Track daily peak momentum score
            self.daily_max_composite_scores[symbol] = max(
                self.daily_max_composite_scores.get(symbol, -1.0),
                composite_score,
            )

            # Exact Breakdown Logging (as specified in requirements)
            price_display = f"{current_price:,.2f}" if current_price >= 1.0 else f"{current_price:,.4f}"
            logger.info(
                "[CYCLE #%d] Symbol: %s | Price: $%s | Technical: %+0.3f | Volume: %+0.3f | "
                "Sentiment: %+0.3f | Composite: %+0.3f => Signal: %s",
                self.cycle_count,
                symbol,
                price_display,
                tech_score,
                vol_score,
                sent_score,
                composite_score,
                signal_type,
            )

            # Execution Logic
            if signal_type == "BUY":
                # Strictly prevent duplicate buy orders if position is already active
                current_pos_val = position.market_value if position else 0.0
                if position and position.qty > 0:
                    logger.info(
                        "Position already open for %s (%.4f units, $%s). Holding position and skipping additional BUY.",
                        symbol,
                        position.qty,
                        f"{current_pos_val:,.2f}",
                    )
                    continue

                if current_pos_val >= self.config.MAX_POSITION_USD:
                    logger.info(
                        "Maximum position limit ($%s) reached for %s. Skipping BUY order.",
                        f"{self.config.MAX_POSITION_USD:,.2f}",
                        symbol,
                    )
                    continue

                # Determine order size bounded by Tradable Cash & remaining cycle cash
                target_order_size = min(
                    self.config.ORDER_SIZE_USD,
                    self.config.MAX_POSITION_USD - current_pos_val,
                )
                target_order_size = min(target_order_size, remaining_cash)
                target_order_size = round(target_order_size, 2)

                if target_order_size < 10.0:
                    needed_cap = self.config.ORDER_SIZE_USD - remaining_cash
                    logger.info(
                        "Insufficient remaining tradable cash ($%s) for %s BUY. Skipping.",
                        f"{remaining_cash:,.2f}",
                        symbol,
                    )
                    if hasattr(self, "liquidity_manager") and self.liquidity_manager:
                        self.liquidity_manager.request_liquidation_for_opportunity(
                            needed_cash=round(max(needed_cap, self.config.ORDER_SIZE_USD), 2),
                            target_symbol=symbol,
                            opportunity_type="Crypto Tri-Factor Breakout BUY",
                            current_price=current_price,
                            reserve_symbol="SGOV",
                        )
                    continue

                try:
                    # Hard capital gate enforcement
                    self.tax_engine.validate_order_budget(account.cash, target_order_size)

                    logger.info(
                        "Submitting BUY Market Order: $%s for %s",
                        f"{target_order_size:,.2f}",
                        symbol,
                    )
                    order_receipt = self.client.submit_market_order(
                        symbol=symbol,
                        side="BUY",
                        notional=target_order_size,
                        estimated_price=current_price,
                    )
                    logger.info(
                        "Order Executed: %s (Status: %s)",
                        order_receipt.client_order_id,
                        order_receipt.status,
                    )

                    # Update local position tracking
                    if symbol not in self.crypto_positions:
                        self.crypto_positions[symbol] = {"qty": 0.0, "entry_price": None, "peak_price": None}
                    self.crypto_positions[symbol]["entry_price"] = current_price
                    self.crypto_positions[symbol]["peak_price"] = current_price
                    remaining_cash -= target_order_size
                    self.trades_executed_today += 1

                    # Dispatch Real-Time Trade Alert to all iCloud devices
                    self.notifier.notify_buy(
                        symbol=symbol,
                        price=current_price,
                        notional=target_order_size,
                        qty=order_receipt.qty or (target_order_size / current_price),
                        composite_score=composite_score,
                        tech_score=tech_score,
                        vol_score=vol_score,
                        sent_score=sent_score,
                        tradable_cash=remaining_cash,
                        tax_reserve=self.tax_engine.current_reserve,
                    )

                except InsufficientTradableCashError as e:
                    logger.error("Order Blocked by Tax Escrow Engine for %s: %s", symbol, e)
                    if hasattr(self, "liquidity_manager") and self.liquidity_manager:
                        self.liquidity_manager.request_liquidation_for_opportunity(
                            needed_cash=round(target_order_size, 2),
                            target_symbol=symbol,
                            opportunity_type="Crypto Tri-Factor Breakout BUY",
                            current_price=current_price,
                            reserve_symbol="SGOV",
                        )
                except Exception as e:
                    logger.exception("Failed to execute BUY order for %s: %s", symbol, e)

            elif signal_type == "SELL":
                if position and position.qty > 0:
                    logger.info("Exiting Position: %s | Reason: %s", symbol, exit_reason)
                    close_order = self.client.close_crypto_position(symbol)

                    if close_order:
                        # Calculate realized PnL and update virtual tax escrow
                        entry_p = (
                            self.crypto_positions.get(symbol, {}).get("entry_price")
                            or position.avg_entry_price
                        )
                        exit_p = close_order.filled_avg_price or current_price
                        trade_record = self.tax_engine.record_closed_trade(
                            symbol=symbol,
                            side="SELL",
                            qty=position.qty,
                            entry_price=entry_p,
                            exit_price=exit_p,
                        )

                        logger.info(
                            "Trade Settled [%s]: Gross PnL=$%+0.2f | Tax Allocated=$%0.2f | Tax Credit=$%0.2f | New Reserve=$%0.2f",
                            symbol,
                            trade_record.gross_pnl,
                            trade_record.tax_allocated,
                            trade_record.tax_credit,
                            trade_record.reserve_after,
                        )

                        # Dispatch Real-Time Trade Alert to all iCloud devices
                        self.notifier.notify_sell(
                            symbol=symbol,
                            exit_price=exit_p,
                            qty=position.qty,
                            reason=exit_reason or "Composite Score <= Sell Threshold",
                            gross_pnl=trade_record.gross_pnl,
                            tax_allocated=trade_record.tax_allocated,
                            tax_credit=trade_record.tax_credit,
                            reserve_after=trade_record.reserve_after,
                        )

                        self.trades_executed_today += 1

                        # Reset position tracking
                        self.crypto_positions[symbol] = {
                            "qty": 0.0,
                            "entry_price": None,
                            "peak_price": None,
                        }
                else:
                    logger.debug(
                        "SELL signal generated for %s but no open position to liquidate.",
                        symbol,
                    )

            elif signal_type == "HOLD":
                pos_peak = self.crypto_positions.get(symbol, {}).get("peak_price")
                if position and pos_peak:
                    pct_drawdown = (pos_peak - current_price) / pos_peak
                    logger.debug(
                        "Holding %s: Drawdown from peak: %.2f%% (Stop threshold: %.2f%%)",
                        symbol,
                        pct_drawdown * 100,
                        self.config.TRAILING_STOP_LOSS_PCT * 100,
                    )

        # Portfolio Summary Logging
        open_positions = [
            f"{s}: {p['qty']:.4f} units"
            for s, p in self.crypto_positions.items()
            if p.get("qty", 0.0) > 0
        ]
        pos_summary_str = ", ".join(open_positions) if open_positions else "None (Flat)"
        logger.info(
            "Portfolio: Cash=$%s | Tax Reserve=$%s | Tradable Cash=$%s | Crypto Positions=[%s]",
            f"{account.cash:,.2f}",
            f"{self.tax_engine.current_reserve:,.2f}",
            f"{remaining_cash:,.2f}",
            pos_summary_str,
        )

        # --- Strategy 2: Multi-Asset Option Wheel Execution ---
        if self.config.WHEEL_ENABLED:
            try:
                self.wheel_engine.step(total_cash=account.cash)
            except Exception as e:
                logger.exception("Error executing Multi-Asset Wheel Strategy cycle: %s", e)

        # --- Strategy 3: End-of-Day Daily Briefing & Zero-Trade Diagnostics ---
        try:
            self.check_and_dispatch_daily_recap(
                account_cash=account.cash,
                tradable_cash=remaining_cash,
            )
        except Exception as e:
            logger.exception("Error executing Daily Briefing check: %s", e)

        # --- Strategy 4: Autonomous Liquidity & Cash Yield Manager ---
        if getattr(self.config, "LIQUIDITY_RESERVE_ENABLED", True) and hasattr(self, "liquidity_manager"):
            try:
                self.liquidity_manager.step()
            except Exception as e:
                logger.exception("Error executing Liquidity Manager cycle: %s", e)

        # --- Strategy 5: Defined-Risk Option Spreads (SPY, QQQ, IWM) ---
        if getattr(self.config, "SPREAD_ENABLED", True) and hasattr(self, "spread_engine"):
            try:
                is_mkt_open = (
                    self.options_client.is_market_open()
                    if hasattr(self.options_client, "is_market_open")
                    else True
                )
                self.spread_engine.step_all(is_market_open=is_mkt_open)
            except Exception as e:
                logger.exception("Error executing Defined-Risk Spreads cycle: %s", e)


    def check_and_dispatch_daily_recap(
        self,
        account_cash: float,
        tradable_cash: float,
        force: bool = False,
    ) -> bool:
        """
        Checks if the daily recap schedule is met, compiles diagnostics, and dispatches notification.
        Returns True if a recap was dispatched, False otherwise.
        """
        if not getattr(self.config, "DAILY_RECAP_ENABLED", True) and not force:
            return False

        now = datetime.now()
        today_str = now.strftime("%Y-%m-%d")

        target_hour = getattr(self.config, "DAILY_RECAP_HOUR", 17)
        target_minute = getattr(self.config, "DAILY_RECAP_MINUTE", 0)

        is_time = (now.hour > target_hour) or (now.hour == target_hour and now.minute >= target_minute)
        if (not is_time or self.last_daily_recap_date == today_str) and not force:
            return False

        # Query total trade fills today from Alpaca if live client is active
        trades_count = self.trades_executed_today
        if not self.dry_run and getattr(self.client, "trading_client", None):
            try:
                from alpaca.trading.requests import GetActivitiesRequest
                from alpaca.trading.enums import ActivityType
                acts = self.client.trading_client.get_activities(
                    GetActivitiesRequest(activity_types=[ActivityType.FILL], date=today_str)
                )
                if acts is not None:
                    trades_count = len(acts)
            except Exception as e:
                logger.debug("Could not query Alpaca daily activities: %s", e)

        # Check if only zero trades mode is enabled
        if getattr(self.config, "DAILY_RECAP_ONLY_ZERO_TRADES", False) and trades_count > 0 and not force:
            self.last_daily_recap_date = today_str
            return False

        # 1. Gather Crypto Scanner diagnostics
        crypto_diag: List[str] = []
        raw_symbols = getattr(self.config, "TARGET_SYMBOLS", [self.config.TARGET_SYMBOL])
        if isinstance(raw_symbols, str):
            symbols = [s.strip().upper() for s in raw_symbols.split(",") if s.strip()]
        else:
            symbols = list(raw_symbols)
        if not symbols:
            symbols = [self.config.TARGET_SYMBOL]

        for sym in symbols:
            pos_info = self.crypto_positions.get(sym, {})
            qty = pos_info.get("qty", 0.0)
            entry = pos_info.get("entry_price")
            peak = pos_info.get("peak_price")
            max_score = self.daily_max_composite_scores.get(sym, 0.0)

            if qty > 0:
                pos = self.client.get_crypto_position(sym)
                curr_p = pos.current_price if pos else (peak or entry or 0.0)
                gain_pct = ((curr_p - entry) / entry * 100) if (entry and entry > 0) else 0.0
                crypto_diag.append(
                    f"{sym}: Active position ({qty:.4f} units, {gain_pct:+.1f}%). 5% trailing stop active."
                )
            else:
                crypto_diag.append(
                    f"{sym}: Peak score {max_score:+.3f} (Trigger: {self.config.BUY_TRIGGER_SCORE:+.2f}). Market below momentum breakout threshold."
                )

        # 2. Gather Option Wheel diagnostics
        wheel_diag: List[str] = []
        if getattr(self.config, "WHEEL_ENABLED", True) and hasattr(self, "wheel_engine"):
            for sym, engine in self.wheel_engine.engines.items():
                active_positions = engine.client.get_active_option_positions(sym)
                open_orders = engine.client.get_open_orders(sym)

                active_puts = [p for p in active_positions if p.contract_type == "put" and p.qty < 0]
                active_calls = [p for p in active_positions if p.contract_type == "call" and p.qty < 0]

                if active_puts:
                    for p in active_puts:
                        wheel_diag.append(f"{sym}: Short Put active ({p.symbol}). Waiting for 50% profit decay.")
                elif active_calls:
                    for p in active_calls:
                        wheel_diag.append(f"{sym}: Covered Call active ({p.symbol}).")
                elif open_orders:
                    for o in open_orders:
                        wheel_diag.append(f"{sym}: Limit order pending in order book ({getattr(o, 'symbol', 'limit')}).")
                else:
                    wheel_diag.append(f"{sym}: Idle / Staging next entry.")
        else:
            wheel_diag.append("Option Wheel strategy disabled.")

        # 3. Dispatch Daily Recap Alert
        logger.info(
            "Dispatching End-of-Day Briefing for %s (Trades Today: %d)...",
            today_str,
            trades_count,
        )
        self.notifier.notify_daily_recap(
            date_str=today_str,
            trades_count=trades_count,
            crypto_diagnostics=crypto_diag,
            wheel_diagnostics=wheel_diag,
            cash=account_cash,
            tradable_cash=tradable_cash,
            tax_reserve=self.tax_engine.current_reserve,
        )

        self.last_daily_recap_date = today_str
        return True

    def start(self, max_cycles: Optional[int] = None) -> None:
        """Starts the daemon loop."""
        target_display = (
            ", ".join(self.config.TARGET_SYMBOLS)
            if isinstance(self.config.TARGET_SYMBOLS, list)
            else self.config.TARGET_SYMBOL
        )
        logger.info("==========================================================")
        logger.info("Starting Crypto Paper Trading Bot (Apple Silicon Edition)")
        logger.info(
            "Target Symbols: %s | Interval: %ds | Paper Mode: %s",
            target_display,
            self.config.CYCLE_INTERVAL_SECONDS,
            self.config.ALPACA_PAPER,
        )
        logger.info(
            "Sentiment Provider: %s | Tax Escrow Rate: %.0f%%",
            self.config.SENTIMENT_PROVIDER,
            self.config.TAX_RATE * 100,
        )
        logger.info("==========================================================")

        while self.running:
            try:
                self.run_cycle()
            except Exception as e:
                logger.exception("Unexpected error in daemon cycle: %s", e)

            if max_cycles is not None and self.cycle_count >= max_cycles:
                break
            if not self.running:
                break

            # Sleep with interruptible sub-intervals for responsive shutdown
            sleep_chunks = int(self.config.CYCLE_INTERVAL_SECONDS)
            for _ in range(sleep_chunks):
                if not self.running:
                    break
                time.sleep(1)

        self.shutdown()

    def shutdown(self) -> None:
        """Logs final performance and tax reserve summary on exit."""
        logger.info("Daemon shutdown complete. Generating final audit summary:")
        summary = self.tax_engine.get_summary()
        logger.info("Tax Escrow Reserve Balance: $%s", f"{summary['tax_reserve']:,.2f}")
        logger.info("Cumulative Realized Profit: $%s", f"{summary['total_realized_profit']:,.2f}")
        logger.info("Cumulative Realized Loss:   $%s", f"{summary['total_realized_loss']:,.2f}")
        logger.info("Net Realized PnL:           $%s", f"{summary['net_realized_pnl']:,.2f}")
        logger.info("Total Trades Logged:        %d", summary["trade_count"])


def main():
    parser = argparse.ArgumentParser(
        description="Cryptocurrency Paper Trading Bot with Alpaca & Virtual Tax Escrow"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single cycle and exit (useful for cron or smoke testing)",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=None,
        help="Run a specific number of cycles and exit",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in mock/offline mode with synthetic data and mock orders",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Override target symbol (e.g. BTC/USD or ETH/USD)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=None,
        help="Override execution interval in seconds",
    )
    args = parser.parse_args()

    config = get_config()
    if args.symbol:
        config.TARGET_SYMBOL = args.symbol
        config.TARGET_SYMBOLS = [args.symbol]
    if args.interval:
        config.CYCLE_INTERVAL_SECONDS = args.interval

    max_cycles = 1 if args.once else args.cycles
    daemon = TradingDaemon(config=config, dry_run=args.dry_run)
    daemon.start(max_cycles=max_cycles)


if __name__ == "__main__":
    main()
