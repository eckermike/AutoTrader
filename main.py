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
from typing import Optional

from config import BotConfig, get_config
from execution.alpaca_client import AlpacaPaperClient, PositionInfo
from factors.sentiment import create_sentiment_analyzer
from factors.technical import TechnicalFactor
from factors.volume_regime import VolumeRegimeFactor
from notifier import TradeNotifier
from options.options_client import AlpacaOptionsClient
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

        # Position and risk tracking state
        self.position_entry_price: Optional[float] = None
        self.position_peak_price: Optional[float] = None
        self.position_qty: float = 0.0

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

        # 5. Initialize Multi-Asset Option Wheel Portfolio Engine
        self.options_client = AlpacaOptionsClient(
            api_key=self.config.ALPACA_API_KEY,
            secret_key=self.config.ALPACA_SECRET_KEY,
            paper=self.config.ALPACA_PAPER,
            base_url=self.config.ALPACA_BASE_URL,
            mock_mode=self.dry_run,
        )
        self.wheel_engine = WheelPortfolioManager(
            config=self.config,
            options_client=self.options_client,
            tax_engine=self.tax_engine,
            notifier=self.notifier,
        )

        # Register POSIX signal handlers for graceful termination
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum, frame):
        """Intercepts termination signals and flags the daemon for clean shutdown."""
        sig_name = signal.Signals(signum).name
        logger.warning(
            "Received signal %s. Initiating graceful shutdown after current cycle...",
            sig_name,
        )
        self.running = False

    def sync_position_state(self) -> Optional[PositionInfo]:
        """Synchronizes local position tracking with Alpaca broker state."""
        position = self.client.get_crypto_position(self.config.TARGET_SYMBOL)
        if position and position.qty > 0:
            self.position_qty = position.qty
            if self.position_entry_price is None:
                self.position_entry_price = position.avg_entry_price
                self.position_peak_price = max(
                    position.avg_entry_price, position.current_price
                )
            else:
                self.position_peak_price = max(
                    self.position_peak_price or position.current_price,
                    position.current_price,
                )
        else:
            self.position_qty = 0.0
            self.position_entry_price = None
            self.position_peak_price = None
        return position

    def evaluate_signals(self, current_price: float) -> tuple[float, float, float, float, str, Optional[str]]:
        """
        Executes the Tri-Factor Decision Engine:
        - Technical Factor (40%)
        - Volume/Regime Factor (30%)
        - Sentiment Factor (30%)
        Returns (tech_score, vol_score, sent_score, composite_score, signal_type, exit_reason).
        """
        # Fetch OHLCV historical bars
        bars_df = self.client.get_crypto_bars(
            symbol=self.config.TARGET_SYMBOL,
            limit=250,
        )

        # 1. Technical Factor
        tech_res = self.technical_factor.evaluate(bars_df)
        tech_score = tech_res.score

        # 2. Volume/Regime Factor
        vol_res = self.volume_factor.evaluate(bars_df)
        vol_score = vol_res.score

        # 3. Sentiment Factor
        market_headline_sample = (
            f"Bitcoin trading around ${current_price:,.2f}. Market participants evaluate "
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

        # Trailing Stop-Loss Check
        trailing_stop_triggered = False
        exit_reason = None
        if self.position_qty > 0 and self.position_peak_price:
            drawdown = (self.position_peak_price - current_price) / self.position_peak_price
            if drawdown >= self.config.TRAILING_STOP_LOSS_PCT:
                trailing_stop_triggered = True
                exit_reason = f"Trailing Stop-Loss Triggered (-{drawdown:.1%} from peak ${self.position_peak_price:,.2f})"

        # Signal Determination
        if trailing_stop_triggered:
            signal_type = "SELL"
        elif composite_score >= self.config.BUY_TRIGGER_SCORE:
            signal_type = "BUY"
        elif composite_score <= self.config.SELL_TRIGGER_SCORE:
            signal_type = "SELL"
            exit_reason = f"Composite Score ({composite_score:+.3f}) <= Sell Threshold ({self.config.SELL_TRIGGER_SCORE:+.3f})"
        else:
            signal_type = "HOLD"

        return tech_score, vol_score, sent_score, composite_score, signal_type, exit_reason

    def run_cycle(self) -> None:
        """Executes a single evaluation and order placement cycle."""
        self.cycle_count += 1
        account = self.client.get_account()
        tradable_cash = self.tax_engine.calculate_tradable_cash(account.cash)
        position = self.sync_position_state()

        # Approximate current price from bars or position
        bars_df = self.client.get_crypto_bars(self.config.TARGET_SYMBOL, limit=5)
        current_price = float(bars_df["close"].iloc[-1])

        # Evaluate Tri-Factor Engine
        tech_score, vol_score, sent_score, composite_score, signal_type, exit_reason = (
            self.evaluate_signals(current_price)
        )

        # Exact Breakdown Logging (as specified in requirements)
        logger.info(
            "[CYCLE #%d] Symbol: %s | Price: $%s | Technical: %+0.3f | Volume: %+0.3f | "
            "Sentiment: %+0.3f | Composite: %+0.3f => Signal: %s",
            self.cycle_count,
            self.config.TARGET_SYMBOL,
            f"{current_price:,.2f}",
            tech_score,
            vol_score,
            sent_score,
            composite_score,
            signal_type,
        )

        logger.info(
            "Portfolio: Cash=$%s | Tax Reserve=$%s | Tradable Cash=$%s | Position=%s",
            f"{account.cash:,.2f}",
            f"{self.tax_engine.current_reserve:,.2f}",
            f"{tradable_cash:,.2f}",
            f"{position.qty:.6f} units (${position.market_value:,.2f})" if position else "None (Flat)",
        )

        # Execution Logic
        if signal_type == "BUY":
            # Check maximum exposure limits
            current_pos_val = position.market_value if position else 0.0
            if current_pos_val >= self.config.MAX_POSITION_USD:
                logger.info("Maximum position limit ($%s) reached. Skipping BUY order.", f"{self.config.MAX_POSITION_USD:,.2f}")
                return

            # Determine order size bounded by Tradable Cash
            target_order_size = min(
                self.config.ORDER_SIZE_USD,
                self.config.MAX_POSITION_USD - current_pos_val,
            )

            try:
                # Hard capital gate enforcement
                self.tax_engine.validate_order_budget(account.cash, target_order_size)
                
                logger.info("Submitting BUY Market Order: $%s for %s", f"{target_order_size:,.2f}", self.config.TARGET_SYMBOL)
                order_receipt = self.client.submit_market_order(
                    symbol=self.config.TARGET_SYMBOL,
                    side="BUY",
                    notional=target_order_size,
                    estimated_price=current_price,
                )
                logger.info("Order Executed: %s (Status: %s)", order_receipt.client_order_id, order_receipt.status)
                
                # Update local position tracking
                self.position_entry_price = current_price
                self.position_peak_price = current_price

                # Dispatch Real-Time Trade Alert to all iCloud devices
                self.notifier.notify_buy(
                    symbol=self.config.TARGET_SYMBOL,
                    price=current_price,
                    notional=target_order_size,
                    qty=order_receipt.qty or (target_order_size / current_price),
                    composite_score=composite_score,
                    tech_score=tech_score,
                    vol_score=vol_score,
                    sent_score=sent_score,
                    tradable_cash=tradable_cash,
                    tax_reserve=self.tax_engine.current_reserve,
                )

            except InsufficientTradableCashError as e:
                logger.error("Order Blocked by Tax Escrow Engine: %s", e)

        elif signal_type == "SELL":
            if position and position.qty > 0:
                logger.info("Exiting Position: %s | Reason: %s", self.config.TARGET_SYMBOL, exit_reason)
                close_order = self.client.close_crypto_position(self.config.TARGET_SYMBOL)
                
                if close_order:
                    # Calculate realized PnL and update virtual tax escrow
                    entry_p = self.position_entry_price or position.avg_entry_price
                    exit_p = close_order.filled_avg_price or current_price
                    trade_record = self.tax_engine.record_closed_trade(
                        symbol=self.config.TARGET_SYMBOL,
                        side="SELL",
                        qty=position.qty,
                        entry_price=entry_p,
                        exit_price=exit_p,
                    )
                    logger.info(
                        "Trade Settled: Gross PnL=$%+0.2f | Tax Allocated=$%0.2f | Tax Credit=$%0.2f | New Reserve=$%0.2f",
                        trade_record.gross_pnl,
                        trade_record.tax_allocated,
                        trade_record.tax_credit,
                        trade_record.reserve_after,
                    )

                    # Dispatch Real-Time Trade Alert to all iCloud devices
                    self.notifier.notify_sell(
                        symbol=self.config.TARGET_SYMBOL,
                        exit_price=exit_p,
                        qty=position.qty,
                        reason=exit_reason or "Composite Score <= Sell Threshold",
                        gross_pnl=trade_record.gross_pnl,
                        tax_allocated=trade_record.tax_allocated,
                        tax_credit=trade_record.tax_credit,
                        reserve_after=trade_record.reserve_after,
                    )

                    # Reset position tracking
                    self.position_qty = 0.0
                    self.position_entry_price = None
                    self.position_peak_price = None
            else:
                logger.debug("SELL signal generated but no open position to liquidate.")

        elif signal_type == "HOLD":
            if position and self.position_peak_price:
                pct_drawdown = (self.position_peak_price - current_price) / self.position_peak_price
                logger.debug(
                    "Holding %s: Drawdown from peak: %.2f%% (Stop threshold: %.2f%%)",
                    self.config.TARGET_SYMBOL,
                    pct_drawdown * 100,
                    self.config.TRAILING_STOP_LOSS_PCT * 100,
                )

        # --- Strategy 2: Multi-Asset Option Wheel Execution ---
        if self.config.WHEEL_ENABLED:
            try:
                self.wheel_engine.step(total_cash=account.cash)
            except Exception as e:
                logger.exception("Error executing Multi-Asset Wheel Strategy cycle: %s", e)

    def start(self, max_cycles: Optional[int] = None) -> None:
        """Starts the daemon loop."""
        logger.info("==========================================================")
        logger.info("Starting Crypto Paper Trading Bot (Apple Silicon Edition)")
        logger.info("Target Symbol: %s | Interval: %ds | Paper Mode: %s", self.config.TARGET_SYMBOL, self.config.CYCLE_INTERVAL_SECONDS, self.config.ALPACA_PAPER)
        logger.info("Sentiment Provider: %s | Tax Escrow Rate: %.0f%%", self.config.SENTIMENT_PROVIDER, self.config.TAX_RATE * 100)
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
    if args.interval:
        config.CYCLE_INTERVAL_SECONDS = args.interval

    max_cycles = 1 if args.once else args.cycles
    daemon = TradingDaemon(config=config, dry_run=args.dry_run)
    daemon.start(max_cycles=max_cycles)


if __name__ == "__main__":
    main()
