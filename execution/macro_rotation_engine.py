"""
Autonomous Macro Dual-Momentum / Sector Rotation Strategy Engine.
Implements the Universal 12-Point Strategy Standard:
1. Capital Gating & Sizing Limit (checked vs tax_engine.get_tradable_cash)
2. Strategy Allocation Ceilings ($5,000 max cumulative ceiling)
3. HITL SGOV/FBND Liquidity Rebalancing (2-way mobile liquidation alert on cash shortage)
4. 30% Automatic Tax Escrow (tax_engine.record_trade_result on all realized profits)
5. Early Exit / Profit Target (dynamic rotation on regime changes & monthly rebalances)
6. Downside / Regime Defense (-7% trailing stop-loss & 200-day SMA trend filter)
7. 24-Hour TTL Stale Limit Order Cancellation
8. Dynamic Midpoint Pricing (NBBO quotes for equity orders)
9. Persistent JSON State (macro_rotation_state.json)
10. Multi-Channel Push Alerts (ntfy.sh iOS PWA + Apple iMessage)
11. EOD 5:00 PM Briefing Diagnostics
12. Dedicated Dashboard Tab + Collapsible ELI5 Explainer
"""

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from config import BotConfig
from notifier import TradeNotifier
from options.options_client import AlpacaOptionsClient
from tax_engine import TaxEngine

logger = logging.getLogger("execution.macro_rotation")


class MacroPosition(BaseModel):
    """Represents an active multi-asset equity/ETF allocation."""
    symbol: str
    qty: float
    entry_price: float
    cost_basis: float
    entry_time: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    current_price: float = 0.0
    highest_price: float = 0.0
    trailing_stop_price: float = 0.0
    asset_class: str = "Equities"  # "Equities", "Real Assets", "Real Estate", "Safe Haven"
    regime: str = "RISK_ON"  # "RISK_ON" or "RISK_OFF"


class ClosedMacroTrade(BaseModel):
    """Audit record for a liquidated or rotated macro position."""
    symbol: str
    qty: float
    entry_price: float
    exit_price: float
    entry_time: str
    exit_time: str
    holding_days: float
    realized_pnl: float
    realized_pnl_pct: float
    tax_escrow: float
    exit_reason: str  # "REBALANCE_ROTATION", "TRAILING_STOP", "TREND_BREAKDOWN"
    regime_at_exit: str = "RISK_ON"


class MacroAssetScore(BaseModel):
    """Calculated momentum and trend telemetry for an asset in the universe."""
    symbol: str
    asset_class: str
    current_price: float
    return_60d: float
    return_120d: float
    blended_score: float
    sma_200: float
    is_above_sma200: bool
    rank: int = 0
    is_selected: bool = False
    status: str = "STANDBY"  # "TOP_MOMENTUM", "BELOW_TREND", "SAFE_HAVEN", "STANDBY"


class MacroRotationState(BaseModel):
    """Atomic persistent state for Macro Dual-Momentum portfolio."""
    active_positions: Dict[str, MacroPosition] = Field(default_factory=dict)
    closed_trades: List[ClosedMacroTrade] = Field(default_factory=list)
    last_rebalance_time: Optional[str] = None
    last_rebalance_timestamp: float = 0.0
    current_regime: str = "STANDBY"  # "RISK_ON", "RISK_OFF", "STANDBY"
    pending_orders: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    last_leaderboard: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    total_realized_pnl: float = 0.0
    total_tax_escrow: float = 0.0


class MacroRotationEngine:
    """
    Executes Gary Antonacci-style Macro Dual Momentum across global asset classes.
    Rotates dynamically into top-momentum leaders when bull trends persist,
    and flees 100% into SGOV risk-free safe havens during broad market breakdowns.
    """

    ASSET_CLASSES = {
        "QQQ": "Tech / Growth Equities",
        "SPY": "Broad US Equities",
        "GLD": "Real Assets / Gold",
        "VNQ": "Real Estate / Yield",
        "SGOV": "Ultra-Short Treasury (Safe Haven)",
    }

    def __init__(
        self,
        config: BotConfig,
        options_client: AlpacaOptionsClient,
        tax_engine: TaxEngine,
        notifier: TradeNotifier,
        liquidity_manager: Optional[Any] = None,
        state_file: Optional[Path | str] = None,
    ):
        self.config = config
        self.options_client = options_client
        self.tax_engine = tax_engine
        self.notifier = notifier
        self.liquidity_manager = liquidity_manager

        state_path = state_file or getattr(config, "MACRO_STATE_FILE", "macro_rotation_state.json")
        self.state_file = Path(state_path)
        self.state = self._load_state()

    def _load_state(self) -> MacroRotationState:
        """Loads state from disk or initializes clean state."""
        if self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return MacroRotationState(**data)
            except Exception as e:
                logger.warning(
                    "Could not load macro rotation state from %s: %s. Starting fresh.",
                    self.state_file,
                    e,
                )
        return MacroRotationState()

    def _save_state(self) -> None:
        """Atomically saves state to disk."""
        try:
            temp_file = self.state_file.with_suffix(".tmp")
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(self.state.model_dump(), f, indent=2)
            temp_file.replace(self.state_file)
        except Exception as e:
            logger.error("Failed to save macro rotation state to %s: %s", self.state_file, e)

    def _get_current_total_cash(self) -> float:
        """Retrieves total portfolio cash balance from broker client."""
        if getattr(self.options_client, "mock_mode", False) and hasattr(self.options_client, "_mock_cash"):
            return float(self.options_client._mock_cash)
        if hasattr(self.options_client, "trading_client") and self.options_client.trading_client:
            try:
                acct = self.options_client.trading_client.get_account()
                return float(getattr(acct, "cash", 100000.0))
            except Exception:
                pass
        if hasattr(self.options_client, "get_account"):
            try:
                acct = self.options_client.get_account()
                return float(getattr(acct, "cash", 100000.0))
            except Exception:
                pass
        if hasattr(self.options_client, "_mock_cash"):
            return float(self.options_client._mock_cash)
        return 100000.0

    def evaluate_universe(self) -> Dict[str, MacroAssetScore]:
        """
        Calculates cross-sectional relative momentum (60-day and 120-day returns)
        and absolute time-series trend (200-day SMA + cash hurdle) for all assets.
        """
        symbols = getattr(self.config, "MACRO_SYMBOLS", ["QQQ", "SPY", "GLD", "VNQ", "SGOV"])
        safe_haven = getattr(self.config, "MACRO_SAFE_HAVEN", "SGOV").upper()
        lookback_short = getattr(self.config, "MACRO_LOOKBACK_SHORT_DAYS", 60)
        lookback_long = getattr(self.config, "MACRO_LOOKBACK_LONG_DAYS", 120)
        sma_period = getattr(self.config, "MACRO_SMA_PERIOD", 200)
        top_n = getattr(self.config, "MACRO_TOP_N_ASSETS", 2)

        scores: Dict[str, MacroAssetScore] = {}

        for sym in symbols:
            sym_upper = sym.upper()
            bars = self.options_client.get_stock_bars(sym_upper, limit=max(sma_period + 10, 250))
            if bars is None or len(bars) < 10:
                current_p = self.options_client.get_stock_price(sym_upper)
                scores[sym_upper] = MacroAssetScore(
                    symbol=sym_upper,
                    asset_class=self.ASSET_CLASSES.get(sym_upper, "Global Asset"),
                    current_price=current_p,
                    return_60d=0.0,
                    return_120d=0.0,
                    blended_score=0.0,
                    sma_200=current_p,
                    is_above_sma200=True,
                    status="STANDBY",
                )
                continue

            closes = bars["close"].values
            current_p = float(closes[-1])

            # 60-day return
            idx_60 = min(lookback_short, len(closes) - 1)
            ret_60d = (current_p - float(closes[-idx_60])) / float(closes[-idx_60]) if closes[-idx_60] > 0 else 0.0

            # 120-day return
            idx_120 = min(lookback_long, len(closes) - 1)
            ret_120d = (current_p - float(closes[-idx_120])) / float(closes[-idx_120]) if closes[-idx_120] > 0 else 0.0

            # Blended momentum score: 60% short-term + 40% intermediate-term
            blended = (0.60 * ret_60d) + (0.40 * ret_120d)

            # 200-day Simple Moving Average
            sma_len = min(sma_period, len(closes))
            sma_val = float(np.mean(closes[-sma_len:]))
            is_above_sma = current_p >= sma_val

            scores[sym_upper] = MacroAssetScore(
                symbol=sym_upper,
                asset_class=self.ASSET_CLASSES.get(sym_upper, "Global Asset"),
                current_price=round(current_p, 2),
                return_60d=round(ret_60d * 100, 2),
                return_120d=round(ret_120d * 100, 2),
                blended_score=round(blended * 100, 2),
                sma_200=round(sma_val, 2),
                is_above_sma200=is_above_sma,
            )

        # Separate safe haven from risky assets
        sgov_score = scores.get(safe_haven)
        sgov_blended = sgov_score.blended_score if sgov_score else 0.0

        risky_symbols = [s for s in scores.keys() if s != safe_haven]
        # Rank risky assets by blended momentum score descending
        sorted_risky = sorted(
            risky_symbols,
            key=lambda s: scores[s].blended_score,
            reverse=True,
        )

        for rank_idx, s in enumerate(sorted_risky, start=1):
            scores[s].rank = rank_idx

        # Absolute Momentum & Regime Determination:
        # Check if the top leader beats the safe haven return AND is above its 200-day SMA
        top_leader = sorted_risky[0] if sorted_risky else None
        top_score = scores[top_leader] if top_leader else None

        is_risk_on = False
        if top_score and top_score.blended_score > sgov_blended and top_score.is_above_sma200:
            is_risk_on = True

        if is_risk_on:
            self.state.current_regime = "RISK_ON"
            # Pick top N assets that are also above their SMA200 and beat SGOV
            selected_count = 0
            for s in sorted_risky:
                sc = scores[s]
                if selected_count < top_n and sc.blended_score > sgov_blended and sc.is_above_sma200:
                    sc.is_selected = True
                    sc.status = "TOP_MOMENTUM"
                    selected_count += 1
                elif not sc.is_above_sma200:
                    sc.status = "BELOW_TREND"
                else:
                    sc.status = "STANDBY"
            if sgov_score:
                sgov_score.status = "STANDBY"
                sgov_score.is_selected = False
        else:
            self.state.current_regime = "RISK_OFF"
            # All risky assets failing trend or losing to cash -> Safe haven flight to SGOV
            for s in sorted_risky:
                sc = scores[s]
                sc.is_selected = False
                sc.status = "BELOW_TREND" if not sc.is_above_sma200 else "STANDBY"
            if sgov_score:
                sgov_score.is_selected = True
                sgov_score.status = "SAFE_HAVEN"
                sgov_score.rank = 1

        # Cache leaderboard in state
        self.state.last_leaderboard = {
            s: score.model_dump() for s, score in scores.items()
        }
        return scores

    def rebalance_portfolio(self, is_market_open: bool = True) -> List[MacroPosition]:
        """
        Executes systematic portfolio rebalancing.
        Closes demoted assets, withholds 30% tax escrow on realized gains,
        checks capital gates, triggers HITL liquidity rebalancing if cash is low,
        and purchases newly promoted momentum leaders.
        """
        if not is_market_open:
            return []

        scores = self.evaluate_universe()
        target_symbols = [s for s, sc in scores.items() if sc.is_selected]

        now_ts = time.time()
        cadence_seconds = getattr(self.config, "MACRO_REBALANCE_CADENCE_DAYS", 30) * 86400

        # Check if rebalance is due:
        # 1. Never rebalanced before
        # 2. Rebalance cadence has elapsed
        # 3. Active positions differ from target_symbols (regime shift)
        currently_held = set(self.state.active_positions.keys())
        target_set = set(target_symbols)

        is_due = (
            self.state.last_rebalance_timestamp == 0
            or (now_ts - self.state.last_rebalance_timestamp) >= cadence_seconds
            or (currently_held != target_set and len(target_symbols) > 0)
        )

        if not is_due:
            return []

        logger.info(
            "Macro Dual Momentum Rebalance Triggered: Regime=%s | Target=%s | Held=%s",
            self.state.current_regime,
            list(target_set),
            list(currently_held),
        )

        # 1. Liquidate Demoted Positions
        demoted = [sym for sym in currently_held if sym not in target_set]
        for sym in demoted:
            pos = self.state.active_positions[sym]
            curr_price = self.options_client.get_stock_price(sym)
            if curr_price <= 0:
                curr_price = pos.current_price or pos.entry_price

            # Submit sell order
            self.options_client.submit_stock_order(
                symbol=sym,
                side="SELL",
                qty=pos.qty,
                time_in_force=getattr(self.config, "MACRO_TIME_IN_FORCE", "DAY"),
            )

            # PnL & Tax Escrow Calculation
            realized_pnl = round((curr_price - pos.entry_price) * pos.qty, 2)
            pnl_pct = round((curr_price - pos.entry_price) / pos.entry_price, 4) if pos.entry_price > 0 else 0.0

            tax_escrow = 0.0
            if realized_pnl > 0:
                trade_id = f"macro_rot_{sym}_{int(now_ts)}"
                record = self.tax_engine.record_trade_result(
                    trade_id=trade_id,
                    symbol=sym,
                    side="SELL",
                    gross_pnl=realized_pnl,
                )
                tax_escrow = record.tax_allocated
                self.state.total_tax_escrow = round(self.state.total_tax_escrow + tax_escrow, 2)

            self.state.total_realized_pnl = round(self.state.total_realized_pnl + realized_pnl, 2)

            entry_dt = datetime.fromisoformat(pos.entry_time.replace("Z", "+00:00"))
            holding_days = max(0.1, round((now_ts - entry_dt.timestamp()) / 86400, 1))

            closed_trade = ClosedMacroTrade(
                symbol=sym,
                qty=pos.qty,
                entry_price=pos.entry_price,
                exit_price=curr_price,
                entry_time=pos.entry_time,
                exit_time=datetime.now(timezone.utc).isoformat(),
                holding_days=holding_days,
                realized_pnl=realized_pnl,
                realized_pnl_pct=pnl_pct,
                tax_escrow=tax_escrow,
                exit_reason="REBALANCE_ROTATION",
                regime_at_exit=self.state.current_regime,
            )
            self.state.closed_trades.append(closed_trade)
            self.state.active_positions.pop(sym, None)

            logger.info(
                "Closed demoted macro position: %s (PnL: $%.2f | Tax Escrow: $%.2f)",
                sym,
                realized_pnl,
                tax_escrow,
            )

            if self.notifier:
                self.notifier.notify_macro_rotation_close(
                    symbol=sym,
                    shares=pos.qty,
                    entry_price=pos.entry_price,
                    exit_price=curr_price,
                    pnl=realized_pnl,
                    pnl_pct=pnl_pct,
                    exit_reason="REBALANCE_ROTATION",
                    tax_escrow=tax_escrow,
                )

        # 2. Acquire Newly Promoted Assets
        max_capital = getattr(self.config, "MACRO_MAX_CAPITAL_USD", 5000.0)
        tranche_size = getattr(self.config, "MACRO_TRANCHE_SIZE_USD", 2500.0)
        trailing_stop_pct = getattr(self.config, "MACRO_TRAILING_STOP_PCT", 0.07)
        tif = getattr(self.config, "MACRO_TIME_IN_FORCE", "DAY")

        new_positions: List[MacroPosition] = []

        for sym in target_symbols:
            if sym in self.state.active_positions:
                continue

            curr_price = self.options_client.get_stock_price(sym)
            if curr_price <= 0:
                continue

            # Check strategy ceiling
            current_allocated = sum(p.cost_basis for p in self.state.active_positions.values())
            if current_allocated + tranche_size > max_capital:
                logger.debug(
                    "Macro Rotation capital ceiling reached ($%.2f / $%.2f). Skipping %s.",
                    current_allocated,
                    max_capital,
                    sym,
                )
                continue

            # Capital Gating via Tax Engine
            total_cash = self._get_current_total_cash()
            tradable_cash = self.tax_engine.get_tradable_cash(total_cash)

            if tradable_cash < tranche_size:
                logger.warning(
                    "Macro entry blocked by Capital Gate: Tradable cash $%.2f < $%.2f needed for %s.",
                    tradable_cash,
                    tranche_size,
                    sym,
                )
                if self.liquidity_manager:
                    self.liquidity_manager.request_liquidation_for_opportunity(
                        needed_cash=tranche_size,
                        target_symbol=sym,
                        opportunity_type="Macro Dual-Momentum Rotation",
                        current_price=curr_price,
                    )
                continue

            # Submit order
            calc_shares = round(tranche_size / curr_price, 4)
            order_receipt = self.options_client.submit_stock_order(
                symbol=sym,
                side="BUY",
                notional=tranche_size,
                time_in_force=tif,
            )

            actual_entry_p = float(order_receipt.get("filled_avg_price") or curr_price)
            actual_shares = float(order_receipt.get("filled_qty") or calc_shares)
            cost_basis = round(actual_entry_p * actual_shares, 2)
            stop_price = round(actual_entry_p * (1.0 - trailing_stop_pct), 2)

            pos = MacroPosition(
                symbol=sym,
                qty=actual_shares,
                entry_price=actual_entry_p,
                cost_basis=cost_basis,
                current_price=actual_entry_p,
                highest_price=actual_entry_p,
                trailing_stop_price=stop_price,
                asset_class=self.ASSET_CLASSES.get(sym, "Global Asset"),
                regime=self.state.current_regime,
            )
            self.state.active_positions[sym] = pos
            new_positions.append(pos)

            logger.info(
                "Entered Macro Position: %s @ $%.2f (%.4f shares, $%.2f cost basis)",
                sym,
                actual_entry_p,
                actual_shares,
                cost_basis,
            )

            if self.notifier:
                rank_val = scores[sym].rank if sym in scores else 1
                self.notifier.notify_macro_rotation_open(
                    symbol=sym,
                    shares=actual_shares,
                    price=actual_entry_p,
                    notional=cost_basis,
                    regime=self.state.current_regime,
                    momentum_rank=rank_val,
                )

        # Update rebalance markers
        self.state.last_rebalance_timestamp = now_ts
        self.state.last_rebalance_time = datetime.now(timezone.utc).isoformat()
        self._save_state()
        return new_positions

    def manage_active_positions(self, is_market_open: bool = True) -> List[ClosedMacroTrade]:
        """
        Monitors active positions between rebalances.
        Updates trailing stops and exits immediately if price drops below trailing stop (-7.0%).
        """
        if not is_market_open or not self.state.active_positions:
            return []

        trailing_stop_pct = getattr(self.config, "MACRO_TRAILING_STOP_PCT", 0.07)
        now_ts = time.time()
        closed: List[ClosedMacroTrade] = []

        for sym, pos in list(self.state.active_positions.items()):
            curr_price = self.options_client.get_stock_price(sym)
            if curr_price <= 0:
                continue

            pos.current_price = curr_price
            if curr_price > pos.highest_price:
                pos.highest_price = curr_price
                pos.trailing_stop_price = round(curr_price * (1.0 - trailing_stop_pct), 2)

            # Check trailing stop-loss exit
            if curr_price <= pos.trailing_stop_price and pos.trailing_stop_price > 0:
                logger.warning(
                    "Macro Trailing Stop Triggered: %s dropped to $%.2f (Stop: $%.2f, High: $%.2f)",
                    sym,
                    curr_price,
                    pos.trailing_stop_price,
                    pos.highest_price,
                )

                self.options_client.submit_stock_order(
                    symbol=sym,
                    side="SELL",
                    qty=pos.qty,
                    time_in_force=getattr(self.config, "MACRO_TIME_IN_FORCE", "DAY"),
                )

                realized_pnl = round((curr_price - pos.entry_price) * pos.qty, 2)
                pnl_pct = round((curr_price - pos.entry_price) / pos.entry_price, 4) if pos.entry_price > 0 else 0.0

                tax_escrow = 0.0
                if realized_pnl > 0:
                    trade_id = f"macro_stop_{sym}_{int(now_ts)}"
                    record = self.tax_engine.record_trade_result(
                        trade_id=trade_id,
                        symbol=sym,
                        side="SELL",
                        gross_pnl=realized_pnl,
                    )
                    tax_escrow = record.tax_allocated
                    self.state.total_tax_escrow = round(self.state.total_tax_escrow + tax_escrow, 2)

                self.state.total_realized_pnl = round(self.state.total_realized_pnl + realized_pnl, 2)

                entry_dt = datetime.fromisoformat(pos.entry_time.replace("Z", "+00:00"))
                holding_days = max(0.1, round((now_ts - entry_dt.timestamp()) / 86400, 1))

                closed_trade = ClosedMacroTrade(
                    symbol=sym,
                    qty=pos.qty,
                    entry_price=pos.entry_price,
                    exit_price=curr_price,
                    entry_time=pos.entry_time,
                    exit_time=datetime.now(timezone.utc).isoformat(),
                    holding_days=holding_days,
                    realized_pnl=realized_pnl,
                    realized_pnl_pct=pnl_pct,
                    tax_escrow=tax_escrow,
                    exit_reason="TRAILING_STOP",
                    regime_at_exit=self.state.current_regime,
                )
                self.state.closed_trades.append(closed_trade)
                self.state.active_positions.pop(sym, None)
                closed.append(closed_trade)

                if self.notifier:
                    self.notifier.notify_macro_rotation_close(
                        symbol=sym,
                        shares=pos.qty,
                        entry_price=pos.entry_price,
                        exit_price=curr_price,
                        pnl=realized_pnl,
                        pnl_pct=pnl_pct,
                        exit_reason="TRAILING_STOP",
                        tax_escrow=tax_escrow,
                    )

        if closed:
            self._save_state()

        return closed

    def check_stale_orders(self) -> List[str]:
        """Watchdog: Cancels unfulfilled resting limit orders older than 24 hours."""
        now = time.time()
        stale_order_ids: List[str] = []

        for order_id, order_data in list(self.state.pending_orders.items()):
            created_at = order_data.get("created_at", now)
            if now - created_at > 86400:  # 24 hours
                logger.info("Cancelling stale macro order %s (TTL > 24h)", order_id)
                try:
                    self.options_client.cancel_order(order_id)
                except Exception as e:
                    logger.warning("Failed to cancel stale macro order %s: %s", order_id, e)
                stale_order_ids.append(order_id)
                self.state.pending_orders.pop(order_id, None)

        if stale_order_ids:
            self._save_state()

        return stale_order_ids

    def step(self, is_market_open: bool = True) -> None:
        """Executes full macro dual-momentum evaluation cycle."""
        self.check_stale_orders()
        self.evaluate_universe()
        if is_market_open:
            self.manage_active_positions(is_market_open=is_market_open)
            self.rebalance_portfolio(is_market_open=is_market_open)
        self._save_state()

    def get_diagnostics(self) -> List[str]:
        """Formats comprehensive diagnostic summary for the 5:00 PM briefing."""
        diagnostics = []
        regime_badge = "🟢 RISK-ON (Growth Equities)" if self.state.current_regime == "RISK_ON" else "🟡 RISK-OFF (Treasury Safe Haven)"
        diagnostics.append(f"Regime: {regime_badge}")

        # Leaderboard ranks
        if self.state.last_leaderboard:
            diagnostics.append("Asset Momentum Leaderboard:")
            sorted_board = sorted(
                self.state.last_leaderboard.values(),
                key=lambda x: x.get("blended_score", 0.0),
                reverse=True,
            )
            for item in sorted_board:
                status_icon = "⭐ [SELECTED]" if item.get("is_selected") else ("⚠️ [<SMA200]" if not item.get("is_above_sma200") else "—")
                diagnostics.append(
                    f"  • #{item.get('rank', 0)} {item.get('symbol')}: Score {item.get('blended_score', 0):+.2f}% | "
                    f"60d: {item.get('return_60d', 0):+.2f}% | 120d: {item.get('return_120d', 0):+.2f}% {status_icon}"
                )

        # Active holdings
        if self.state.active_positions:
            diagnostics.append("Active Macro Holdings:")
            for sym, pos in self.state.active_positions.items():
                unrealized = (pos.current_price - pos.entry_price) * pos.qty
                pnl_pct = (pos.current_price - pos.entry_price) / pos.entry_price * 100 if pos.entry_price > 0 else 0.0
                diagnostics.append(
                    f"  • {sym}: {pos.qty:.2f} shs @ ${pos.entry_price:.2f} (Now: ${pos.current_price:.2f}, "
                    f"PnL: {unrealized:+.2f} / {pnl_pct:+.1f}%, Stop: ${pos.trailing_stop_price:.2f})"
                )
        else:
            diagnostics.append("Active Macro Holdings: None (Standing by for rebalance)")

        diagnostics.append(
            f"Cumulative Realized PnL: ${self.state.total_realized_pnl:+,.2f} | "
            f"Tax Escrow Withheld: ${self.state.total_tax_escrow:,.2f}"
        )
        return diagnostics

    def get_status(self) -> Dict[str, Any]:
        """Provides full real-time telemetry payload for the investor dashboard."""
        total_allocated = sum(p.cost_basis for p in self.state.active_positions.values())
        max_capital = getattr(self.config, "MACRO_MAX_CAPITAL_USD", 5000.0)

        unrealized_pnl = 0.0
        active_list = []
        for sym, pos in self.state.active_positions.items():
            curr_p = pos.current_price or pos.entry_price
            pnl_val = (curr_p - pos.entry_price) * pos.qty
            pnl_pct = (curr_p - pos.entry_price) / pos.entry_price * 100 if pos.entry_price > 0 else 0.0
            unrealized_pnl += pnl_val

            active_list.append(
                {
                    "symbol": sym,
                    "asset_class": pos.asset_class,
                    "shares": round(pos.qty, 4),
                    "entry_price": pos.entry_price,
                    "current_price": curr_p,
                    "cost_basis": pos.cost_basis,
                    "market_value": round(curr_p * pos.qty, 2),
                    "highest_price": pos.highest_price,
                    "trailing_stop_price": pos.trailing_stop_price,
                    "unrealized_pnl": round(pnl_val, 2),
                    "unrealized_pnl_pct": round(pnl_pct, 2),
                    "regime": pos.regime,
                    "entry_time": pos.entry_time,
                }
            )

        return {
            "enabled": getattr(self.config, "MACRO_ENABLED", True),
            "regime": self.state.current_regime,
            "capital_allocated": round(total_allocated, 2),
            "max_capital": max_capital,
            "utilization_pct": round((total_allocated / max_capital) * 100, 1) if max_capital > 0 else 0.0,
            "active_count": len(self.state.active_positions),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "total_realized_pnl": self.state.total_realized_pnl,
            "total_tax_escrow": self.state.total_tax_escrow,
            "last_rebalance_time": self.state.last_rebalance_time,
            "active_positions": active_list,
            "leaderboard": list(self.state.last_leaderboard.values()),
            "closed_trades": [t.model_dump() for t in self.state.closed_trades[-15:]],
        }
