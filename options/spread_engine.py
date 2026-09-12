"""
Defined-Risk Option Spreads Strategy Engine for Index ETFs (SPY, QQQ, IWM).
Implements Bull Put Credit Spreads with strictly capped collateral ($500 max per contract):
1. Out-of-the-money strike pair discovery (~0.20 delta short put + $5-wide protective long put).
2. Atomic multi-leg order execution via Alpaca MLEG (OrderClass.MLEG).
3. 50% Max-Profit Target Buy-to-Close early exit rule.
4. Stop-loss guard (2.5x credit) and 5 DTE expiration defense to prevent assignment.
5. Strict coordination with Virtual Tax Escrow Engine (30% withholding) and push alerts.
6. Persistent state synchronization to spreads_state.json.
"""

import enum
import json
import logging
import os
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from config import BotConfig
from notifier import TradeNotifier
from options.options_client import AlpacaOptionsClient, OptionContractInfo
from tax_engine import TaxEngine

logger = logging.getLogger("options.spreads")


class SpreadState(str, enum.Enum):
    """Lifecycle states of a credit spread."""

    DISCOVERY = "DISCOVERY"
    PENDING_ENTRY = "PENDING_ENTRY"
    ACTIVE = "ACTIVE"
    PENDING_EXIT = "PENDING_EXIT"
    CLOSED = "CLOSED"


class SpreadPosition(BaseModel):
    """Active credit spread position tracking model."""

    id: str
    underlying: str
    short_symbol: str
    long_symbol: str
    short_strike: float
    long_strike: float
    width: float = 5.0
    qty: int = 1
    expiration_date: str
    days_to_expiration: int
    entry_credit: float
    collateral_locked: float
    max_profit: float
    max_loss: float
    entry_timestamp: str
    current_value: float = 0.0
    unrealized_pnl: float = 0.0
    progress_to_target: float = 0.0
    order_id: Optional[str] = None
    exit_order_id: Optional[str] = None
    status: str = "ACTIVE"
    close_timestamp: Optional[str] = None
    realized_pnl: Optional[float] = None


class SpreadEngine:
    """
    Autonomous Defined-Risk Credit Spread engine for a single underlying symbol.
    """

    def __init__(
        self,
        config: BotConfig,
        options_client: AlpacaOptionsClient,
        tax_engine: TaxEngine,
        notifier: TradeNotifier,
        symbol: str = "SPY",
        liquidity_manager: Optional[Any] = None,
    ):
        self.config = config
        self.client = options_client
        self.tax_engine = tax_engine
        self.notifier = notifier
        self.symbol = symbol.upper()
        self.liquidity_manager = liquidity_manager

        # Active position tracking
        self.active_spread: Optional[SpreadPosition] = None
        self.pending_order_id: Optional[str] = None
        self.pending_order_time: Optional[datetime] = None

    def select_spread_contracts(
        self, current_price: float
    ) -> Optional[Dict[str, OptionContractInfo]]:
        """
        Discovers the optimal Bull Put Credit Spread contract pair:
        - Target DTE: 21 to 45 days
        - Short Put: ~3% to 5% OTM (~0.20 delta)
        - Long Put: Strike exactly config.SPREAD_WIDTH_USD below Short Put
        """
        put_contracts = self.client.get_option_contracts(
            underlying=self.symbol,
            contract_type="put",
            min_dte=self.config.SPREAD_TARGET_DTE_MIN,
            max_dte=self.config.SPREAD_TARGET_DTE_MAX,
        )

        if not put_contracts:
            logger.debug("No active put contracts found for %s in target DTE window.", self.symbol)
            return None

        # Group contracts by expiration date
        by_expiry: Dict[str, List[OptionContractInfo]] = {}
        for c in put_contracts:
            by_expiry.setdefault(c.expiration_date, []).append(c)

        target_short_strike = current_price * (1.0 - self.config.SPREAD_TARGET_DELTA)
        width = self.config.SPREAD_WIDTH_USD

        best_pair = None
        best_diff = float("inf")

        # Find the best expiration date and strike pair
        for exp_date, contracts in by_expiry.items():
            contracts_by_strike = {round(c.strike_price, 2): c for c in contracts}
            otm_strikes = sorted([s for s in contracts_by_strike if s < current_price], reverse=True)

            for s_strike in otm_strikes:
                l_strike = round(s_strike - width, 2)
                if l_strike in contracts_by_strike:
                    diff = abs(s_strike - target_short_strike)
                    if diff < best_diff:
                        best_diff = diff
                        best_pair = {
                            "short": contracts_by_strike[s_strike],
                            "long": contracts_by_strike[l_strike],
                        }

        return best_pair

    def step(self, current_spread_collateral: float = 0.0) -> Dict[str, Any]:
        """
        Main execution step for this underlying asset:
        1. If active spread exists: monitor current price toward 50% profit exit or stop loss.
        2. If pending order exists: monitor fill status.
        3. If no active spread: search for high-conviction credit spread opportunities.
        """
        current_price = self.client.get_stock_price(self.symbol)
        now_dt = datetime.now(timezone.utc)

        # 1. Monitor Active Position
        if self.active_spread and self.active_spread.status == "ACTIVE":
            return self._monitor_active_spread(current_price)

        # 2. Check Pending Entry Order
        if self.pending_order_id:
            return self._check_pending_order(now_dt)

        # 3. Discover New Spread Opportunity
        return self._evaluate_new_spread(current_price, current_spread_collateral)

    def _monitor_active_spread(self, current_price: float) -> Dict[str, Any]:
        """Monitors an active spread for 50% profit target or risk exit."""
        spread = self.active_spread
        quote = self.client.get_spread_quote(spread.short_symbol, spread.long_symbol)

        # Calculate cost to buy back the spread
        close_cost_per_share = quote.get("close_debit", 0.0) if quote else 0.40
        current_spread_value = close_cost_per_share * 100 * spread.qty
        initial_credit_total = spread.entry_credit * 100 * spread.qty

        # Profit captured so far
        profit_captured = initial_credit_total - current_spread_value
        unrealized_pnl = round(profit_captured, 2)
        profit_ratio = profit_captured / max(1.0, initial_credit_total)
        progress_to_target = min(1.0, max(0.0, profit_ratio / self.config.SPREAD_PROFIT_TARGET_PCT))

        spread.current_value = round(current_spread_value, 2)
        spread.unrealized_pnl = unrealized_pnl
        spread.progress_to_target = round(progress_to_target, 3)

        # --- Trigger 1: 50% Profit Target Early Exit ---
        target_debit = spread.entry_credit * (1.0 - self.config.SPREAD_PROFIT_TARGET_PCT)
        if close_cost_per_share <= target_debit:
            logger.info(
                "🎯 [SPREAD] %s hit 50%% Profit Target! (Close debit: $%0.2f <= target: $%0.2f). Exiting early.",
                self.symbol,
                close_cost_per_share,
                target_debit,
            )
            return self._execute_spread_exit(
                reason="PROFIT_TARGET_50_PCT",
                debit_price=close_cost_per_share,
                realized_pnl=unrealized_pnl,
            )

        # --- Trigger 2: Stop-Loss Guard (2.5x initial credit) ---
        max_loss_debit = spread.entry_credit * self.config.SPREAD_STOP_LOSS_RATIO
        if close_cost_per_share >= max_loss_debit:
            logger.warning(
                "🛑 [SPREAD] %s hit Stop-Loss Guard! (Debit $%0.2f >= $%0.2f). Aborting position.",
                self.symbol,
                close_cost_per_share,
                max_loss_debit,
            )
            return self._execute_spread_exit(
                reason="STOP_LOSS_GUARD",
                debit_price=close_cost_per_share,
                realized_pnl=unrealized_pnl,
            )

        # --- Trigger 3: 5 DTE Expiration Safety ---
        today = date.today()
        exp_date = datetime.strptime(spread.expiration_date, "%Y-%m-%d").date()
        dte = (exp_date - today).days
        spread.days_to_expiration = dte

        if dte <= 5 and current_price <= spread.short_strike:
            logger.warning(
                "⚠️ [SPREAD] %s has %d DTE and is ITM. Closing early to eliminate assignment risk.",
                self.symbol,
                dte,
            )
            return self._execute_spread_exit(
                reason="DTE_ASSIGNMENT_DEFENSE",
                debit_price=close_cost_per_share,
                realized_pnl=unrealized_pnl,
            )

        return {
            "status": "MONITORING",
            "symbol": self.symbol,
            "unrealized_pnl": unrealized_pnl,
            "progress_to_target": progress_to_target,
            "dte": dte,
        }

    def _execute_spread_exit(
        self, reason: str, debit_price: float, realized_pnl: float
    ) -> Dict[str, Any]:
        """Closes the active spread and records realized profit in the TaxEngine."""
        spread = self.active_spread
        close_res = self.client.close_mleg_spread_order(
            short_symbol=spread.short_symbol,
            long_symbol=spread.long_symbol,
            qty=spread.qty,
            limit_debit=debit_price,
            time_in_force=self.config.SPREAD_TIME_IN_FORCE,
        )

        spread.status = "CLOSED"
        spread.close_timestamp = datetime.now(timezone.utc).isoformat()
        spread.realized_pnl = realized_pnl
        spread.exit_order_id = close_res.get("id")

        # Record realized profit with 30% tax segregation
        self.tax_engine.record_trade_result(
            trade_id=close_res.get("id", f"spread_close_{self.symbol}"),
            symbol=f"{self.symbol} Spread",
            side="BUY_TO_CLOSE",
            gross_pnl=realized_pnl,
        )

        # Push notification
        tax_rate = getattr(self.config, "TAX_RATE", 0.30)
        tax_allocated = (realized_pnl * tax_rate) if realized_pnl > 0 else 0.0
        if hasattr(self.notifier, "notify_spread_close"):
            self.notifier.notify_spread_close(
                underlying=self.symbol,
                short_strike=spread.short_strike,
                long_strike=spread.long_strike,
                reason=reason,
                realized_pnl=realized_pnl,
                tax_allocated=tax_allocated,
                tax_reserve_after=self.tax_engine.tax_reserve,
            )
        else:
            sign = "+" if realized_pnl >= 0 else "-"
            abs_pnl = abs(realized_pnl)
            self.notifier.notify(
                f"🎯 [SPREAD] {self.symbol} Closed ({reason})! PnL: {sign}${abs_pnl:.2f}. "
                f"Tax Reserve Balance: ${self.tax_engine.tax_reserve:.2f}"
            )

        closed_spread = self.active_spread
        self.active_spread = None

        return {
            "status": "CLOSED",
            "reason": reason,
            "realized_pnl": realized_pnl,
            "spread": closed_spread.model_dump(),
        }

    def _evaluate_new_spread(
        self, current_price: float, current_spread_collateral: float
    ) -> Dict[str, Any]:
        """Evaluates entry criteria and places a new multi-leg credit spread order."""
        # 1. Check portfolio capital budget
        collateral_needed = self.config.SPREAD_WIDTH_USD * 100 * self.config.SPREAD_ORDER_QTY
        if current_spread_collateral + collateral_needed > self.config.SPREAD_MAX_CAPITAL_USD:
            logger.debug(
                "Skipping %s spread: total collateral ($%0.2f) would exceed max ($%0.2f)",
                self.symbol,
                current_spread_collateral + collateral_needed,
                self.config.SPREAD_MAX_CAPITAL_USD,
            )
            return {"status": "BUDGET_CAPPED"}

        # 2. Check TaxEngine Hard Capital Gate
        tradable_cash = self.tax_engine.get_tradable_cash(100000.0)
        if collateral_needed > tradable_cash:
            needed_capital = collateral_needed - tradable_cash
            logger.warning(
                "Capital Gate: Collateral ($%0.2f) exceeds tradable cash ($%0.2f)",
                collateral_needed,
                tradable_cash,
            )
            if self.liquidity_manager and hasattr(self.liquidity_manager, "request_liquidation_for_opportunity"):
                self.liquidity_manager.request_liquidation_for_opportunity(
                    needed_cash=round(needed_capital, 2),
                    target_symbol=self.symbol,
                    opportunity_type=f"Defined-Risk Spread Collateral ({self.symbol})",
                    current_price=current_price,
                    reserve_symbol="SGOV",
                )
            return {"status": "CAPITAL_GATE_REJECTED"}

        # 3. Select optimal contract pair
        pair = self.select_spread_contracts(current_price)
        if not pair:
            return {"status": "NO_CONTRACT_PAIR"}

        short_contract = pair["short"]
        long_contract = pair["long"]

        # 4. Fetch live quotes to ensure minimum credit
        quote = self.client.get_spread_quote(short_contract.symbol, long_contract.symbol)
        credit = quote.get("entry_credit", 0.0) if quote else 0.85
        if credit < self.config.SPREAD_MIN_CREDIT_USD:
            # Fallback to mid credit if bid-ask spread is wide
            credit = quote.get("mid_credit", 0.0) if quote else 0.85

        if credit < self.config.SPREAD_MIN_CREDIT_USD:
            logger.debug(
                "Spread for %s offers $%0.2f credit, below minimum $%0.2f",
                self.symbol,
                credit,
                self.config.SPREAD_MIN_CREDIT_USD,
            )
            return {"status": "CREDIT_TOO_LOW"}

        # 5. Submit MLEG Order
        order_res = self.client.submit_mleg_spread_order(
            short_symbol=short_contract.symbol,
            long_symbol=long_contract.symbol,
            qty=self.config.SPREAD_ORDER_QTY,
            limit_credit=credit,
            time_in_force=self.config.SPREAD_TIME_IN_FORCE,
        )

        max_profit = round(credit * 100 * self.config.SPREAD_ORDER_QTY, 2)
        max_loss = round((self.config.SPREAD_WIDTH_USD - credit) * 100 * self.config.SPREAD_ORDER_QTY, 2)

        spread_pos = SpreadPosition(
            id=order_res.get("id", f"sprd_{self.symbol}"),
            underlying=self.symbol,
            short_symbol=short_contract.symbol,
            long_symbol=long_contract.symbol,
            short_strike=short_contract.strike_price,
            long_strike=long_contract.strike_price,
            width=self.config.SPREAD_WIDTH_USD,
            qty=self.config.SPREAD_ORDER_QTY,
            expiration_date=short_contract.expiration_date,
            days_to_expiration=short_contract.days_to_expiration,
            entry_credit=credit,
            collateral_locked=collateral_needed,
            max_profit=max_profit,
            max_loss=max_loss,
            entry_timestamp=datetime.now(timezone.utc).isoformat(),
            current_value=max_profit,
            order_id=order_res.get("id"),
            status="ACTIVE",
        )

        self.active_spread = spread_pos

        target_profit = round(max_profit * self.config.SPREAD_PROFIT_TARGET_PCT, 2)
        if hasattr(self.notifier, "notify_spread_open"):
            self.notifier.notify_spread_open(
                underlying=self.symbol,
                short_strike=short_contract.strike_price,
                long_strike=long_contract.strike_price,
                expiration=short_contract.expiration_date,
                dte=short_contract.days_to_expiration,
                net_credit=credit,
                collateral_locked=collateral_needed,
                max_profit=max_profit,
                target_exit_profit=target_profit,
            )
        else:
            self.notifier.notify(
                f"⚡ [SPREAD] New Bull Put Spread Opened on {self.symbol}!\n"
                f"• Strikes: ${short_contract.strike_price:0.2f}P / ${long_contract.strike_price:0.2f}P\n"
                f"• Net Credit: +${credit:.2f}/sh (+${max_profit:.2f} total)\n"
                f"• Collateral Locked: ${collateral_needed:.2f}\n"
                f"• 50% Profit Exit Target: +${max_profit * 0.50:.2f}"
            )

        return {
            "status": "OPENED",
            "symbol": self.symbol,
            "credit": credit,
            "collateral": collateral_needed,
            "spread": spread_pos.model_dump(),
        }

    def _check_pending_order(self, now_dt: datetime) -> Dict[str, Any]:
        """Checks if a pending order should be cancelled due to TTL."""
        if not self.pending_order_time:
            self.pending_order_id = None
            return {"status": "PENDING_CLEARED"}

        elapsed_hours = (now_dt - self.pending_order_time).total_seconds() / 3600.0
        if elapsed_hours >= 24.0:
            logger.info("Pending spread order %s expired 24h TTL. Cancelling.", self.pending_order_id)
            self.client.cancel_order(self.pending_order_id)
            self.pending_order_id = None
            self.pending_order_time = None
            return {"status": "CANCELLED_TTL"}

        return {"status": "PENDING", "order_id": self.pending_order_id}


class SpreadPortfolioManager:
    """
    Coordinates Defined-Risk Option Spreads across multiple index ETFs (SPY, QQQ, IWM).
    Handles persistent JSON state storage and aggregate metrics.
    """

    def __init__(
        self,
        config: BotConfig,
        options_client: AlpacaOptionsClient,
        tax_engine: TaxEngine,
        notifier: TradeNotifier,
        liquidity_manager: Optional[Any] = None,
    ):
        self.config = config
        self.client = options_client
        self.tax_engine = tax_engine
        self.notifier = notifier
        self.liquidity_manager = liquidity_manager
        self.state_file = config.SPREAD_STATE_FILE

        # Initialize per-symbol engines
        symbols = (
            config.SPREAD_SYMBOLS
            if isinstance(config.SPREAD_SYMBOLS, list)
            else [s.strip().upper() for s in str(config.SPREAD_SYMBOLS).split(",") if s.strip()]
        )
        self.engines: Dict[str, SpreadEngine] = {
            sym: SpreadEngine(
                config=config,
                options_client=options_client,
                tax_engine=tax_engine,
                notifier=notifier,
                symbol=sym,
                liquidity_manager=liquidity_manager,
            )
            for sym in symbols
        }

        # Historical closed spreads
        self.closed_spreads: List[Dict[str, Any]] = []

        # Load persisted state
        self.load_state()

    def load_state(self) -> None:
        """Loads active and closed spread positions from persistent JSON state file."""
        if not os.path.exists(self.state_file):
            logger.info("No spread state file found at %s. Initializing fresh.", self.state_file)
            return

        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            active_list = data.get("active_spreads", [])
            for item in active_list:
                sym = item.get("underlying")
                if sym in self.engines:
                    self.engines[sym].active_spread = SpreadPosition(**item)

            self.closed_spreads = data.get("closed_spreads", [])
            logger.info(
                "Loaded %d active spreads and %d closed spreads from %s",
                len([e for e in self.engines.values() if e.active_spread]),
                len(self.closed_spreads),
                self.state_file,
            )
        except Exception as e:
            logger.warning("Failed to load spread state from %s: %s", self.state_file, e)

    def save_state(self) -> None:
        """Persists active and closed spreads to JSON state file."""
        try:
            active_spreads = [
                e.active_spread.model_dump()
                for e in self.engines.values()
                if e.active_spread
            ]
            payload = {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "active_spreads": active_spreads,
                "closed_spreads": self.closed_spreads[-50:],  # retain last 50
            }
            tmp_file = f"{self.state_file}.tmp"
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp_file, self.state_file)
        except Exception as e:
            logger.warning("Failed to save spread state to %s: %s", self.state_file, e)

    def step_all(self, is_market_open: bool = True) -> Dict[str, Any]:
        """
        Executes one evaluation step across all configured spread symbols.
        Only attempts new entries during market hours.
        """
        results: Dict[str, Any] = {}

        # Calculate current total collateral locked
        total_collateral = sum(
            e.active_spread.collateral_locked
            for e in self.engines.values()
            if e.active_spread and e.active_spread.status == "ACTIVE"
        )

        for sym, engine in self.engines.items():
            try:
                # If market is closed and no active spread, skip discovery
                if not is_market_open and not engine.active_spread:
                    continue

                res = engine.step(current_spread_collateral=total_collateral)
                results[sym] = res

                # Check if a position just closed
                if res.get("status") == "CLOSED" and "spread" in res:
                    self.closed_spreads.append(res["spread"])

            except Exception as e:
                logger.error("Error stepping spread engine for %s: %s", sym, e, exc_info=True)
                results[sym] = {"status": "ERROR", "error": str(e)}

        self.save_state()
        return results

    def get_summary_metrics(self) -> Dict[str, Any]:
        """Aggregates portfolio-level metrics for the live dashboard."""
        active = [
            e.active_spread.model_dump()
            for e in self.engines.values()
            if e.active_spread and e.active_spread.status == "ACTIVE"
        ]
        total_collateral = sum(item["collateral_locked"] for item in active)
        total_unrealized = sum(item["unrealized_pnl"] for item in active)
        total_max_profit = sum(item["max_profit"] for item in active)

        realized_wins = [t for t in self.closed_spreads if (t.get("realized_pnl") or 0) > 0]
        total_realized = sum((t.get("realized_pnl") or 0) for t in self.closed_spreads)
        win_rate = (len(realized_wins) / len(self.closed_spreads) * 100.0) if self.closed_spreads else 100.0

        return {
            "active_spreads": active,
            "active_count": len(active),
            "collateral_locked": round(total_collateral, 2),
            "unrealized_pnl": round(total_unrealized, 2),
            "max_profit": round(total_max_profit, 2),
            "closed_count": len(self.closed_spreads),
            "total_realized_pnl": round(total_realized, 2),
            "win_rate": round(win_rate, 1),
        }
