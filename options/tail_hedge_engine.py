"""
Black Swan Tail-Risk Crash Hedge Strategy Engine for S&P 500 (SPY).
Implements systematic portfolio catastrophe insurance:
1. Discovers deep Out-of-the-Money put contracts on SPY (~15% OTM, 45 to 90 DTE).
2. Strict monthly insurance budget ceiling ($150/mo) funded by option wheel & spread profits.
3. Automated Monetization Exit (+250% profit target / 3.5x entry cost) to lock in crash windfalls.
4. Theta Defense Roll rule (closes/rolls at 21 DTE to preserve residual capital before steep decay).
5. 30% Virtual Tax Escrow segregation on all realized crash gains.
6. Persistent state synchronization to tail_hedge_state.json.
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

logger = logging.getLogger("options.tail_hedge")


class TailHedgeStatus(str, enum.Enum):
    """Lifecycle states of a tail-risk crash hedge."""

    DISCOVERY = "DISCOVERY"
    PENDING_ENTRY = "PENDING_ENTRY"
    ACTIVE = "ACTIVE"
    MONETIZED = "MONETIZED"
    ROLLED_OUT = "ROLLED_OUT"
    EXPIRED = "EXPIRED"


class TailHedgePosition(BaseModel):
    """Active tail hedge put position tracking model."""

    id: str
    underlying: str = "SPY"
    contract_symbol: str
    strike_price: float
    expiration_date: str
    days_to_expiration: int
    qty: int = 1
    entry_price_per_share: float
    entry_total_cost: float
    entry_timestamp: str
    current_price_per_share: float = 0.0
    current_market_value: float = 0.0
    unrealized_pnl: float = 0.0
    gain_pct: float = 0.0
    progress_to_target: float = 0.0
    target_monetization_price: float
    order_id: Optional[str] = None
    exit_order_id: Optional[str] = None
    status: str = "ACTIVE"
    close_timestamp: Optional[str] = None
    close_price_per_share: Optional[float] = None
    realized_pnl: Optional[float] = None
    exit_reason: Optional[str] = None


class TailHedgeEngine:
    """
    Autonomous Tail-Risk Crash Hedge Engine.
    Manages long deep-OTM put protection for the entire portfolio.
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
        self.underlying = getattr(config, "HEDGE_UNDERLYING", "SPY").upper()
        self.state_file = getattr(config, "HEDGE_STATE_FILE", "tail_hedge_state.json")

        # Active position tracking
        self.active_hedge: Optional[TailHedgePosition] = None
        self.pending_order_id: Optional[str] = None
        self.pending_order_time: Optional[datetime] = None

        # Monthly expenditure tracking
        self.current_month: str = datetime.now(timezone.utc).strftime("%Y-%m")
        self.monthly_spent_usd: float = 0.0
        self.closed_hedges: List[Dict[str, Any]] = []

        # Load persisted state
        self.load_state()

    def load_state(self) -> None:
        """Loads active hedge and expenditure history from persistent JSON file."""
        if not os.path.exists(self.state_file):
            logger.info("No tail hedge state file found at %s. Initializing fresh.", self.state_file)
            return

        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            now_month = datetime.now(timezone.utc).strftime("%Y-%m")
            saved_month = data.get("current_month", now_month)
            if saved_month == now_month:
                self.monthly_spent_usd = float(data.get("monthly_spent_usd", 0.0))
            else:
                self.monthly_spent_usd = 0.0
                self.current_month = now_month

            active_data = data.get("active_hedge")
            if active_data:
                self.active_hedge = TailHedgePosition(**active_data)

            self.closed_hedges = data.get("closed_hedges", [])
            logger.info(
                "Loaded tail hedge state from %s (Active: %s, Monthly Spent: $%.2f)",
                self.state_file,
                bool(self.active_hedge),
                self.monthly_spent_usd,
            )
        except Exception as e:
            logger.warning("Failed to load tail hedge state from %s: %s", self.state_file, e)

    def save_state(self) -> None:
        """Persists active hedge and monthly budget stats to JSON file atomically."""
        try:
            payload = {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "current_month": self.current_month,
                "monthly_spent_usd": round(self.monthly_spent_usd, 2),
                "active_hedge": self.active_hedge.model_dump() if self.active_hedge else None,
                "closed_hedges": self.closed_hedges[-50:],
            }
            tmp_file = f"{self.state_file}.tmp"
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp_file, self.state_file)
        except Exception as e:
            logger.warning("Failed to save tail hedge state to %s: %s", self.state_file, e)

    def select_hedge_contract(self, current_price: float) -> Optional[OptionContractInfo]:
        """
        Discovers the optimal deep Out-of-the-Money Put contract on SPY:
        - DTE in [HEDGE_TARGET_DTE_MIN (45), HEDGE_TARGET_DTE_MAX (90)]
        - Target Strike: ~12% to 18% below current spot price (~0.05-0.10 delta)
        - Price: strictly <= HEDGE_MAX_COST_PER_CONTRACT_USD ($1.00/sh or $100/contract)
        """
        min_dte = getattr(self.config, "HEDGE_TARGET_DTE_MIN", 45)
        max_dte = getattr(self.config, "HEDGE_TARGET_DTE_MAX", 90)
        otm_pct = getattr(self.config, "HEDGE_OTM_PCT", 0.15)
        max_cost = getattr(self.config, "HEDGE_MAX_COST_PER_CONTRACT_USD", 1.00)

        contracts = self.client.get_option_contracts(
            underlying=self.underlying,
            contract_type="put",
            min_dte=min_dte,
            max_dte=max_dte,
        )
        if not contracts:
            logger.debug("No put contracts found for %s in %d-%d DTE window.", self.underlying, min_dte, max_dte)
            return None

        target_strike = current_price * (1.0 - otm_pct)
        # Filter for contracts below spot price and within cost threshold
        otm_puts = [
            c for c in contracts
            if c.strike_price < current_price and (c.ask <= max_cost or c.mid_price <= max_cost)
        ]
        if not otm_puts:
            logger.debug("No OTM put contracts on %s met the max cost ceiling ($%.2f/sh).", self.underlying, max_cost)
            return None

        # Pick the contract with strike closest to target_strike
        best_contract = min(otm_puts, key=lambda c: abs(c.strike_price - target_strike))
        return best_contract

    def step(self, is_market_open: bool = True) -> Dict[str, Any]:
        """
        Main execution step for the Tail-Risk Crash Hedge:
        1. Resets monthly budget counter if month changed.
        2. Monitors active hedge for +250% monetization or 21 DTE roll.
        3. Manages pending entry orders and 24h TTL expiration.
        4. If no active hedge and market is open, discovers and purchases cheap crash protection.
        """
        now_dt = datetime.now(timezone.utc)
        now_month = now_dt.strftime("%Y-%m")
        if now_month != self.current_month:
            logger.info("New calendar month (%s). Resetting tail hedge monthly budget counter.", now_month)
            self.current_month = now_month
            self.monthly_spent_usd = 0.0

        current_price = self.client.get_stock_price(self.underlying)

        # 1. Monitor Active Position
        if self.active_hedge and self.active_hedge.status == "ACTIVE":
            res = self._monitor_active_hedge(current_price)
            self.save_state()
            return res

        # 2. Check Pending Entry Order
        if self.pending_order_id:
            res = self._check_pending_order(now_dt)
            self.save_state()
            return res

        # 3. If market is closed and no active hedge, do not place new orders
        if not is_market_open:
            return {"status": "MARKET_CLOSED"}

        # 4. Evaluate New Crash Hedge Purchase
        res = self._evaluate_new_hedge(current_price)
        self.save_state()
        return res

    def _monitor_active_hedge(self, current_price: float) -> Dict[str, Any]:
        """
        Monitors active crash put for:
        - Trigger 1: Systematic Monetization Exit (+250% target / 3.5x entry cost).
        - Trigger 2: 21 DTE Theta Defense Roll (prevents steep final-month decay).
        """
        hedge = self.active_hedge
        quote = self.client.get_option_quote(hedge.contract_symbol)
        curr_price_per_share = quote.get("bid_price", 0.0) if quote else hedge.entry_price_per_share
        if curr_price_per_share <= 0:
            curr_price_per_share = quote.get("mid_price", 0.0) if quote else hedge.entry_price_per_share

        curr_market_val = round(curr_price_per_share * 100 * hedge.qty, 2)
        unrealized_pnl = round(curr_market_val - hedge.entry_total_cost, 2)
        gain_pct = round((curr_price_per_share - hedge.entry_price_per_share) / max(0.01, hedge.entry_price_per_share) * 100.0, 1)

        # Progress toward +250% monetization target
        profit_target_pct = getattr(self.config, "HEDGE_PROFIT_TARGET_PCT", 2.50)
        target_profit_dollars = hedge.entry_total_cost * profit_target_pct
        progress_to_target = min(1.0, max(0.0, unrealized_pnl / max(1.0, target_profit_dollars)))

        hedge.current_price_per_share = curr_price_per_share
        hedge.current_market_value = curr_market_val
        hedge.unrealized_pnl = unrealized_pnl
        hedge.gain_pct = gain_pct
        hedge.progress_to_target = round(progress_to_target, 3)

        # Recalculate DTE
        today = date.today()
        exp_date = datetime.strptime(hedge.expiration_date, "%Y-%m-%d").date()
        dte = (exp_date - today).days
        hedge.days_to_expiration = dte

        # --- Trigger 1: Systematic Monetization Exit (+250% profit target) ---
        target_price = hedge.target_monetization_price
        if curr_price_per_share >= target_price or (curr_market_val >= hedge.entry_total_cost * (1.0 + profit_target_pct)):
            logger.info(
                "🚨 [CRASH HEDGE] %s reached Monetization Target! ($%.2f/sh >= $%.2f/sh, +%.1f%%). Monetizing windfall!",
                hedge.contract_symbol,
                curr_price_per_share,
                target_price,
                gain_pct,
            )
            return self._execute_hedge_exit(
                reason="MONETIZATION_PROFIT_TARGET",
                exit_price=curr_price_per_share,
                realized_pnl=unrealized_pnl,
            )

        # --- Trigger 2: 21 DTE Theta Defense Roll ---
        roll_dte = getattr(self.config, "HEDGE_ROLL_DTE", 21)
        if dte <= roll_dte:
            logger.info(
                "🛡️ [CRASH HEDGE] %s reached %d DTE (<= %d). Closing to preserve residual capital before final theta decay.",
                hedge.contract_symbol,
                dte,
                roll_dte,
            )
            return self._execute_hedge_exit(
                reason="THETA_DEFENSE_ROLL_21_DTE",
                exit_price=curr_price_per_share,
                realized_pnl=unrealized_pnl,
            )

        return {
            "status": "MONITORING",
            "contract": hedge.contract_symbol,
            "dte": dte,
            "current_value": curr_market_val,
            "unrealized_pnl": unrealized_pnl,
            "gain_pct": gain_pct,
            "progress_to_target": progress_to_target,
        }

    def _execute_hedge_exit(
        self, reason: str, exit_price: float, realized_pnl: float
    ) -> Dict[str, Any]:
        """Sells the long put to close, segregates 30% tax escrow on net gains, and alerts user."""
        hedge = self.active_hedge
        order_res = self.client.submit_option_order(
            symbol=hedge.contract_symbol,
            side="SELL",
            position_intent="SELL_TO_CLOSE",
            qty=hedge.qty,
            limit_price=exit_price,
            time_in_force=getattr(self.config, "HEDGE_TIME_IN_FORCE", "DAY"),
        )

        hedge.status = "MONETIZED" if "MONETIZATION" in reason else "ROLLED_OUT"
        hedge.close_timestamp = datetime.now(timezone.utc).isoformat()
        hedge.close_price_per_share = exit_price
        hedge.realized_pnl = realized_pnl
        hedge.exit_reason = reason
        hedge.exit_order_id = order_res.get("id")

        # Record trade result in TaxEngine
        self.tax_engine.record_trade_result(
            trade_id=order_res.get("id", f"hedge_close_{hedge.contract_symbol}"),
            symbol=f"{self.underlying} Tail Hedge",
            side="SELL_TO_CLOSE",
            gross_pnl=realized_pnl,
        )

        tax_rate = getattr(self.config, "TAX_RATE", 0.30)
        tax_allocated = (realized_pnl * tax_rate) if realized_pnl > 0 else 0.0

        # Push Notification
        if "MONETIZATION" in reason:
            if hasattr(self.notifier, "notify_tail_hedge_monetized"):
                self.notifier.notify_tail_hedge_monetized(
                    underlying=self.underlying,
                    contract_symbol=hedge.contract_symbol,
                    strike=hedge.strike_price,
                    entry_cost=hedge.entry_total_cost,
                    exit_value=exit_price * 100 * hedge.qty,
                    realized_pnl=realized_pnl,
                    gain_pct=hedge.gain_pct,
                    tax_allocated=tax_allocated,
                    tax_reserve_after=self.tax_engine.tax_reserve,
                )
            else:
                self.notifier.notify(
                    f"🚨 [CRASH HEDGE MONETIZED] {hedge.contract_symbol} locked in +${realized_pnl:,.2f} windfall! "
                    f"Tax Escrow: +${tax_allocated:,.2f}"
                )
        else:
            if hasattr(self.notifier, "notify_tail_hedge_rolled"):
                self.notifier.notify_tail_hedge_rolled(
                    underlying=self.underlying,
                    contract_symbol=hedge.contract_symbol,
                    dte=hedge.days_to_expiration,
                    residual_value=exit_price * 100 * hedge.qty,
                    realized_pnl=realized_pnl,
                )
            else:
                self.notifier.notify(
                    f"🛡️ [CRASH HEDGE ROLLED] {hedge.contract_symbol} closed at {hedge.days_to_expiration} DTE to preserve capital."
                )

        closed_record = hedge.model_dump()
        self.closed_hedges.append(closed_record)
        self.active_hedge = None

        return {
            "status": "CLOSED",
            "reason": reason,
            "realized_pnl": realized_pnl,
            "record": closed_record,
        }

    def _evaluate_new_hedge(self, current_price: float) -> Dict[str, Any]:
        """Checks budget and capital gates, discovers optimal deep-OTM put, and submits buy order."""
        monthly_budget = getattr(self.config, "HEDGE_MONTHLY_BUDGET_USD", 150.0)
        if self.monthly_spent_usd >= monthly_budget:
            logger.debug(
                "Skipping new tail hedge: monthly expenditure ($%.2f) reached budget ceiling ($%.2f)",
                self.monthly_spent_usd,
                monthly_budget,
            )
            return {"status": "MONTHLY_BUDGET_REACHED"}

        contract = self.select_hedge_contract(current_price)
        if not contract:
            return {"status": "NO_SUITABLE_CONTRACT"}

        # Determine limit price from quote
        quote = self.client.get_option_quote(contract.symbol)
        limit_price = quote.get("ask_price") or quote.get("mid_price") or contract.ask or contract.mid_price or 0.60
        total_cost = round(limit_price * 100 * 1, 2)

        # Check monthly budget ceiling
        if self.monthly_spent_usd + total_cost > monthly_budget:
            logger.debug(
                "Skipping hedge %s: total cost ($%.2f) would exceed remaining monthly budget ($%.2f)",
                contract.symbol,
                total_cost,
                monthly_budget - self.monthly_spent_usd,
            )
            return {"status": "EXCEEDS_REMAINING_BUDGET"}

        # Check TaxEngine Tradable Cash Gate
        tradable_cash = self.tax_engine.get_tradable_cash(100000.0)
        if total_cost > tradable_cash:
            logger.warning("Tail Hedge Capital Gate: Cost ($%.2f) exceeds tradable cash ($%.2f)", total_cost, tradable_cash)
            if self.liquidity_manager and hasattr(self.liquidity_manager, "request_liquidation_for_opportunity"):
                needed = round(total_cost - tradable_cash, 2)
                self.liquidity_manager.request_liquidation_for_opportunity(
                    needed_cash=needed,
                    target_symbol=self.underlying,
                    opportunity_type="Black Swan Crash Hedge",
                    current_price=current_price,
                    reserve_symbol="SGOV",
                )
            return {"status": "CAPITAL_GATE_REJECTED"}

        # Submit Buy-To-Open Option Order
        order_res = self.client.submit_option_order(
            symbol=contract.symbol,
            side="BUY",
            position_intent="BUY_TO_OPEN",
            qty=1,
            limit_price=limit_price,
            time_in_force=getattr(self.config, "HEDGE_TIME_IN_FORCE", "DAY"),
        )

        profit_target_pct = getattr(self.config, "HEDGE_PROFIT_TARGET_PCT", 2.50)
        target_monetize_price = round(limit_price * (1.0 + profit_target_pct), 2)

        hedge_pos = TailHedgePosition(
            id=order_res.get("id", f"hdg_{contract.symbol}"),
            underlying=self.underlying,
            contract_symbol=contract.symbol,
            strike_price=contract.strike_price,
            expiration_date=contract.expiration_date,
            days_to_expiration=contract.days_to_expiration,
            qty=1,
            entry_price_per_share=limit_price,
            entry_total_cost=total_cost,
            entry_timestamp=datetime.now(timezone.utc).isoformat(),
            current_price_per_share=limit_price,
            current_market_value=total_cost,
            target_monetization_price=target_monetize_price,
            order_id=order_res.get("id"),
            status="ACTIVE",
        )

        self.active_hedge = hedge_pos
        self.monthly_spent_usd += total_cost

        # Dispatch Push Notification
        if hasattr(self.notifier, "notify_tail_hedge_open"):
            self.notifier.notify_tail_hedge_open(
                underlying=self.underlying,
                contract_symbol=contract.symbol,
                strike=contract.strike_price,
                expiration=contract.expiration_date,
                dte=contract.days_to_expiration,
                cost_per_share=limit_price,
                total_cost=total_cost,
                monetize_target_price=target_monetize_price,
                monthly_spent=self.monthly_spent_usd,
                monthly_budget=monthly_budget,
            )
        else:
            self.notifier.notify(
                f"🛡️ [CRASH HEDGE OPENED] Bought 1 {contract.symbol} Put @ ${limit_price:.2f} (${total_cost:.2f} total).\n"
                f"• Target Monetization Exit: ${target_monetize_price:.2f}/sh (+{profit_target_pct * 100:.0f}%)\n"
                f"• Monthly Budget: ${self.monthly_spent_usd:.2f} / ${monthly_budget:.2f}"
            )

        return {
            "status": "OPENED",
            "contract": contract.symbol,
            "cost": total_cost,
            "hedge": hedge_pos.model_dump(),
        }

    def _check_pending_order(self, now_dt: datetime) -> Dict[str, Any]:
        """Cancels stale pending orders after 24h TTL."""
        if not self.pending_order_time:
            self.pending_order_id = None
            return {"status": "PENDING_CLEARED"}

        elapsed_hours = (now_dt - self.pending_order_time).total_seconds() / 3600.0
        if elapsed_hours >= 24.0:
            logger.info("Pending tail hedge order %s expired 24h TTL. Cancelling.", self.pending_order_id)
            self.client.cancel_order(self.pending_order_id)
            self.pending_order_id = None
            self.pending_order_time = None
            return {"status": "CANCELLED_TTL"}

        return {"status": "PENDING", "order_id": self.pending_order_id}

    def get_summary_metrics(self) -> Dict[str, Any]:
        """Aggregates portfolio-level telemetry for Investor Dashboard Tab 7."""
        monthly_budget = getattr(self.config, "HEDGE_MONTHLY_BUDGET_USD", 150.0)
        active_dict = self.active_hedge.model_dump() if self.active_hedge else None

        realized_gains = sum((h.get("realized_pnl") or 0.0) for h in self.closed_hedges)
        monetized_count = len([h for h in self.closed_hedges if "MONETIZATION" in str(h.get("exit_reason", ""))])

        return {
            "enabled": getattr(self.config, "HEDGE_ENABLED", True),
            "underlying": self.underlying,
            "monthly_budget": round(monthly_budget, 2),
            "monthly_spent": round(self.monthly_spent_usd, 2),
            "monthly_budget_remaining": round(max(0.0, monthly_budget - self.monthly_spent_usd), 2),
            "active_hedge": active_dict,
            "has_active_hedge": bool(self.active_hedge),
            "unrealized_pnl": self.active_hedge.unrealized_pnl if self.active_hedge else 0.0,
            "gain_pct": self.active_hedge.gain_pct if self.active_hedge else 0.0,
            "progress_to_target": self.active_hedge.progress_to_target if self.active_hedge else 0.0,
            "total_realized_pnl": round(realized_gains, 2),
            "monetized_count": monetized_count,
            "closed_count": len(self.closed_hedges),
            "closed_hedges": self.closed_hedges[-10:],
        }
