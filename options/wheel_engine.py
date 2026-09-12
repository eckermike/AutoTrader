"""
Option Wheel Strategy Engine for INTC (Intel Corporation).
Implements the Triple Income Option Wheel state machine:
1. Cash-Secured Puts (CSP) to acquire shares at a discount while collecting premium.
2. 50% Max-Profit Target Buy-to-Close early exit.
3. Covered Calls (CC) on assigned shares to generate recurring yield.
4. Coordination with Virtual Tax Escrow Engine and ntfy push notifications.
"""

import enum
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from config import BotConfig
from notifier import TradeNotifier
from options.options_client import (
    AlpacaOptionsClient,
    OptionContractInfo,
    OptionPositionInfo,
    StockPositionInfo,
)
from tax_engine import TaxEngine

logger = logging.getLogger("options.wheel")


class WheelState(str, enum.Enum):
    """Core states of the Option Wheel cycle."""

    CASH_SECURED_PUT = "CASH_SECURED_PUT"
    MONITORING_PUT = "MONITORING_PUT"
    COVERED_CALL = "COVERED_CALL"
    MONITORING_CALL = "MONITORING_CALL"


class WheelStatus(BaseModel):
    """Snapshot of current Wheel strategy status."""

    state: WheelState
    underlying: str
    stock_price: float
    shares_held: int
    cost_basis: Optional[float] = None
    active_contract: Optional[str] = None
    contract_entry_premium: Optional[float] = None
    collateral_locked: float = 0.0
    details: Dict[str, Any] = Field(default_factory=dict)


class WheelEngine:
    """
    Autonomous Option Wheel strategy engine for INTC.
    """

    def __init__(
        self,
        config: BotConfig,
        options_client: AlpacaOptionsClient,
        tax_engine: TaxEngine,
        notifier: TradeNotifier,
        symbol: Optional[str] = None,
        liquidity_manager: Optional[Any] = None,
    ):
        self.config = config
        self.client = options_client
        self.tax_engine = tax_engine
        self.notifier = notifier
        self.symbol = (symbol or config.WHEEL_SYMBOL).upper()
        self.liquidity_manager = liquidity_manager

        # Local tracking state
        self.cost_basis: Optional[float] = None
        self.active_contract_symbol: Optional[str] = None
        self.active_contract_premium: Optional[float] = None
        self.active_contract_strike: Optional[float] = None

    def evaluate_state(self) -> WheelState:
        """Determines the current phase of the Wheel lifecycle."""
        stock_pos = self.client.get_stock_position(self.symbol)
        shares = stock_pos.qty if stock_pos else 0
        opt_positions = self.client.get_active_option_positions(self.symbol)
        get_orders_fn = getattr(self.client, "get_open_orders", None)
        open_orders = get_orders_fn(self.symbol) if get_orders_fn else []

        active_puts = [p for p in opt_positions if p.contract_type == "put" and p.qty < 0]
        active_calls = [p for p in opt_positions if p.contract_type == "call" and p.qty < 0]
        pending_puts = [o for o in open_orders if "P" in getattr(o, "symbol", "")]
        pending_calls = [o for o in open_orders if "C" in getattr(o, "symbol", "")]

        if shares < 100:
            if active_puts or pending_puts:
                return WheelState.MONITORING_PUT
            return WheelState.CASH_SECURED_PUT
        else:
            if active_calls or pending_calls:
                return WheelState.MONITORING_CALL
            return WheelState.COVERED_CALL

    def select_put_contract(self, current_price: float) -> Optional[OptionContractInfo]:
        """
        Finds the optimal Cash-Secured Put contract:
        - DTE in [WHEEL_TARGET_DTE_MIN, WHEEL_TARGET_DTE_MAX]
        - Strike price ~5% to 10% Out-Of-The-Money (below current stock price)
        """
        contracts = self.client.get_option_contracts(
            underlying=self.symbol,
            contract_type="put",
            min_dte=self.config.WHEEL_TARGET_DTE_MIN,
            max_dte=self.config.WHEEL_TARGET_DTE_MAX,
        )
        if not contracts:
            logger.warning("No active put contracts found for %s in target DTE window.", self.symbol)
            return None

        # Target strike: ~5% to 8% below current price (OTM delta ~0.25)
        target_strike = current_price * (1.0 - 0.06)

        # Pick the put contract with strike closest to target_strike but strictly below current price
        otm_puts = [c for c in contracts if c.strike_price < current_price]
        if not otm_puts:
            otm_puts = contracts

        best_contract = min(otm_puts, key=lambda c: abs(c.strike_price - target_strike))
        return best_contract

    def select_call_contract(self, current_price: float, cost_basis: float) -> Optional[OptionContractInfo]:
        """
        Finds the optimal Covered Call contract:
        - DTE in [WHEEL_TARGET_DTE_MIN, WHEEL_TARGET_DTE_MAX]
        - Strike price >= cost_basis and Out-Of-The-Money
        """
        contracts = self.client.get_option_contracts(
            underlying=self.symbol,
            contract_type="call",
            min_dte=self.config.WHEEL_TARGET_DTE_MIN,
            max_dte=self.config.WHEEL_TARGET_DTE_MAX,
        )
        if not contracts:
            logger.warning("No active call contracts found for %s in target DTE window.", self.symbol)
            return None

        # Minimum acceptable strike is cost basis
        min_strike = max(current_price * 1.03, cost_basis)
        viable_calls = [c for c in contracts if c.strike_price >= min_strike]
        if not viable_calls:
            viable_calls = [c for c in contracts if c.strike_price > current_price]

        if not viable_calls:
            return None

        # Pick the call with strike closest to min_strike
        best_contract = min(viable_calls, key=lambda c: abs(c.strike_price - min_strike))
        return best_contract

    def step(self, total_cash: float, available_tradable_cash: Optional[float] = None) -> WheelStatus:
        """
        Executes a single cycle of the Option Wheel state machine.
        """
        stock_price = self.client.get_stock_price(self.symbol)
        stock_pos = self.client.get_stock_position(self.symbol)
        shares_held = stock_pos.qty if stock_pos else 0
        state = self.evaluate_state()
        tradable_cash = (
            available_tradable_cash
            if available_tradable_cash is not None
            else self.tax_engine.calculate_tradable_cash(total_cash)
        )

        logger.info(
            "[WHEEL CYCLE] %s: $%0.2f | Shares: %d | Phase: %s | Tradable Cash: $%s",
            self.symbol,
            stock_price,
            shares_held,
            state.value,
            f"{tradable_cash:,.2f}",
        )

        collateral_locked = 0.0

        # --- Phase 1: Sell Cash-Secured Put ---
        if state == WheelState.CASH_SECURED_PUT:
            contract = self.select_put_contract(stock_price)
            if contract:
                # Query real-time market quote for optimal execution
                live_quote = self.client.get_option_quote(contract.symbol)
                if live_quote and live_quote.get("bid_price", 0.0) > 0:
                    limit_p = live_quote["bid_price"]
                    contract.bid = live_quote["bid_price"]
                    contract.ask = live_quote["ask_price"]
                    contract.mid_price = live_quote["mid_price"]
                else:
                    limit_p = contract.mid_price

                # Ensure limit price is valid and floored at $0.05
                limit_p = max(0.05, round(limit_p, 2))
                req_collateral = contract.strike_price * 100 * self.config.WHEEL_CONTRACTS

                # Hard Capital Gate check
                if req_collateral > tradable_cash:
                    needed_capital = req_collateral - tradable_cash
                    logger.warning(
                        "Capital Gate Rejection: Required collateral $%0.2f exceeds Tradable Cash $%0.2f for %s. Skipping CSP.",
                        req_collateral,
                        tradable_cash,
                        self.symbol,
                    )
                    if self.liquidity_manager and hasattr(self.liquidity_manager, "request_liquidation_for_opportunity"):
                        self.liquidity_manager.request_liquidation_for_opportunity(
                            needed_cash=round(needed_capital, 2),
                            target_symbol=self.symbol,
                            opportunity_type=f"Option Wheel Cash-Secured Put (${contract.strike_price:0.2f}P)",
                            current_price=stock_price,
                            reserve_symbol="SGOV",
                        )
                else:
                    collateral_locked = req_collateral
                    order_tif = getattr(self.config, "WHEEL_TIME_IN_FORCE", "DAY")
                    logger.info(
                        "Submitting Cash-Secured Put order: SELL_TO_OPEN %d %s @ $%0.2f (TIF: %s)",
                        self.config.WHEEL_CONTRACTS,
                        contract.symbol,
                        limit_p,
                        order_tif,
                    )
                    order_res = self.client.submit_option_order(
                        symbol=contract.symbol,
                        side="SELL",
                        position_intent="SELL_TO_OPEN",
                        qty=self.config.WHEEL_CONTRACTS,
                        limit_price=limit_p,
                        time_in_force=order_tif,
                    )
                    
                    premium_collected = limit_p * self.config.WHEEL_CONTRACTS * 100
                    
                    # Allocate 30% to Virtual Tax Escrow
                    tax_record = self.tax_engine.record_option_premium(
                        symbol=self.symbol,
                        contract_symbol=contract.symbol,
                        contracts=self.config.WHEEL_CONTRACTS,
                        premium_total=premium_collected,
                    )

                    # Update local state
                    self.active_contract_symbol = contract.symbol
                    self.active_contract_premium = limit_p
                    self.active_contract_strike = contract.strike_price

                    # Dispatch Push Notification
                    rem_cash = self.tax_engine.calculate_tradable_cash(total_cash + premium_collected)
                    self.notifier.notify_wheel_csp_open(
                        underlying=self.symbol,
                        contract_symbol=contract.symbol,
                        strike=contract.strike_price,
                        expiration=contract.expiration_date,
                        dte=contract.days_to_expiration,
                        premium=limit_p,
                        contracts=self.config.WHEEL_CONTRACTS,
                        collateral=req_collateral,
                        tax_allocated=tax_record.tax_allocated,
                        tradable_cash=rem_cash,
                    )

        # --- Phase 2: Monitor Short Put (Profit-Taking or Assignment) ---
        elif state == WheelState.MONITORING_PUT:
            opt_positions = self.client.get_active_option_positions(self.symbol)
            active_put = next((p for p in opt_positions if p.contract_type == "put" and p.qty < 0), None)
            
            if active_put:
                entry_p = self.active_contract_premium or active_put.avg_entry_price
                curr_p = active_put.current_price
                
                # Check 50% profit target: current price <= 50% of entry premium
                profit_threshold = entry_p * (1.0 - self.config.WHEEL_PROFIT_TARGET_PCT)
                if curr_p <= profit_threshold:
                    logger.info(
                        "50%% Profit Target reached on %s (Entry: $%0.2f, Current: $%0.2f). Buying to close!",
                        active_put.symbol,
                        entry_p,
                        curr_p,
                    )
                    self.client.submit_option_order(
                        symbol=active_put.symbol,
                        side="BUY",
                        position_intent="BUY_TO_CLOSE",
                        qty=abs(active_put.qty),
                        limit_price=curr_p,
                    )
                    
                    open_total = entry_p * abs(active_put.qty) * 100
                    close_total = curr_p * abs(active_put.qty) * 100
                    net_profit = open_total - close_total
                    pct_profit = net_profit / open_total if open_total > 0 else 0.50

                    self.tax_engine.record_option_close(
                        contract_symbol=active_put.symbol,
                        contracts=abs(active_put.qty),
                        open_premium=open_total,
                        close_cost=close_total,
                    )

                    self.notifier.notify_wheel_profit_close(
                        underlying=self.symbol,
                        contract_symbol=active_put.symbol,
                        contracts=abs(active_put.qty),
                        open_premium=open_total,
                        close_cost=close_total,
                        net_profit=net_profit,
                        pct_profit=pct_profit,
                    )

                    self.active_contract_symbol = None
                    self.active_contract_premium = None
            else:
                # No filled position: check if pending order in order book is stale (>24 hours)
                open_orders = self.client.get_open_orders(self.symbol)
                order_ttl = getattr(self.config, "WHEEL_ORDER_TTL_HOURS", 24.0)
                cancelled_stale = False

                for o in open_orders:
                    status = str(getattr(o, "status", "")).lower()
                    if any(term in status for term in ("cancel", "reject", "expire", "fill")):
                        continue

                    sub_time = getattr(o, "submitted_at", None)
                    if sub_time:
                        try:
                            now_utc = datetime.now(timezone.utc)
                            if isinstance(sub_time, str):
                                order_dt = datetime.fromisoformat(sub_time.replace("Z", "+00:00"))
                            else:
                                order_dt = sub_time
                            elapsed_hours = (now_utc - order_dt).total_seconds() / 3600.0
                            if elapsed_hours >= order_ttl:
                                oid = str(getattr(o, "id", ""))
                                osym = getattr(o, "symbol", self.symbol)
                                logger.warning(
                                    "Pending CSP order %s (%s) is stale (elapsed: %.1f hrs >= limit %.1f hrs). Cancelling for daily refresh.",
                                    oid,
                                    osym,
                                    elapsed_hours,
                                    order_ttl,
                                )
                                self.client.cancel_order(oid)
                                cancelled_stale = True
                        except Exception as e:
                            logger.debug("Error checking order age for %s: %s", getattr(o, "id", ""), e)

                if cancelled_stale:
                    self.active_contract_symbol = None
                    self.active_contract_premium = None
                    logger.info("Stale orders cancelled for %s. Re-evaluating on next cycle with fresh quotes.", self.symbol)
                else:
                    logger.info(
                        "Cash-Secured Put order for %s is pending in order book. Awaiting execution.",
                        self.symbol,
                    )

        # --- Phase 3: Sell Covered Call ---
        elif state == WheelState.COVERED_CALL:
            basis = self.cost_basis or (stock_pos.avg_entry_price if stock_pos else stock_price)
            contract = self.select_call_contract(stock_price, basis)
            
            if contract:
                num_contracts = min(self.config.WHEEL_CONTRACTS, shares_held // 100)
                if num_contracts > 0:
                    live_quote = self.client.get_option_quote(contract.symbol)
                    if live_quote and live_quote.get("bid_price", 0.0) > 0:
                        call_limit_p = live_quote["bid_price"]
                        contract.bid = live_quote["bid_price"]
                        contract.ask = live_quote["ask_price"]
                        contract.mid_price = live_quote["mid_price"]
                    else:
                        call_limit_p = contract.mid_price

                    call_limit_p = max(0.05, round(call_limit_p, 2))
                    order_tif = getattr(self.config, "WHEEL_TIME_IN_FORCE", "DAY")

                    logger.info(
                        "Submitting Covered Call order: SELL_TO_OPEN %d %s @ $%0.2f (Cost Basis: $%0.2f, TIF: %s)",
                        num_contracts,
                        contract.symbol,
                        call_limit_p,
                        basis,
                        order_tif,
                    )
                    self.client.submit_option_order(
                        symbol=contract.symbol,
                        side="SELL",
                        position_intent="SELL_TO_OPEN",
                        qty=num_contracts,
                        limit_price=call_limit_p,
                        time_in_force=order_tif,
                    )
                    
                    call_prem = call_limit_p * num_contracts * 100
                    tax_rec = self.tax_engine.record_option_premium(
                        symbol=self.symbol,
                        contract_symbol=contract.symbol,
                        contracts=num_contracts,
                        premium_total=call_prem,
                    )

                    self.active_contract_symbol = contract.symbol
                    self.active_contract_premium = call_limit_p

                    self.notifier.notify_wheel_cc_open(
                        underlying=self.symbol,
                        contract_symbol=contract.symbol,
                        strike=contract.strike_price,
                        expiration=contract.expiration_date,
                        dte=contract.days_to_expiration,
                        premium=call_limit_p,
                        contracts=num_contracts,
                        tax_allocated=tax_rec.tax_allocated,
                    )

        # --- Phase 4: Monitor Covered Call ---
        elif state == WheelState.MONITORING_CALL:
            opt_positions = self.client.get_active_option_positions(self.symbol)
            active_call = next((p for p in opt_positions if p.contract_type == "call" and p.qty < 0), None)
            
            if active_call:
                entry_p = self.active_contract_premium or active_call.avg_entry_price
                curr_p = active_call.current_price
                
                # Check 50% profit target
                if curr_p <= entry_p * (1.0 - self.config.WHEEL_PROFIT_TARGET_PCT):
                    logger.info("50%% Profit target reached on Covered Call %s. Buying to close!", active_call.symbol)
                    self.client.submit_option_order(
                        symbol=active_call.symbol,
                        side="BUY",
                        position_intent="BUY_TO_CLOSE",
                        qty=abs(active_call.qty),
                        limit_price=curr_p,
                    )
                    open_tot = entry_p * abs(active_call.qty) * 100
                    close_tot = curr_p * abs(active_call.qty) * 100
                    net_prof = open_tot - close_tot

                    self.tax_engine.record_option_close(
                        contract_symbol=active_call.symbol,
                        contracts=abs(active_call.qty),
                        open_premium=open_tot,
                        close_cost=close_tot,
                    )

                    self.notifier.notify_wheel_profit_close(
                        underlying=self.symbol,
                        contract_symbol=active_call.symbol,
                        contracts=abs(active_call.qty),
                        open_premium=open_tot,
                        close_cost=close_tot,
                        net_profit=net_prof,
                        pct_profit=0.50,
                    )

                    self.active_contract_symbol = None
                    self.active_contract_premium = None
            else:
                # No filled position: check if pending call order is stale (>24 hours)
                open_orders = self.client.get_open_orders(self.symbol)
                order_ttl = getattr(self.config, "WHEEL_ORDER_TTL_HOURS", 24.0)
                cancelled_stale = False

                for o in open_orders:
                    status = str(getattr(o, "status", "")).lower()
                    if any(term in status for term in ("cancel", "reject", "expire", "fill")):
                        continue

                    sub_time = getattr(o, "submitted_at", None)
                    if sub_time:
                        try:
                            now_utc = datetime.now(timezone.utc)
                            if isinstance(sub_time, str):
                                order_dt = datetime.fromisoformat(sub_time.replace("Z", "+00:00"))
                            else:
                                order_dt = sub_time
                            elapsed_hours = (now_utc - order_dt).total_seconds() / 3600.0
                            if elapsed_hours >= order_ttl:
                                oid = str(getattr(o, "id", ""))
                                osym = getattr(o, "symbol", self.symbol)
                                logger.warning(
                                    "Pending Covered Call order %s (%s) is stale (elapsed: %.1f hrs >= limit %.1f hrs). Cancelling for daily refresh.",
                                    oid,
                                    osym,
                                    elapsed_hours,
                                    order_ttl,
                                )
                                self.client.cancel_order(oid)
                                cancelled_stale = True
                        except Exception as e:
                            logger.debug("Error checking call order age for %s: %s", getattr(o, "id", ""), e)

                if cancelled_stale:
                    self.active_contract_symbol = None
                    self.active_contract_premium = None
                    logger.info("Stale call orders cancelled for %s. Re-evaluating on next cycle.", self.symbol)
                else:
                    logger.info(
                        "Covered Call order for %s is pending in order book. Awaiting execution.",
                        self.symbol,
                    )

        return WheelStatus(
            state=state,
            underlying=self.symbol,
            stock_price=stock_price,
            shares_held=shares_held,
            cost_basis=self.cost_basis,
            active_contract=self.active_contract_symbol,
            contract_entry_premium=self.active_contract_premium,
            collateral_locked=collateral_locked,
        )


class WheelPortfolioManager:
    """
    Multi-Asset Option Wheel Portfolio Engine.
    Coordinates multiple isolated WheelEngine instances across target equity symbols.
    Enforces shared capital gating across the portfolio.
    """

    def __init__(
        self,
        config: BotConfig,
        options_client: AlpacaOptionsClient,
        tax_engine: TaxEngine,
        notifier: TradeNotifier,
        symbols: Optional[List[str]] = None,
        liquidity_manager: Optional[Any] = None,
    ):
        self.config = config
        self.client = options_client
        self.tax_engine = tax_engine
        self.notifier = notifier
        self.liquidity_manager = liquidity_manager
        target_symbols = symbols or config.WHEEL_SYMBOLS or [config.WHEEL_SYMBOL]
        self.symbols = [s.upper() for s in target_symbols]
        self.engines: Dict[str, WheelEngine] = {
            sym: WheelEngine(
                config=config,
                options_client=options_client,
                tax_engine=tax_engine,
                notifier=notifier,
                symbol=sym,
                liquidity_manager=liquidity_manager,
            )
            for sym in self.symbols
        }

    def step(self, total_cash: float) -> Dict[str, WheelStatus]:
        """
        Executes an evaluation cycle across all configured wheel assets.
        Shared capital gating: Each newly opened CSP decrements available tradable cash
        for subsequent wheels in the same cycle.
        """
        tradable_cash = self.tax_engine.calculate_tradable_cash(total_cash)
        remaining_cash = tradable_cash
        statuses: Dict[str, WheelStatus] = {}

        for sym in self.symbols:
            engine = self.engines[sym]
            try:
                status = engine.step(total_cash=total_cash, available_tradable_cash=remaining_cash)
                statuses[sym] = status
                if status.collateral_locked > 0:
                    remaining_cash = max(0.0, remaining_cash - status.collateral_locked)
            except Exception as e:
                logger.exception("Error executing wheel cycle for %s: %s", sym, e)

        return statuses
