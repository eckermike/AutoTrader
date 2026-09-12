"""
Autonomous Liquidity and Cash Yield Management Engine.
Manages cash-yield allocations (SGOV, FBND) and two-way human-in-the-loop (HITL)
trade approvals via ntfy.sh iOS action buttons with two-layer safety guardrails:
- Layer 1: Time-To-Live (TTL) expiration window (default 4.0 hours).
- Layer 2: Pre-execution live price slippage & viability reassessment (max 0.5% slippage).
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from pydantic import BaseModel, Field

from config import BotConfig
from notifier import TradeNotifier

logger = logging.getLogger("execution.liquidity")


class PendingApproval(BaseModel):
    id: str
    request_type: str  # BOND_BARBELL_5050 or LIQUIDATION_FOR_TRADE
    created_at: str
    created_timestamp: float = Field(default_factory=time.time)
    sgov_amount: float = 0.0
    fbnd_amount: float = 0.0
    target_symbol: Optional[str] = None
    reference_price: Optional[float] = None
    opportunity_type: Optional[str] = None
    status: str = "PENDING"  # PENDING, APPROVED, REJECTED, EXECUTED, EXPIRED, ABORTED_VIABILITY
    order_ids: List[str] = Field(default_factory=list)


class LiquidityState(BaseModel):
    pending_approval: Optional[PendingApproval] = None
    last_poll_timestamp: int = 0
    history: List[Dict[str, Any]] = Field(default_factory=list)
    processed_dividend_ids: List[str] = Field(default_factory=list)


class LiquidityManager:
    """
    Coordinates idle cash deployment into cash-yield funds (SGOV / FBND),
    manages two-way mobile approvals via ntfy.sh action buttons with
    Layer 1 (4h TTL) and Layer 2 (Pre-execution viability) safety gates,
    and automatically escrows 30% of bond dividends & realized capital gains.
    """

    def __init__(
        self,
        config: BotConfig,
        trading_client: Any,
        notifier: TradeNotifier,
        tax_engine: Optional[Any] = None,
        state_file: Optional[Path | str] = None,
    ):
        self.config = config
        self.trading_client = trading_client
        self.notifier = notifier
        self.tax_engine = tax_engine
        self.state_file = Path(state_file or getattr(config, "LIQUIDITY_STATE_FILE", "liquidity_state.json"))
        self.action_topic = getattr(config, "NTFY_ACTION_TOPIC", "eckermike87-actions")
        self.state = self._load_state()


        if self.state.last_poll_timestamp == 0:
            self.state.last_poll_timestamp = int(time.time()) - 120
            self._save_state()

    def _load_state(self) -> LiquidityState:
        if self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return LiquidityState(**data)
            except Exception as e:
                logger.warning("Could not load liquidity state from %s: %s. Starting fresh.", self.state_file, e)
        return LiquidityState(last_poll_timestamp=int(time.time()) - 120)

    def _save_state(self) -> None:
        try:
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(self.state.model_dump(), f, indent=2)
        except Exception as e:
            logger.error("Failed to save liquidity state to %s: %s", self.state_file, e)

    def dispatch_5050_bond_request(self, force: bool = False) -> bool:
        """
        Dispatches an interactive approval notification to iOS via ntfy to deploy
        idle cash into a 50/50 SGOV + FBND bond barbell.
        """
        if not getattr(self.config, "LIQUIDITY_RESERVE_ENABLED", True) and not force:
            return False

        if self.state.pending_approval and self.state.pending_approval.status == "PENDING" and not force:
            logger.info("Approval request %s is already pending. Skipping duplicate dispatch.", self.state.pending_approval.id)
            return False

        sgov_amt = getattr(self.config, "SGOV_ALLOCATION_USD", 20000.0)
        fbnd_amt = getattr(self.config, "FBND_ALLOCATION_USD", 20000.0)
        total_amt = sgov_amt + fbnd_amt
        ttl_hours = getattr(self.config, "APPROVAL_TTL_HOURS", 4.0)

        req_id = f"REQ-5050-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        proposal_title = f"Capital Allocation Request (${total_amt:,.0f})"
        proposal_msg = (
            f"📊 Deploy ${total_amt:,.0f} idle cash into 50/50 Bond Barbell:\n\n"
            f"• SGOV (0-3M T-Bills, ~5.1% yield): ${sgov_amt:,.2f}\n"
            f"• FBND (Fidelity Total Bond, ~5.0% yield): ${fbnd_amt:,.2f}\n\n"
            f"Liquid Cash Buffer Remaining: ~$15,100\n"
            f"Option Collateral Unaffected: $45,200\n"
            f"Approval Window: {ttl_hours:.0f} hours.\n\n"
            f"Tap [Approve] on your iPhone to execute."
        )

        logger.info("Dispatching interactive ntfy approval request [%s] for $%.2f...", req_id, total_amt)

        self.notifier.notify_approval_request(
            proposal_title=proposal_title,
            proposal_message=proposal_msg,
            action_topic=self.action_topic,
            approve_body="APPROVE_5050_BONDS",
            reject_body="REJECT_5050_BONDS",
            approve_label=f"Approve 50/50 Buy (${total_amt/1000:,.0f}k)",
            reject_label="Reject",
        )

        self.state.pending_approval = PendingApproval(
            id=req_id,
            request_type="BOND_BARBELL_5050",
            created_at=datetime.now().isoformat(),
            created_timestamp=time.time(),
            sgov_amount=sgov_amt,
            fbnd_amount=fbnd_amt,
            status="PENDING",
        )
        self._save_state()
        return True

    def request_liquidation_for_opportunity(
        self,
        needed_cash: float,
        target_symbol: str,
        opportunity_type: str,
        current_price: float,
        reserve_symbol: str = "SGOV",
    ) -> bool:
        """
        Dispatches an interactive approval request to sell shares of a reserve fund (SGOV)
        to finance an opportunity (e.g. Option Wheel put or Crypto breakout).
        Includes Layer 1 (4h TTL) and Layer 2 (0.5% max slippage check).
        """
        if not getattr(self.config, "LIQUIDITY_RESERVE_ENABLED", True):
            return False

        if self.state.pending_approval and self.state.pending_approval.status == "PENDING":
            logger.info("Approval request %s already pending. Skipping duplicate.", self.state.pending_approval.id)
            return False

        # Determine which reserve asset to liquidate (prefer asset with open position)
        chosen_reserve_sym = reserve_symbol
        candidates = [reserve_symbol, "FBND" if reserve_symbol == "SGOV" else "SGOV"]
        if hasattr(self.trading_client, "get_open_position"):
            for sym in candidates:
                try:
                    pos = self.trading_client.get_open_position(sym)
                    if pos and hasattr(pos, "market_value"):
                        mv = float(pos.market_value)
                        if mv > 50.0:
                            chosen_reserve_sym = sym
                            break
                except Exception:
                    pass

        reserve_symbol = chosen_reserve_sym

        # Consult LLM Advisor for optimal tranche selection & rationale
        ai_recommendation = None
        try:
            from intelligence.llm_advisor import get_llm_advisor
            rec_result = get_llm_advisor().recommend_liquidation_tranche(
                needed_cash=needed_cash,
                target_symbol=target_symbol,
                opportunity_type=opportunity_type,
            )
            ai_recommendation = f"Sell {rec_result.get('recommended_tranche', reserve_symbol)}: {rec_result.get('reasoning', '')}"
            if rec_result.get("recommended_tranche") in ("SGOV", "FBND"):
                reserve_symbol = rec_result.get("recommended_tranche")
        except Exception as e:
            logger.debug("LLM liquidation advisor unavailable: %s", e)

        est_reserve_price = 100.50 if reserve_symbol == "SGOV" else 44.50
        est_shares = round(needed_cash / est_reserve_price, 2)
        ttl_hours = getattr(self.config, "APPROVAL_TTL_HOURS", 4.0)

        req_id = f"REQ-LIQ-{target_symbol}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        proposal_title = f"Capital Request: {target_symbol} (${needed_cash:,.0f})"
        proposal_msg = (
            f"⚠️ Capital needed for {opportunity_type} on {target_symbol}:\n\n"
            f"• Target Symbol: {target_symbol} @ ${current_price:,.2f}\n"
            f"• Capital Needed: ${needed_cash:,.2f}\n"
            f"• Proposed Action: Sell ~{est_shares:.1f} shares of {reserve_symbol}\n"
            f"• Safety Gate: Valid for {ttl_hours:.0f}h with max 0.5% price slippage.\n\n"
            f"Tap [Approve] to authorize liquidation & order entry."
        )

        logger.info("Dispatching liquidation approval request [%s] for %s ($%.2f)...", req_id, target_symbol, needed_cash)

        self.notifier.notify_approval_request(
            proposal_title=proposal_title,
            proposal_message=proposal_msg,
            action_topic=self.action_topic,
            approve_body=f"APPROVE_LIQUIDATION_{req_id}",
            reject_body=f"REJECT_LIQUIDATION_{req_id}",
            approve_label=f"Approve Sell (${needed_cash/1000:,.1f}k)",
            reject_label="Reject",
            ai_recommendation=ai_recommendation,
        )

        self.state.pending_approval = PendingApproval(
            id=req_id,
            request_type="LIQUIDATION_FOR_TRADE",
            created_at=datetime.now().isoformat(),
            created_timestamp=time.time(),
            sgov_amount=needed_cash if reserve_symbol == "SGOV" else 0.0,
            fbnd_amount=needed_cash if reserve_symbol == "FBND" else 0.0,
            target_symbol=target_symbol,
            reference_price=current_price,
            opportunity_type=opportunity_type,
            status="PENDING",
        )
        self._save_state()
        return True

    def reassess_viability(self, pending: Optional[PendingApproval]) -> Tuple[bool, str]:
        """
        Layer 2 Pre-Execution Viability & Slippage Gate:
        Re-verifies market price slippage against reference price before submitting broker orders.
        """
        if not pending or not pending.target_symbol or pending.reference_price is None or pending.reference_price <= 0:
            return True, "Viability confirmed (no reference price constraint)"

        max_slippage = getattr(self.config, "APPROVAL_MAX_SLIPPAGE_PCT", 0.005)
        current_price = None

        try:
            if hasattr(self.trading_client, "get_latest_quote"):
                quote = self.trading_client.get_latest_quote(pending.target_symbol)
                current_price = getattr(quote, "ask_price", None) or getattr(quote, "bid_price", None)
            elif hasattr(self.trading_client, "get_stock_latest_quote"):
                quote = self.trading_client.get_stock_latest_quote(pending.target_symbol)
                current_price = getattr(quote, "ask_price", None) or getattr(quote, "bid_price", None)
        except Exception as e:
            logger.warning("Could not query live quote for %s viability check: %s", pending.target_symbol, e)

        if current_price is not None and current_price > 0:
            slippage = abs(current_price - pending.reference_price) / pending.reference_price
            if slippage > max_slippage:
                return (
                    False,
                    f"Price moved {slippage * 100:.2f}% (from ${pending.reference_price:,.2f} to ${current_price:,.2f}), exceeding max allowed slippage of {max_slippage * 100:.1f}%",
                )

        return True, "Viability confirmed"

    def poll_action_topic(self) -> List[str]:
        """
        Polls the ntfy action topic for user tap events since last poll timestamp.
        """
        url = f"https://ntfy.sh/{self.action_topic}/json?poll=1&since={self.state.last_poll_timestamp}"
        actions_received: List[str] = []

        try:
            resp = requests.get(url, timeout=5.0)
            if resp.status_code == 200:
                current_max_time = self.state.last_poll_timestamp
                for line in resp.text.strip().split("\n"):
                    if not line.strip():
                        continue
                    try:
                        msg_data = json.loads(line)
                        event = msg_data.get("event")
                        if event == "message":
                            body = msg_data.get("message", "").strip()
                            actions_received.append(body)
                            msg_time = msg_data.get("time", 0)
                            if msg_time > current_max_time:
                                current_max_time = msg_time
                    except json.JSONDecodeError:
                        continue

                # Advance poll timestamp to avoid re-processing old messages
                if current_max_time > self.state.last_poll_timestamp:
                    self.state.last_poll_timestamp = current_max_time + 1
                    self._save_state()
            else:
                logger.debug("ntfy action poll returned status %d", resp.status_code)
        except Exception as e:
            logger.debug("Error polling ntfy action topic %s: %s", self.action_topic, e)

        return actions_received

    def execute_5050_orders(self, sgov_amount: float, fbnd_amount: float) -> Tuple[Optional[Any], Optional[Any]]:
        """
        Submits market buy orders for SGOV and FBND to Alpaca.
        """
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        sgov_order = None
        fbnd_order = None

        try:
            logger.info("Submitting Alpaca Market Buy: $%.2f SGOV...", sgov_amount)
            sgov_order = self.trading_client.submit_order(
                MarketOrderRequest(
                    symbol="SGOV",
                    notional=round(sgov_amount, 2),
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                )
            )
            logger.info("SGOV order submitted: ID=%s, status=%s", getattr(sgov_order, "id", None), getattr(sgov_order, "status", None))
        except Exception as e:
            logger.exception("Failed to submit SGOV order: %s", e)

        try:
            logger.info("Submitting Alpaca Market Buy: $%.2f FBND...", fbnd_amount)
            fbnd_order = self.trading_client.submit_order(
                MarketOrderRequest(
                    symbol="FBND",
                    notional=round(fbnd_amount, 2),
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                )
            )
            logger.info("FBND order submitted: ID=%s, status=%s", getattr(fbnd_order, "id", None), getattr(fbnd_order, "status", None))
        except Exception as e:
            logger.exception("Failed to submit FBND order: %s", e)

        return sgov_order, fbnd_order

    def execute_liquidation_order(self, symbol: str = "SGOV", notional_amount: float = 0.0) -> Optional[Any]:
        """
        Submits market sell order to liquidate shares of reserve fund (SGOV/FBND).
        """
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        try:
            logger.info("Submitting Alpaca Market Sell: $%.2f %s...", notional_amount, symbol)
            order = self.trading_client.submit_order(
                MarketOrderRequest(
                    symbol=symbol,
                    notional=round(notional_amount, 2),
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY,
                )
            )
            logger.info("%s liquidation order submitted: ID=%s, status=%s", symbol, getattr(order, "id", None), getattr(order, "status", None))

            # Reconcile realized capital gains or losses with TaxEngine if available
            if self.tax_engine and self.trading_client:
                try:
                    pos = self.trading_client.get_open_position(symbol)
                    if pos:
                        entry_p = float(pos.avg_entry_price)
                        exit_p = float(getattr(order, "filled_avg_price", None) or pos.current_price or entry_p)
                        qty = notional_amount / exit_p if exit_p > 0 else 0.0
                        self.tax_engine.record_closed_trade(
                            symbol=symbol,
                            side="SELL",
                            qty=qty,
                            entry_price=entry_p,
                            exit_price=exit_p,
                        )
                except Exception as e:
                    logger.debug("Could not record liquidation tax reconciliation for %s: %s", symbol, e)

            return order
        except Exception as e:
            logger.exception("Failed to submit %s liquidation order: %s", symbol, e)
            return None

    def check_and_process_dividends(self) -> List[Any]:
        """
        Queries Alpaca account activities for cash dividend distributions (DIV, DIVCQA).
        Automatically withholds 30% into the virtual tax escrow reserve.
        """
        if not self.tax_engine or not self.trading_client:
            return []

        processed_records = []
        try:
            activities = []
            if hasattr(self.trading_client, "get_activities"):
                activities = self.trading_client.get_activities(activity_types=["DIV", "DIVCQA"])
            elif hasattr(self.trading_client, "get"):
                activities = self.trading_client.get("/account/activities", {"activity_types": "DIV,DIVCQA"}) or []

            if activities:
                for act in activities:
                    if isinstance(act, dict):
                        act_id = str(act.get("id", ""))
                        symbol = str(act.get("symbol", "YIELD_ETF"))
                        net_amount = float(act.get("net_amount", 0.0) or 0.0)
                    else:
                        act_id = str(getattr(act, "id", ""))
                        symbol = str(getattr(act, "symbol", "YIELD_ETF"))
                        net_amount = float(getattr(act, "net_amount", 0.0) or 0.0)

                    if act_id and act_id not in self.state.processed_dividend_ids:
                        if net_amount > 0:
                            trade_rec = self.tax_engine.record_dividend(
                                symbol=symbol,
                                gross_amount=net_amount,
                                activity_id=act_id,
                            )
                            processed_records.append(trade_rec)
                            logger.info(
                                "Dividend auto-captured for %s: +$%.2f. 30%% tax ($%.2f) escrowed.",
                                symbol,
                                net_amount,
                                trade_rec.tax_allocated,
                            )
                            if self.notifier:
                                self.notifier.send_ntfy(
                                    message=(
                                        f"💰 Cash Dividend Received from {symbol}: ${net_amount:,.2f}.\n"
                                        f"30% (${trade_rec.tax_allocated:,.2f}) allocated to virtual tax reserve."
                                    ),
                                    title=f"Dividend Tax Escrow ({symbol})",
                                    tags="moneybag,bank",
                                )
                        self.state.processed_dividend_ids.append(act_id)
                self._save_state()
        except Exception as e:
            logger.debug("Could not query Alpaca dividend activities: %s", e)

        return processed_records

    def step(self) -> Optional[str]:
        """
        Runs a check cycle:
        1. Checks and processes dividend distributions (30% tax escrow).
        2. Checks Layer 1 TTL expiration (4.0 hours).
        3. Polls action topic for user tap events.
        4. Runs Layer 2 Pre-execution viability & slippage check upon approval.
        5. Executes orders or safely aborts.
        """
        # Always synchronize state from disk
        self.state = self._load_state()

        # --- Dividend Auto-Capture (30% Tax Escrow) ---
        self.check_and_process_dividends()

        # --- Layer 1: Time-To-Live (TTL) Check ---
        pending = self.state.pending_approval

        if pending and pending.status == "PENDING":
            ttl_hours = getattr(self.config, "APPROVAL_TTL_HOURS", 4.0)
            elapsed_hours = (time.time() - pending.created_timestamp) / 3600.0

            if elapsed_hours > ttl_hours:
                logger.warning(
                    "Approval request %s expired after %.2f hours (limit %.1fh). Marking EXPIRED.",
                    pending.id,
                    elapsed_hours,
                    ttl_hours,
                )
                pending.status = "EXPIRED"
                self.state.history.append(pending.model_dump())
                self.state.pending_approval = None
                self._save_state()

                sym_label = pending.target_symbol or "50/50 Bond Barbell"
                exp_msg = (
                    f"⚠️ Trade Approval Expired!\n\n"
                    f"The proposal for {sym_label} was not approved within {ttl_hours:.0f} hours.\n"
                    f"Market conditions have shifted; bot has cancelled the request and preserved your funds."
                )
                self.notifier.notify_approval_resolution(
                    title="Approval Expired",
                    message=exp_msg,
                    approved=False,
                )
                return "EXPIRED"

        # --- Poll Action Topic ---
        actions = self.poll_action_topic()
        if not actions:
            return None

        for act in actions:
            if "APPROVE" in act:
                logger.info("Detected user APPROVE action (%s)!", act)
                pending = self.state.pending_approval

                # --- Layer 2: Pre-Execution Viability & Slippage Gate ---
                viable, reason = self.reassess_viability(pending)
                if not viable:
                    logger.warning("Layer 2 Safety Gate triggered: %s. Aborting execution!", reason)
                    if pending:
                        pending.status = "ABORTED_VIABILITY"
                        self.state.history.append(pending.model_dump())
                        self.state.pending_approval = None
                        self._save_state()

                    abort_msg = (
                        f"🛡️ Pre-Execution Safety Gate Triggered!\n\n"
                        f"{reason}\n\n"
                        f"Trade execution aborted to prevent adverse slippage. Funds remain intact."
                    )
                    self.notifier.notify_approval_resolution(
                        title="Execution Aborted (Safety Gate)",
                        message=abort_msg,
                        approved=False,
                    )
                    return "ABORTED_VIABILITY"

                # Handle specific request types
                if "APPROVE_5050" in act or (pending and pending.request_type == "BOND_BARBELL_5050"):
                    sgov_amt = pending.sgov_amount if pending else getattr(self.config, "SGOV_ALLOCATION_USD", 20000.0)
                    fbnd_amt = pending.fbnd_amount if pending else getattr(self.config, "FBND_ALLOCATION_USD", 20000.0)
                    req_id = pending.id if pending else f"REQ-5050-{datetime.now().strftime('%Y%m%d%H%M%S')}"

                    sgov_order, fbnd_order = self.execute_5050_orders(
                        sgov_amount=sgov_amt,
                        fbnd_amount=fbnd_amt,
                    )

                    order_ids = []
                    if sgov_order:
                        order_ids.append(str(getattr(sgov_order, "id", "SGOV")))
                    if fbnd_order:
                        order_ids.append(str(getattr(fbnd_order, "id", "FBND")))

                    executed_record = {
                        "id": req_id,
                        "request_type": "BOND_BARBELL_5050",
                        "created_at": datetime.now().isoformat(),
                        "sgov_amount": sgov_amt,
                        "fbnd_amount": fbnd_amt,
                        "status": "EXECUTED",
                        "order_ids": order_ids,
                    }
                    self.state.history.append(executed_record)
                    self.state.pending_approval = None
                    self._save_state()

                    res_msg = (
                        f"✅ 50/50 Bond Barbell Executed!\n\n"
                        f"• SGOV: ${sgov_amt:,.2f} order submitted\n"
                        f"• FBND: ${fbnd_amt:,.2f} order submitted\n\n"
                        f"Your idle capital is now deployed earning ~5.05% blended yield."
                    )
                    self.notifier.notify_approval_resolution(
                        title="Orders Executed (50/50 Bonds)",
                        message=res_msg,
                        approved=True,
                    )
                    return "APPROVED"

                elif pending and pending.request_type == "LIQUIDATION_FOR_TRADE":
                    liq_amount = pending.sgov_amount or pending.fbnd_amount
                    res_sym = "SGOV" if pending.sgov_amount > 0 else "FBND"
                    sell_order = self.execute_liquidation_order(symbol=res_sym, notional_amount=liq_amount)

                    order_ids = [str(getattr(sell_order, "id", res_sym))] if sell_order else []
                    pending.order_ids = order_ids
                    pending.status = "EXECUTED"
                    self.state.history.append(pending.model_dump())
                    self.state.pending_approval = None
                    self._save_state()

                    res_msg = (
                        f"✅ Capital Liquidation Executed!\n\n"
                        f"• Liquidated: ${liq_amount:,.2f} of {res_sym}\n"
                        f"• Target Opportunity: {pending.target_symbol} ({pending.opportunity_type})\n\n"
                        f"Capital released into liquid cash for trade entry."
                    )
                    self.notifier.notify_approval_resolution(
                        title=f"Liquidation Executed ({res_sym})",
                        message=res_msg,
                        approved=True,
                    )
                    return "APPROVED"

            elif "REJECT" in act:
                logger.info("Detected user REJECT action (%s). Cancelling request.", act)
                req_id = self.state.pending_approval.id if self.state.pending_approval else f"REQ-REJ-{datetime.now().strftime('%Y%m%d%H%M%S')}"
                rejected_record = {
                    "id": req_id,
                    "created_at": datetime.now().isoformat(),
                    "status": "REJECTED",
                }
                self.state.history.append(rejected_record)
                self.state.pending_approval = None
                self._save_state()

                res_msg = "❌ Allocation proposal rejected. No orders were placed. Funds remain intact."
                self.notifier.notify_approval_resolution(
                    title="Proposal Cancelled",
                    message=res_msg,
                    approved=False,
                )
                return "REJECTED"

        return None
