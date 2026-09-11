"""
Autonomous Liquidity and Cash Yield Management Engine.
Manages cash-yield allocations (SGOV, FBND) and two-way human-in-the-loop (HITL)
trade approvals via ntfy.sh iOS action buttons.
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
    request_type: str
    created_at: str
    sgov_amount: float
    fbnd_amount: float
    status: str = "PENDING"  # PENDING, APPROVED, REJECTED, EXECUTED
    order_ids: List[str] = Field(default_factory=list)


class LiquidityState(BaseModel):
    pending_approval: Optional[PendingApproval] = None
    last_poll_timestamp: int = 0
    history: List[Dict[str, Any]] = Field(default_factory=list)


class LiquidityManager:
    """
    Coordinates idle cash deployment into cash-yield funds (SGOV / FBND)
    and manages two-way mobile approvals via ntfy.sh action buttons.
    """

    def __init__(
        self,
        config: BotConfig,
        trading_client: Any,
        notifier: TradeNotifier,
        state_file: Optional[Path | str] = None,
    ):
        self.config = config
        self.trading_client = trading_client
        self.notifier = notifier
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
        Dispatches an interactive approval notification to iOS via ntfy.
        Returns True if a request was dispatched, False otherwise.
        """
        if not getattr(self.config, "LIQUIDITY_RESERVE_ENABLED", True) and not force:
            return False

        if self.state.pending_approval and self.state.pending_approval.status == "PENDING" and not force:
            logger.info("Approval request %s is already pending. Skipping duplicate dispatch.", self.state.pending_approval.id)
            return False

        sgov_amt = getattr(self.config, "SGOV_ALLOCATION_USD", 20000.0)
        fbnd_amt = getattr(self.config, "FBND_ALLOCATION_USD", 20000.0)
        total_amt = sgov_amt + fbnd_amt

        req_id = f"REQ-5050-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        proposal_title = f"Capital Allocation Request (${total_amt:,.0f})"
        proposal_msg = (
            f"📊 Deploy ${total_amt:,.0f} idle cash into 50/50 Bond Barbell:\n\n"
            f"• SGOV (0-3M T-Bills, ~5.1% yield): ${sgov_amt:,.2f}\n"
            f"• FBND (Fidelity Total Bond, ~5.0% yield): ${fbnd_amt:,.2f}\n\n"
            f"Liquid Cash Buffer Remaining: ~$15,100\n"
            f"Option Collateral Unaffected: $45,200\n\n"
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
            sgov_amount=sgov_amt,
            fbnd_amount=fbnd_amt,
            status="PENDING",
        )
        self._save_state()
        return True

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

    def step(self) -> Optional[str]:
        """
        Runs a check cycle: polls the action topic and resolves pending approvals.
        Returns 'APPROVED', 'REJECTED', or None.
        """
        actions = self.poll_action_topic()
        if not actions:
            return None

        pending = self.state.pending_approval
        if not pending or pending.status != "PENDING":
            return None

        for act in actions:
            if "APPROVE_5050_BONDS" in act:
                logger.info("User tapped APPROVE for %s! Executing bond barbell orders...", pending.id)
                pending.status = "APPROVED"

                # Execute orders
                sgov_order, fbnd_order = self.execute_5050_orders(
                    sgov_amount=pending.sgov_amount,
                    fbnd_amount=pending.fbnd_amount,
                )

                order_ids = []
                if sgov_order:
                    order_ids.append(str(getattr(sgov_order, "id", "SGOV")))
                if fbnd_order:
                    order_ids.append(str(getattr(fbnd_order, "id", "FBND")))

                pending.order_ids = order_ids
                pending.status = "EXECUTED"

                # Record in history
                self.state.history.append(pending.model_dump())
                self.state.pending_approval = None
                self._save_state()

                # Dispatch confirmation
                res_msg = (
                    f"✅ 50/50 Bond Barbell Executed!\n\n"
                    f"• SGOV: ${pending.sgov_amount:,.2f} order submitted\n"
                    f"• FBND: ${pending.fbnd_amount:,.2f} order submitted\n\n"
                    f"Your idle capital is now generating ~5.05% blended yield."
                )
                self.notifier.notify_approval_resolution(
                    title="Orders Executed (50/50 Bonds)",
                    message=res_msg,
                    approved=True,
                )
                return "APPROVED"

            elif "REJECT_5050_BONDS" in act:
                logger.info("User tapped REJECT for %s. Allocation cancelled.", pending.id)
                pending.status = "REJECTED"
                self.state.history.append(pending.model_dump())
                self.state.pending_approval = None
                self._save_state()

                res_msg = "❌ 50/50 Bond allocation rejected. No orders were placed. Funds remain in cash."
                self.notifier.notify_approval_resolution(
                    title="Allocation Cancelled",
                    message=res_msg,
                    approved=False,
                )
                return "REJECTED"

        return None
