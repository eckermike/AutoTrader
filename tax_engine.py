"""
Virtual Tax Escrow Engine.
Tracks realized crypto trading gains and losses, sets aside a virtual tax escrow
(30% by default), applies tax credits on losses floored at $0.00, and enforces
a hard capital gate preventing order sizes from exceeding Tradable Cash.
"""

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger("tax_engine")


class InsufficientTradableCashError(Exception):
    """Raised when an order request exceeds available tradable cash after tax reserve escrow."""

    def __init__(self, requested_amount: float, tradable_cash: float, tax_reserve: float, total_cash: float):
        super().__init__(
            f"Capital Gate Rejection: Requested ${requested_amount:.2f} exceeds "
            f"Tradable Cash ${tradable_cash:.2f} (Total Cash: ${total_cash:.2f}, "
            f"Tax Escrow: ${tax_reserve:.2f})"
        )
        self.requested_amount = requested_amount
        self.tradable_cash = tradable_cash
        self.tax_reserve = tax_reserve
        self.total_cash = total_cash


class TradeRecord(BaseModel):
    """Immutable audit record of a closed trade and its tax escrow allocation."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    symbol: str
    side: str
    qty: float
    entry_price: float
    exit_price: float
    fee: float = 0.0
    gross_pnl: float
    tax_allocated: float = 0.0
    tax_credit: float = 0.0
    reserve_after: float


class TaxReserveState(BaseModel):
    """State schema for persistent tax escrow tracking."""

    tax_reserve: float = 0.0
    tax_rate: float = 0.30
    total_realized_profit: float = 0.0
    total_realized_loss: float = 0.0
    total_tax_allocated: float = 0.0
    total_tax_credits: float = 0.0
    trade_count: int = 0
    trade_history: List[TradeRecord] = Field(default_factory=list)
    last_updated: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class TaxEngine:
    """
    Virtual Tax Escrow Engine.
    Maintains persistent JSON state of cumulative tax liabilities and calculates
    tradable cash dynamically.
    """

    def __init__(self, filepath: Path | str = "tax_reserve.json", tax_rate: float = 0.30):
        self.filepath = Path(filepath)
        self.tax_rate = tax_rate
        self.state = self._load_state()

    def _load_state(self) -> TaxReserveState:
        """Loads state from JSON file or initializes new state."""
        if self.filepath.exists():
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return TaxReserveState.model_validate(data)
            except Exception as e:
                logger.warning(
                    "Failed to parse %s (%s). Re-initializing fresh state.",
                    self.filepath,
                    e,
                )
        fresh_state = TaxReserveState(tax_rate=self.tax_rate)
        self._save_state(fresh_state)
        return fresh_state

    def _save_state(self, state: Optional[TaxReserveState] = None) -> None:
        """Atomically persists state to the JSON file to prevent partial writes."""
        if state is None:
            state = self.state
        state.last_updated = datetime.now(timezone.utc).isoformat()
        
        temp_file = self.filepath.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(state.model_dump_json(indent=2))
        os.replace(temp_file, self.filepath)
        logger.debug("Tax reserve state saved to %s", self.filepath)

    @property
    def current_reserve(self) -> float:
        """Current virtual tax escrow reserve in USD."""
        return round(self.state.tax_reserve, 2)

    def calculate_tradable_cash(self, alpaca_cash_balance: float) -> float:
        """
        Hard capital gate calculation:
        Tradable Cash = Alpaca Cash Balance - Current Tax Reserve.
        Never drops below 0.0.
        """
        tradable = max(0.0, alpaca_cash_balance - self.state.tax_reserve)
        return round(tradable, 2)

    def validate_order_budget(self, alpaca_cash_balance: float, order_cost: float) -> float:
        """
        Validates whether the proposed order cost is within tradable cash.
        Raises InsufficientTradableCashError if violated.
        Returns available tradable cash.
        """
        tradable_cash = self.calculate_tradable_cash(alpaca_cash_balance)
        if order_cost > tradable_cash:
            raise InsufficientTradableCashError(
                requested_amount=order_cost,
                tradable_cash=tradable_cash,
                tax_reserve=self.current_reserve,
                total_cash=alpaca_cash_balance,
            )
        return tradable_cash

    def record_closed_trade(
        self,
        symbol: str,
        side: str,
        qty: float,
        entry_price: float,
        exit_price: float,
        fee: float = 0.0,
    ) -> TradeRecord:
        """
        Processes a closed trade:
        - Profitable: Allocates tax_rate * net_pnl into tax_reserve.
        - Loss: Applies tax credit against tax_reserve (strictly floored at $0.00).
        """
        # Calculate gross and net PnL (assuming long position exit)
        if side.upper() in ("SELL", "LONG_EXIT"):
            gross_pnl = (exit_price - entry_price) * qty - fee
        else:
            # Short position exit (if supported in future)
            gross_pnl = (entry_price - exit_price) * qty - fee

        tax_allocated = 0.0
        tax_credit = 0.0

        if gross_pnl > 0:
            tax_allocated = gross_pnl * self.tax_rate
            self.state.tax_reserve += tax_allocated
            self.state.total_realized_profit += gross_pnl
            self.state.total_tax_allocated += tax_allocated
            logger.info(
                "Profitable trade closed (+${:,.2f}). Allocated ${:,.2f} ({:.0%}) to tax reserve. New Reserve: ${:,.2f}",
                gross_pnl,
                tax_allocated,
                self.tax_rate,
                self.state.tax_reserve,
            )
        elif gross_pnl < 0:
            loss_magnitude = abs(gross_pnl)
            potential_credit = loss_magnitude * self.tax_rate
            # Floor tax reserve at $0.00
            actual_credit = min(self.state.tax_reserve, potential_credit)
            self.state.tax_reserve = max(0.0, self.state.tax_reserve - potential_credit)
            self.state.total_realized_loss += loss_magnitude
            self.state.total_tax_credits += actual_credit
            tax_credit = actual_credit
            logger.info(
                "Losing trade closed (-${:,.2f}). Applied ${:,.2f} tax credit against reserve. New Reserve: ${:,.2f}",
                loss_magnitude,
                actual_credit,
                self.state.tax_reserve,
            )
        else:
            logger.info("Breakeven trade closed ($0.00 PnL). No tax reserve adjustments.")

        self.state.tax_reserve = round(self.state.tax_reserve, 4)
        self.state.trade_count += 1

        record = TradeRecord(
            symbol=symbol,
            side=side,
            qty=qty,
            entry_price=entry_price,
            exit_price=exit_price,
            fee=fee,
            gross_pnl=round(gross_pnl, 2),
            tax_allocated=round(tax_allocated, 2),
            tax_credit=round(tax_credit, 2),
            reserve_after=round(self.state.tax_reserve, 2),
        )

        self.state.trade_history.append(record)
        self._save_state()
        return record

    def record_option_premium(
        self,
        symbol: str,
        contract_symbol: str,
        contracts: int,
        premium_total: float,
    ) -> TradeRecord:
        """
        Records option premium collected upon selling a Cash-Secured Put or Covered Call.
        Allocates 30% of collected income to virtual tax escrow.
        """
        tax_allocated = premium_total * self.tax_rate
        self.state.tax_reserve += tax_allocated
        self.state.total_realized_profit += premium_total
        self.state.total_tax_allocated += tax_allocated
        self.state.trade_count += 1

        record = TradeRecord(
            symbol=contract_symbol,
            side="SELL_TO_OPEN",
            qty=float(contracts),
            entry_price=premium_total / (contracts * 100) if contracts > 0 else 0.0,
            exit_price=0.0,
            fee=0.0,
            gross_pnl=round(premium_total, 2),
            tax_allocated=round(tax_allocated, 2),
            tax_credit=0.0,
            reserve_after=round(self.state.tax_reserve, 2),
        )
        self.state.trade_history.append(record)
        self._save_state()
        logger.info(
            "Option Premium Recorded: %s (%d contracts, +$%0.2f). Tax Allocated: $%0.2f. New Reserve: $%0.2f",
            contract_symbol,
            contracts,
            premium_total,
            tax_allocated,
            self.state.tax_reserve,
        )
        return record

    def record_option_close(
        self,
        contract_symbol: str,
        contracts: int,
        open_premium: float,
        close_cost: float,
    ) -> TradeRecord:
        """
        Records closing an option early (e.g. at 50% profit target Buy-to-Close).
        Adjusts tax reserve and realized PnL.
        """
        net_profit = open_premium - close_cost
        return self.record_closed_trade(
            symbol=contract_symbol,
            side="BUY_TO_CLOSE",
            qty=float(contracts),
            entry_price=open_premium / (contracts * 100) if contracts > 0 else 0.0,
            exit_price=close_cost / (contracts * 100) if contracts > 0 else 0.0,
        )

    def get_summary(self) -> Dict[str, Any]:
        """Returns diagnostic summary of current tax escrow and cumulative performance."""
        return {
            "tax_reserve": self.current_reserve,
            "tax_rate": self.state.tax_rate,
            "total_realized_profit": round(self.state.total_realized_profit, 2),
            "total_realized_loss": round(self.state.total_realized_loss, 2),
            "net_realized_pnl": round(
                self.state.total_realized_profit - self.state.total_realized_loss, 2
            ),
            "total_tax_allocated": round(self.state.total_tax_allocated, 2),
            "total_tax_credits": round(self.state.total_tax_credits, 2),
            "trade_count": self.state.trade_count,
            "last_updated": self.state.last_updated,
        }
