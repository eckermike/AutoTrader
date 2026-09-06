"""
Broker Integration Module for Alpaca Spot Crypto Paper Trading.
Wraps alpaca-py with strict Paper Trading enforcement (paper=True), fractional sizing,
and GTC (Good-Til-Cancelled) order execution for BTC/USD and spot crypto pairs.
"""

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

# Alpaca SDK imports
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import (
        MarketOrderRequest,
        LimitOrderRequest,
        GetOrdersRequest,
        ClosePositionRequest,
    )
    from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
    from alpaca.data.historical.crypto import CryptoHistoricalDataClient
    from alpaca.data.requests import CryptoBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
except ImportError:
    TradingClient = None

logger = logging.getLogger("execution.alpaca_client")


class AccountInfo(BaseModel):
    """Account status snapshot."""

    cash: float
    buying_power: float
    portfolio_value: float
    status: str


class PositionInfo(BaseModel):
    """Active cryptocurrency position details."""

    symbol: str
    qty: float
    market_value: float
    avg_entry_price: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float


class OrderResult(BaseModel):
    """Order placement receipt."""

    id: str
    client_order_id: str
    symbol: str
    side: str
    order_type: str
    qty: Optional[float] = None
    notional: Optional[float] = None
    limit_price: Optional[float] = None
    status: str
    time_in_force: str
    filled_avg_price: Optional[float] = None
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class AlpacaPaperClient:
    """
    Production-grade Alpaca wrapper strictly enforced for Paper Trading.
    Supports spot crypto pairs (BTC/USD, ETH/USD, etc.), fractional orders,
    and market/limit order placement with TimeInForce.GTC.
    """

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        paper: bool = True,
        base_url: str = "https://paper-api.alpaca.markets",
        mock_mode: bool = False,
    ):
        # Strict enforcement of paper mode
        if not paper:
            raise ValueError(
                "CRITICAL ERROR: Live trading is strictly disallowed. "
                "AlpacaPaperClient must be initialized with paper=True."
            )

        self.api_key = api_key
        self.secret_key = secret_key
        self.paper = paper
        self.base_url = base_url
        self.mock_mode = (
            mock_mode
            or api_key.startswith("MOCK")
            or not api_key
            or api_key == "your_alpaca_paper_api_key_here"
        )

        if self.mock_mode:
            logger.info("Initializing AlpacaPaperClient in MOCK / DRY-RUN mode.")
            self._mock_cash = 100000.00
            self._mock_positions: Dict[str, Dict[str, float]] = {}
            self._mock_orders: List[Dict[str, Any]] = []
            self.trading_client = None
            self.crypto_data_client = None
        else:
            logger.info("Initializing live connection to Alpaca Paper API (%s)", base_url)
            self.trading_client = TradingClient(
                api_key=api_key,
                secret_key=secret_key,
                paper=True,
                url_override=base_url,
            )
            self.crypto_data_client = CryptoHistoricalDataClient(
                api_key=api_key,
                secret_key=secret_key,
            )

    def get_account(self) -> AccountInfo:
        """Fetches account cash balance, buying power, and portfolio value."""
        if self.mock_mode:
            pos_val = sum(
                p["qty"] * p.get("current_price", p["avg_entry_price"])
                for p in self._mock_positions.values()
            )
            return AccountInfo(
                cash=round(self._mock_cash, 2),
                buying_power=round(self._mock_cash, 2),
                portfolio_value=round(self._mock_cash + pos_val, 2),
                status="ACTIVE",
            )

        account = self.trading_client.get_account()
        return AccountInfo(
            cash=float(account.cash),
            buying_power=float(account.buying_power),
            portfolio_value=float(account.portfolio_value),
            status=str(account.status),
        )

    def get_crypto_position(self, symbol: str = "BTC/USD") -> Optional[PositionInfo]:
        """Fetches the current open position for the specified crypto pair."""
        # Clean symbol representation for Alpaca query (Alpaca accepts BTC/USD or BTCUSD)
        clean_sym = symbol.replace("/", "")
        slash_sym = symbol if "/" in symbol else f"{symbol[:-3]}/{symbol[-3:]}"

        if self.mock_mode:
            pos = self._mock_positions.get(slash_sym) or self._mock_positions.get(clean_sym)
            if not pos or pos["qty"] <= 0:
                return None
            qty = pos["qty"]
            entry = pos["avg_entry_price"]
            curr = pos.get("current_price", entry)
            market_val = qty * curr
            unrealized_pnl = (curr - entry) * qty
            pct = (curr - entry) / entry if entry > 0 else 0.0
            return PositionInfo(
                symbol=slash_sym,
                qty=round(qty, 6),
                market_value=round(market_val, 2),
                avg_entry_price=round(entry, 2),
                current_price=round(curr, 2),
                unrealized_pnl=round(unrealized_pnl, 2),
                unrealized_pnl_pct=round(pct, 4),
            )

        try:
            position = self.trading_client.get_open_position(clean_sym)
            return PositionInfo(
                symbol=symbol,
                qty=float(position.qty),
                market_value=float(position.market_value),
                avg_entry_price=float(position.avg_entry_price),
                current_price=float(position.current_price),
                unrealized_pnl=float(position.unrealized_pl),
                unrealized_pnl_pct=float(position.unrealized_plpc),
            )
        except Exception as e:
            # 404 indicates position does not exist
            if "position does not exist" in str(e).lower() or "404" in str(e):
                return None
            logger.warning("Error fetching crypto position for %s: %s", symbol, e)
            return None

    def submit_market_order(
        self,
        symbol: str,
        side: str,
        notional: Optional[float] = None,
        qty: Optional[float] = None,
        estimated_price: float = 60000.0,
    ) -> OrderResult:
        """
        Submits a spot crypto market order with TimeInForce.GTC and fractional sizing.
        Accepts either 'notional' (USD amount) or 'qty' (crypto quantity).
        """
        order_side_str = side.upper()
        if order_side_str not in ("BUY", "SELL"):
            raise ValueError(f"Invalid order side: {side}. Must be 'BUY' or 'SELL'.")

        if notional is None and qty is None:
            raise ValueError("Must specify either 'notional' or 'qty' for market order.")

        client_order_id = f"auto_{uuid.uuid4().hex[:10]}"

        if self.mock_mode:
            # Mock order simulation
            order_price = estimated_price
            if notional is not None:
                calc_qty = notional / order_price
                calc_notional = notional
            else:
                calc_qty = qty
                calc_notional = qty * order_price

            if order_side_str == "BUY":
                self._mock_cash -= calc_notional
                cur_pos = self._mock_positions.get(symbol, {"qty": 0.0, "avg_entry_price": 0.0})
                new_qty = cur_pos["qty"] + calc_qty
                new_avg = (
                    (cur_pos["qty"] * cur_pos["avg_entry_price"] + calc_notional) / new_qty
                    if new_qty > 0
                    else order_price
                )
                self._mock_positions[symbol] = {
                    "qty": new_qty,
                    "avg_entry_price": new_avg,
                    "current_price": order_price,
                }
            else:
                cur_pos = self._mock_positions.get(symbol, {"qty": 0.0, "avg_entry_price": order_price})
                sell_qty = min(cur_pos["qty"], calc_qty)
                self._mock_cash += sell_qty * order_price
                cur_pos["qty"] -= sell_qty
                if cur_pos["qty"] <= 1e-8:
                    self._mock_positions.pop(symbol, None)
                else:
                    self._mock_positions[symbol] = cur_pos

            return OrderResult(
                id=f"mock_ord_{uuid.uuid4().hex[:8]}",
                client_order_id=client_order_id,
                symbol=symbol,
                side=order_side_str,
                order_type="MARKET",
                qty=round(calc_qty, 6),
                notional=round(calc_notional, 2),
                status="FILLED",
                time_in_force="GTC",
                filled_avg_price=order_price,
            )

        # Live Alpaca API order placement
        alpaca_side = OrderSide.BUY if order_side_str == "BUY" else OrderSide.SELL
        clean_symbol = symbol.replace("/", "")

        order_req = MarketOrderRequest(
            symbol=clean_symbol,
            qty=qty,
            notional=notional,
            side=alpaca_side,
            time_in_force=TimeInForce.GTC,
            client_order_id=client_order_id,
        )

        submitted_order = self.trading_client.submit_order(order_data=order_req)
        logger.info(
            "Alpaca Market Order Submitted: %s %s %s (ID: %s)",
            order_side_str,
            f"${notional:.2f}" if notional else f"{qty} units",
            symbol,
            submitted_order.id,
        )

        return OrderResult(
            id=str(submitted_order.id),
            client_order_id=str(submitted_order.client_order_id),
            symbol=symbol,
            side=order_side_str,
            order_type="MARKET",
            qty=float(submitted_order.qty) if submitted_order.qty else None,
            notional=float(submitted_order.notional) if submitted_order.notional else None,
            status=str(submitted_order.status),
            time_in_force="GTC",
            filled_avg_price=float(submitted_order.filled_avg_price)
            if submitted_order.filled_avg_price
            else None,
        )

    def submit_limit_order(
        self,
        symbol: str,
        side: str,
        limit_price: float,
        qty: Optional[float] = None,
        notional: Optional[float] = None,
    ) -> OrderResult:
        """Submits a spot crypto limit order with TimeInForce.GTC."""
        order_side_str = side.upper()
        if order_side_str not in ("BUY", "SELL"):
            raise ValueError(f"Invalid order side: {side}")

        client_order_id = f"auto_lim_{uuid.uuid4().hex[:10]}"

        if self.mock_mode:
            calc_qty = qty if qty is not None else (notional / limit_price if notional else 0.01)
            return OrderResult(
                id=f"mock_lim_{uuid.uuid4().hex[:8]}",
                client_order_id=client_order_id,
                symbol=symbol,
                side=order_side_str,
                order_type="LIMIT",
                qty=round(calc_qty, 6),
                limit_price=limit_price,
                status="NEW",
                time_in_force="GTC",
            )

        alpaca_side = OrderSide.BUY if order_side_str == "BUY" else OrderSide.SELL
        clean_symbol = symbol.replace("/", "")

        order_req = LimitOrderRequest(
            symbol=clean_symbol,
            limit_price=round(limit_price, 2),
            qty=qty,
            notional=notional,
            side=alpaca_side,
            time_in_force=TimeInForce.GTC,
            client_order_id=client_order_id,
        )

        submitted_order = self.trading_client.submit_order(order_data=order_req)
        return OrderResult(
            id=str(submitted_order.id),
            client_order_id=str(submitted_order.client_order_id),
            symbol=symbol,
            side=order_side_str,
            order_type="LIMIT",
            qty=float(submitted_order.qty) if submitted_order.qty else None,
            limit_price=float(submitted_order.limit_price),
            status=str(submitted_order.status),
            time_in_force="GTC",
        )

    def close_crypto_position(self, symbol: str = "BTC/USD") -> Optional[OrderResult]:
        """Liquidates an open cryptocurrency position."""
        if self.mock_mode:
            pos = self._mock_positions.pop(symbol, None)
            if not pos or pos["qty"] <= 0:
                return None
            qty = pos["qty"]
            curr_price = pos.get("current_price", pos["avg_entry_price"])
            proceeds = qty * curr_price
            self._mock_cash += proceeds
            return OrderResult(
                id=f"mock_close_{uuid.uuid4().hex[:8]}",
                client_order_id=f"close_{uuid.uuid4().hex[:8]}",
                symbol=symbol,
                side="SELL",
                order_type="MARKET",
                qty=round(qty, 6),
                notional=round(proceeds, 2),
                status="FILLED",
                time_in_force="GTC",
                filled_avg_price=curr_price,
            )

        clean_symbol = symbol.replace("/", "")
        try:
            closed_order = self.trading_client.close_position(clean_symbol)
            return OrderResult(
                id=str(closed_order.id),
                client_order_id=str(closed_order.client_order_id),
                symbol=symbol,
                side="SELL",
                order_type="MARKET",
                qty=float(closed_order.qty) if closed_order.qty else None,
                status=str(closed_order.status),
                time_in_force="GTC",
            )
        except Exception as e:
            logger.warning("Failed to close position for %s: %s", symbol, e)
            return None

    def get_crypto_bars(
        self,
        symbol: str = "BTC/USD",
        timeframe: Optional[TimeFrame] = None,
        limit: int = 250,
    ) -> pd.DataFrame:
        """
        Fetches historical OHLCV crypto bars. Returns DataFrame with:
        ['open', 'high', 'low', 'close', 'volume'].
        """
        if timeframe is None:
            timeframe = TimeFrame(1, TimeFrameUnit.Hour)

        if self.mock_mode:
            # Generate realistic synthetic crypto bars for testing/dry-run
            return self._generate_synthetic_bars(limit=limit)

        clean_symbol = symbol if "/" in symbol else f"{symbol[:-3]}/{symbol[-3:]}"
        now = datetime.now(timezone.utc)
        start_time = now - timedelta(hours=limit + 50)

        req = CryptoBarsRequest(
            symbol_or_symbols=clean_symbol,
            timeframe=timeframe,
            start=start_time,
            end=now,
        )

        bars = self.crypto_data_client.get_crypto_bars(req)
        df = bars.df

        if df.empty:
            logger.warning("Empty bars returned by Alpaca for %s. Generating fallback bars.", symbol)
            return self._generate_synthetic_bars(limit=limit)

        # Reset multi-index if returned by Alpaca
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(clean_symbol, level=0)

        df.columns = [c.lower() for c in df.columns]
        return df.tail(limit)

    def _generate_synthetic_bars(self, limit: int = 250, base_price: float = 65000.0) -> pd.DataFrame:
        """Generates synthetic hourly bars for offline testing and dry-run mode."""
        np.random.seed(42)
        returns = np.random.normal(0.0002, 0.008, limit)
        prices = base_price * np.exp(np.cumsum(returns))

        highs = prices * (1 + np.abs(np.random.normal(0.002, 0.002, limit)))
        lows = prices * (1 - np.abs(np.random.normal(0.002, 0.002, limit)))
        opens = np.roll(prices, 1)
        opens[0] = base_price
        volumes = np.random.uniform(50.0, 500.0, limit)

        dates = [
            datetime.now(timezone.utc) - timedelta(hours=limit - i)
            for i in range(limit)
        ]

        df = pd.DataFrame(
            {
                "open": opens,
                "high": highs,
                "low": lows,
                "close": prices,
                "volume": volumes,
            },
            index=pd.DatetimeIndex(dates),
        )
        return df
