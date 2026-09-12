"""
Alpaca Options Broker Client.
Wraps alpaca-py options endpoints with strict Paper Trading enforcement (paper=True).
Supports querying option chains, stock quotes, positions, and executing
Cash-Secured Puts and Covered Calls via Market/Limit orders with PositionIntent.
"""

import logging
import uuid
from datetime import date, datetime, timezone, timedelta
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
        GetOptionContractsRequest,
        ClosePositionRequest,
        GetOrdersRequest,
        OptionLegRequest,
    )
    from alpaca.trading.enums import (
        OrderSide,
        TimeInForce,
        ContractType,
        PositionIntent,
        ExerciseStyle,
        QueryOrderStatus,
        OrderClass,
        OrderType,
    )
    from alpaca.data.historical.stock import StockHistoricalDataClient
    from alpaca.data.requests import (
        StockLatestQuoteRequest,
        StockLatestTradeRequest,
        StockBarsRequest,
    )
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.data.historical.option import OptionHistoricalDataClient
    from alpaca.data.requests import OptionLatestQuoteRequest
except ImportError:
    TradingClient = None
    OptionLegRequest = None
    OrderClass = None
    OrderType = None
    StockBarsRequest = None
    TimeFrame = None
    TimeFrameUnit = None

logger = logging.getLogger("options.client")


class OptionContractInfo(BaseModel):
    """Normalized option contract details."""

    symbol: str
    underlying: str
    contract_type: str  # 'put' or 'call'
    strike_price: float
    expiration_date: str
    days_to_expiration: int
    bid: float = 0.0
    ask: float = 0.0
    mid_price: float = 0.0


class OptionPositionInfo(BaseModel):
    """Active option contract position."""

    symbol: str
    underlying: str
    contract_type: str
    strike_price: float
    expiration_date: str
    qty: int  # Negative for short (CSP / CC)
    avg_entry_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float


class StockPositionInfo(BaseModel):
    """Active underlying equity position."""

    symbol: str
    qty: float
    avg_entry_price: float
    current_price: float
    market_value: float


class AlpacaOptionsClient:
    """
    Production-grade options execution wrapper for Alpaca Paper Trading.
    Supports option chain discovery, pricing, and multi-leg order intents.
    """

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        paper: bool = True,
        base_url: str = "https://paper-api.alpaca.markets",
        mock_mode: bool = False,
    ):
        if not paper:
            raise ValueError(
                "CRITICAL SECURITY REJECTION: Live trading is disallowed. "
                "AlpacaOptionsClient must have paper=True."
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

        self._price_cache: Dict[str, float] = {}

        if self.mock_mode:
            logger.info("Initializing AlpacaOptionsClient in MOCK simulation mode.")
            self.trading_client = None
            self.stock_data_client = None
            self.option_data_client = None
            # Mock state
            self._mock_cash = 100000.00
            self._mock_stock_positions: Dict[str, StockPositionInfo] = {}
            self._mock_option_positions: Dict[str, OptionPositionInfo] = {}
            self._mock_orders: List[Dict[str, Any]] = []
        else:
            logger.info("Connecting AlpacaOptionsClient to Alpaca Paper Trading (%s)", base_url)
            self.trading_client = TradingClient(
                api_key=api_key,
                secret_key=secret_key,
                paper=True,
                url_override=base_url,
            )
            self.stock_data_client = StockHistoricalDataClient(
                api_key=api_key,
                secret_key=secret_key,
            )
            self.option_data_client = OptionHistoricalDataClient(
                api_key=api_key,
                secret_key=secret_key,
            )

    def is_market_open(self) -> bool:
        """Checks if the US equity options market is currently open via Alpaca clock."""
        if self.mock_mode:
            return True
        try:
            clock = self.trading_client.get_clock()
            return bool(getattr(clock, "is_open", True))
        except Exception as e:
            logger.debug("Failed to query market clock: %s", e)
            return True

    def get_stock_price(self, symbol: str = "INTC") -> float:
        """Fetches the latest equity price for the underlying asset."""
        sym = symbol.upper()
        if self.mock_mode:
            mock_prices = {
                "INTC": 21.50,
                "F": 12.00,
                "SOFI": 8.50,
                "HOOD": 22.00,
                "PLTR": 32.00,
                "XLF": 44.00,
                "SPY": 560.00,
                "QQQ": 485.00,
                "IWM": 220.00,
                "AAPL": 225.00,
                "MSFT": 420.00,
                "GOOGL": 165.00,
                "AMZN": 185.00,
                "NVDA": 115.00,
                "GLD": 235.00,
                "VNQ": 88.00,
                "SGOV": 100.50,
            }
            return mock_prices.get(sym, 21.50)

        # 1. Try real-time quote (ask/bid midpoint)
        try:
            req = StockLatestQuoteRequest(symbol_or_symbols=sym)
            quotes = self.stock_data_client.get_stock_latest_quote(req)
            q = quotes.get(sym)
            if q and q.ask_price > 0 and q.bid_price > 0:
                mid = round((q.ask_price + q.bid_price) / 2.0, 2)
                self._price_cache[sym] = mid
                return mid
            elif q and q.ask_price > 0:
                price = round(float(q.ask_price), 2)
                self._price_cache[sym] = price
                return price
        except Exception as e:
            logger.debug("Live quote request error for %s: %s", sym, e)

        # 2. Fallback to latest trade price (reliable even when market is closed)
        try:
            treq = StockLatestTradeRequest(symbol_or_symbols=sym)
            trades = self.stock_data_client.get_stock_latest_trade(treq)
            t = trades.get(sym)
            if t and t.price > 0:
                trade_price = round(float(t.price), 2)
                self._price_cache[sym] = trade_price
                return trade_price
        except Exception as e:
            logger.debug("Latest trade price request error for %s: %s", sym, e)

        # 3. Fallback to cached price or static fallback
        if sym in self._price_cache:
            return self._price_cache[sym]

        default_fallbacks = {
            "INTC": 95.80,
            "F": 14.62,
            "SOFI": 18.21,
            "HOOD": 122.06,
            "PLTR": 174.25,
            "XLF": 58.15,
            "SPY": 560.00,
            "QQQ": 485.00,
            "IWM": 220.00,
            "AAPL": 225.00,
            "MSFT": 420.00,
            "GOOGL": 165.00,
            "AMZN": 185.00,
            "NVDA": 115.00,
            "GLD": 235.00,
            "VNQ": 88.00,
            "SGOV": 100.50,
        }
        fallback = default_fallbacks.get(sym, 21.50)
        logger.warning("Using fallback price $%0.2f for %s", fallback, sym)
        return fallback

    def get_stock_position(self, symbol: str = "INTC") -> Optional[StockPositionInfo]:
        """Queries whether we currently own shares of the underlying stock."""
        if self.mock_mode:
            return self._mock_stock_positions.get(symbol)

        try:
            pos = self.trading_client.get_open_position(symbol)
            qty = float(pos.qty)
            if qty > 0:
                return StockPositionInfo(
                    symbol=symbol,
                    qty=qty,
                    avg_entry_price=float(pos.avg_entry_price),
                    current_price=float(pos.current_price),
                    market_value=float(pos.market_value),
                )
        except Exception as e:
            if "position does not exist" not in str(e).lower() and "404" not in str(e):
                logger.debug("Error checking stock position for %s: %s", symbol, e)
        return None

    def get_option_contracts(
        self,
        underlying: str = "INTC",
        contract_type: str = "put",
        min_dte: int = 21,
        max_dte: int = 45,
    ) -> List[OptionContractInfo]:
        """
        Queries active option contracts for the underlying within target DTE window.
        """
        today = date.today()
        ctype = ContractType.PUT if contract_type.lower() == "put" else ContractType.CALL

        if self.mock_mode:
            # Generate synthetic realistic contracts for testing
            sym = underlying.upper()
            base_price = self.get_stock_price(sym)
            contracts = []
            target_dte = min(max_dte, max(min_dte, 33))
            exp_date = today + timedelta(days=target_dte)
            exp_date_str = exp_date.strftime("%Y-%m-%d")
            dte = target_dte
            if sym == "INTC":
                strikes = [19.0, 19.5, 20.0, 20.5, 21.0, 21.5, 22.0, 22.5, 23.0]
            elif sym in ["SPY", "QQQ", "IWM"]:
                step = 5.0
                base_rounded = round(base_price / step) * step
                strikes = [round(base_rounded + i * step, 1) for i in range(-25, 6)]
            else:
                strikes = sorted(set([round(base_price * m, 1) for m in [0.80, 0.85, 0.88, 0.90, 0.92, 0.95, 0.98, 1.00, 1.02, 1.05, 1.08]]))
            for strike in strikes:
                if sym in ["SPY", "QQQ", "IWM"]:
                    diff = strike - base_price
                    if contract_type == "put":
                        # OTM put has strike < base_price
                        dist_pct = (base_price - strike) / base_price
                        prem = max(0.15, round(6.0 * max(0.02, (1.0 - dist_pct * 6)), 2))
                    else:
                        dist_pct = (strike - base_price) / base_price
                        prem = max(0.20, round(6.0 * max(0.05, (1.0 - dist_pct * 12)), 2))
                else:
                    prem = max(0.20, round(abs(strike - base_price) * 0.4 + 0.35, 2))
                contracts.append(
                    OptionContractInfo(
                        symbol=f"{underlying}261009{'P' if contract_type == 'put' else 'C'}{int(strike * 1000):08d}",
                        underlying=underlying,
                        contract_type=contract_type.lower(),
                        strike_price=strike,
                        expiration_date=exp_date_str,
                        days_to_expiration=dte,
                        bid=round(prem - 0.05, 2),
                        ask=round(prem + 0.05, 2),
                        mid_price=prem,
                    )
                )
            return contracts

        try:
            req = GetOptionContractsRequest(
                underlying_symbols=[underlying],
                status="active",
                expiration_date_gte=today + timedelta(days=min_dte),
                expiration_date_lte=today + timedelta(days=max_dte),
                type=ctype,
                limit=100,
            )
            resp = self.trading_client.get_option_contracts(req)
            contract_list = resp.option_contracts if hasattr(resp, "option_contracts") else resp

            results: List[OptionContractInfo] = []
            for c in contract_list:
                if c.type != ctype:
                    continue

                exp_dt = c.expiration_date
                if isinstance(exp_dt, str):
                    exp_date = datetime.strptime(exp_dt, "%Y-%m-%d").date()
                else:
                    exp_date = exp_dt

                dte = (exp_date - today).days
                if min_dte <= dte <= max_dte:
                    strike = float(c.strike_price)
                    # Estimate mid price or 0.50
                    prem = 0.50
                    results.append(
                        OptionContractInfo(
                            symbol=c.symbol,
                            underlying=underlying,
                            contract_type=contract_type.lower(),
                            strike_price=strike,
                            expiration_date=str(exp_date),
                            days_to_expiration=dte,
                            bid=round(prem - 0.05, 2),
                            ask=round(prem + 0.05, 2),
                            mid_price=prem,
                        )
                    )

            # Sort by strike price
            results.sort(key=lambda x: x.strike_price)
            return results

        except Exception as e:
            logger.warning("Failed to retrieve option contracts for %s: %s", underlying, e)
            return []

    def get_active_option_positions(self, underlying: str = "INTC") -> List[OptionPositionInfo]:
        """Fetches active option positions matching the underlying asset."""
        if self.mock_mode:
            return [
                p for p in self._mock_option_positions.values()
                if p.underlying.upper() == underlying.upper() or p.symbol.upper().startswith(underlying.upper())
            ]

        try:
            positions = self.trading_client.get_all_positions()
            results: List[OptionPositionInfo] = []
            for p in positions:
                sym = p.symbol
                # Option symbols typically follow OCC format e.g. INTC260918P00020000
                if sym.startswith(underlying) and len(sym) > len(underlying) + 4:
                    c_type = "call" if "C" in sym[len(underlying):] else "put"
                    qty = int(float(p.qty))
                    results.append(
                        OptionPositionInfo(
                            symbol=sym,
                            underlying=underlying,
                            contract_type=c_type,
                            strike_price=float(p.avg_entry_price),  # fallback
                            expiration_date="",
                            qty=qty,
                            avg_entry_price=float(p.avg_entry_price),
                            current_price=float(p.current_price),
                            market_value=float(p.market_value),
                            unrealized_pnl=float(p.unrealized_pl),
                        )
                    )
            return results
        except Exception as e:
            logger.warning("Error fetching active option positions: %s", e)
            return []

    def get_open_orders(self, underlying: str = "INTC") -> List[Any]:
        """Fetches pending open orders for underlying equity or options."""
        if self.mock_mode:
            return []

        try:
            req = GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=50)
            orders = self.trading_client.get_orders(filter=req)
            return [o for o in orders if getattr(o, "symbol", "").startswith(underlying)]
        except Exception as e:
            logger.debug("Error checking open orders for %s: %s", underlying, e)
            return []

    def get_option_quote(self, symbol: str) -> Optional[Dict[str, float]]:
        """
        Fetches live real-time bid, ask, and midpoint quotes for a specific option contract.
        """
        if self.mock_mode:
            return {"bid_price": 0.45, "ask_price": 0.55, "mid_price": 0.50}

        try:
            if self.option_data_client:
                req = OptionLatestQuoteRequest(symbol_or_symbols=symbol)
                quotes = self.option_data_client.get_option_latest_quote(req)
                q = quotes.get(symbol)
                if q:
                    bid = float(q.bid_price) if getattr(q, "bid_price", None) is not None else 0.0
                    ask = float(q.ask_price) if getattr(q, "ask_price", None) is not None else 0.0
                    if bid > 0 and ask > 0:
                        mid = round((bid + ask) / 2.0, 2)
                    elif bid > 0:
                        mid = round(bid, 2)
                    elif ask > 0:
                        mid = round(ask, 2)
                    else:
                        mid = 0.0
                    return {"bid_price": bid, "ask_price": ask, "mid_price": mid}
        except Exception as e:
            logger.debug("Live option quote error for %s: %s", symbol, e)

        return None

    def cancel_order(self, order_id: str) -> bool:
        """Cancels an open order by ID."""
        if self.mock_mode:
            logger.info("Mock mode: cancelled order %s", order_id)
            return True

        try:
            self.trading_client.cancel_order_by_id(order_id)
            logger.info("Cancelled open option order %s", order_id)
            return True
        except Exception as e:
            logger.warning("Failed to cancel order %s: %s", order_id, e)
            return False

    def cancel_open_orders_for_symbol(self, underlying: str) -> List[str]:
        """Cancels all pending open orders matching the underlying ticker."""
        open_orders = self.get_open_orders(underlying)
        cancelled_ids = []
        for o in open_orders:
            oid = str(getattr(o, "id", ""))
            if oid and self.cancel_order(oid):
                cancelled_ids.append(oid)
        return cancelled_ids

    def submit_option_order(
        self,
        symbol: str,
        side: str,
        position_intent: str,
        qty: int = 1,
        limit_price: Optional[float] = None,
        time_in_force: str = "DAY",
    ) -> Dict[str, Any]:
        """
        Submits an option order (e.g. SELL_TO_OPEN or BUY_TO_CLOSE).
        Supports DAY or GTC time-in-force.
        """
        client_order_id = f"opt_{uuid.uuid4().hex[:10]}"
        side_upper = side.upper()
        intent_upper = position_intent.upper()
        tif_upper = time_in_force.upper()

        if self.mock_mode:
            price = limit_price or 0.50
            und = "SPY"
            for known_sym in ["INTC", "SOFI", "HOOD", "PLTR", "XLF", "F", "SPY", "QQQ", "IWM"]:
                if symbol.upper().startswith(known_sym):
                    und = known_sym
                    break

            if intent_upper == "BUY_TO_OPEN":
                # Open long option position (e.g. Tail-Risk Put)
                self._mock_option_positions[symbol] = OptionPositionInfo(
                    symbol=symbol,
                    underlying=und,
                    contract_type="put" if "P" in symbol else "call",
                    strike_price=450.0 if und in ["SPY", "QQQ"] else 20.0,
                    expiration_date="2026-11-20",
                    qty=qty,
                    avg_entry_price=price,
                    current_price=price,
                    market_value=qty * price * 100,
                    unrealized_pnl=0.0,
                )
            elif intent_upper == "SELL_TO_OPEN" or ("SELL" in side_upper and intent_upper != "SELL_TO_CLOSE"):
                # Open short option position (e.g. Cash-Secured Put / Covered Call)
                self._mock_option_positions[symbol] = OptionPositionInfo(
                    symbol=symbol,
                    underlying=und,
                    contract_type="put" if "P" in symbol else "call",
                    strike_price=20.0,
                    expiration_date="2026-10-09",
                    qty=-qty,
                    avg_entry_price=price,
                    current_price=price,
                    market_value=-qty * price * 100,
                    unrealized_pnl=0.0,
                )
            else:
                # Close option position (BUY_TO_CLOSE or SELL_TO_CLOSE)
                self._mock_option_positions.pop(symbol, None)

            return {
                "id": f"mock_opt_{uuid.uuid4().hex[:8]}",
                "client_order_id": client_order_id,
                "symbol": symbol,
                "qty": qty,
                "side": side_upper,
                "position_intent": intent_upper,
                "filled_avg_price": price,
                "status": "FILLED",
            }

        # Alpaca live paper option order placement
        alpaca_side = OrderSide.BUY if side_upper == "BUY" else OrderSide.SELL
        alpaca_intent = getattr(PositionIntent, intent_upper, PositionIntent.SELL_TO_OPEN)
        alpaca_tif = TimeInForce.DAY if tif_upper == "DAY" else TimeInForce.GTC

        if limit_price is not None:
            order_req = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=alpaca_side,
                time_in_force=alpaca_tif,
                limit_price=round(limit_price, 2),
                position_intent=alpaca_intent,
                client_order_id=client_order_id,
            )
        else:
            order_req = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=alpaca_side,
                time_in_force=alpaca_tif,
                position_intent=alpaca_intent,
                client_order_id=client_order_id,
            )

        order_res = self.trading_client.submit_order(order_data=order_req)
        logger.info(
            "Alpaca Option Order Submitted: %s %d %s (ID: %s, Intent: %s, TIF: %s)",
            side_upper,
            qty,
            symbol,
            order_res.id,
            intent_upper,
            tif_upper,
        )

        return {
            "id": str(order_res.id),
            "client_order_id": str(order_res.client_order_id),
            "symbol": symbol,
            "qty": int(order_res.qty) if order_res.qty else qty,
            "side": side_upper,
            "position_intent": intent_upper,
            "status": str(order_res.status),
        }

    def get_spread_quote(self, short_symbol: str, long_symbol: str) -> Optional[Dict[str, float]]:
        """
        Fetches live bid, ask, and mid prices for both legs of a spread and calculates
        net credit to enter and net debit to close.
        """
        if self.mock_mode:
            return {
                "short_bid": 1.45,
                "short_ask": 1.55,
                "long_bid": 0.55,
                "long_ask": 0.65,
                "entry_credit": 0.80,  # short_bid - long_ask
                "close_debit": 1.00,   # short_ask - long_bid
                "mid_credit": 0.90,    # short_mid - long_mid
            }

        short_q = self.get_option_quote(short_symbol)
        long_q = self.get_option_quote(long_symbol)

        if not short_q or not long_q:
            return None

        short_bid = short_q.get("bid_price", 0.0)
        short_ask = short_q.get("ask_price", 0.0)
        long_bid = long_q.get("bid_price", 0.0)
        long_ask = long_q.get("ask_price", 0.0)

        short_mid = short_q.get("mid_price", 0.0)
        long_mid = long_q.get("mid_price", 0.0)

        entry_credit = max(0.0, round(short_bid - long_ask, 2))
        close_debit = max(0.0, round(short_ask - long_bid, 2))
        mid_credit = max(0.0, round(short_mid - long_mid, 2))

        return {
            "short_bid": short_bid,
            "short_ask": short_ask,
            "long_bid": long_bid,
            "long_ask": long_ask,
            "entry_credit": entry_credit,
            "close_debit": close_debit,
            "mid_credit": mid_credit,
        }

    def submit_mleg_spread_order(
        self,
        short_symbol: str,
        long_symbol: str,
        qty: int = 1,
        limit_credit: Optional[float] = None,
        time_in_force: str = "DAY",
    ) -> Dict[str, Any]:
        """
        Submits a multi-leg credit spread order (OrderClass.MLEG).
        Leg 1: Short Put (SELL_TO_OPEN)
        Leg 2: Long Put (BUY_TO_OPEN)
        In Alpaca MLEG, a credit limit order is specified as a negative limit_price.
        """
        client_order_id = f"mleg_{uuid.uuid4().hex[:10]}"
        tif_upper = time_in_force.upper()
        alpaca_tif = TimeInForce.DAY if tif_upper == "DAY" else TimeInForce.GTC

        if self.mock_mode:
            fill_credit = limit_credit if limit_credit is not None else 0.90
            logger.info("MOCK MLEG Spread Order: Sell %s / Buy %s x %d (Credit: $%0.2f)", short_symbol, long_symbol, qty, fill_credit)
            return {
                "id": f"mock_mleg_{uuid.uuid4().hex[:8]}",
                "client_order_id": client_order_id,
                "short_symbol": short_symbol,
                "long_symbol": long_symbol,
                "qty": qty,
                "credit": fill_credit,
                "status": "FILLED",
                "order_class": "mleg",
            }

        legs = [
            OptionLegRequest(
                symbol=short_symbol,
                ratio_qty=1.0,
                side=OrderSide.SELL,
                position_intent=PositionIntent.SELL_TO_OPEN,
            ),
            OptionLegRequest(
                symbol=long_symbol,
                ratio_qty=1.0,
                side=OrderSide.BUY,
                position_intent=PositionIntent.BUY_TO_OPEN,
            ),
        ]

        if limit_credit is not None:
            # Negative limit price in Alpaca represents net credit received
            alpaca_limit_price = -round(abs(limit_credit), 2)
            order_req = LimitOrderRequest(
                qty=qty,
                order_class=OrderClass.MLEG,
                legs=legs,
                time_in_force=alpaca_tif,
                limit_price=alpaca_limit_price,
                client_order_id=client_order_id,
            )
        else:
            order_req = MarketOrderRequest(
                qty=qty,
                order_class=OrderClass.MLEG,
                legs=legs,
                time_in_force=alpaca_tif,
                client_order_id=client_order_id,
            )

        order_res = self.trading_client.submit_order(order_data=order_req)
        logger.info(
            "Alpaca MLEG Credit Spread Submitted: Short %s / Long %s x %d (ID: %s, Credit: $%s)",
            short_symbol,
            long_symbol,
            qty,
            order_res.id,
            limit_credit,
        )

        return {
            "id": str(order_res.id),
            "client_order_id": str(order_res.client_order_id),
            "short_symbol": short_symbol,
            "long_symbol": long_symbol,
            "qty": int(order_res.qty) if order_res.qty else qty,
            "credit": limit_credit,
            "status": str(order_res.status),
            "order_class": "mleg",
        }

    def close_mleg_spread_order(
        self,
        short_symbol: str,
        long_symbol: str,
        qty: int = 1,
        limit_debit: Optional[float] = None,
        time_in_force: str = "DAY",
    ) -> Dict[str, Any]:
        """
        Submits an MLEG closing order to Buy-to-Close Short Put and Sell-to-Close Long Put.
        In Alpaca MLEG, debit price is specified as a positive limit_price.
        """
        client_order_id = f"mleg_close_{uuid.uuid4().hex[:10]}"
        tif_upper = time_in_force.upper()
        alpaca_tif = TimeInForce.DAY if tif_upper == "DAY" else TimeInForce.GTC

        if self.mock_mode:
            fill_debit = limit_debit if limit_debit is not None else 0.45
            logger.info("MOCK MLEG Close Spread: Buy %s / Sell %s x %d (Debit: $%0.2f)", short_symbol, long_symbol, qty, fill_debit)
            return {
                "id": f"mock_mleg_close_{uuid.uuid4().hex[:8]}",
                "client_order_id": client_order_id,
                "short_symbol": short_symbol,
                "long_symbol": long_symbol,
                "qty": qty,
                "debit": fill_debit,
                "status": "FILLED",
                "order_class": "mleg",
            }

        legs = [
            OptionLegRequest(
                symbol=short_symbol,
                ratio_qty=1.0,
                side=OrderSide.BUY,
                position_intent=PositionIntent.BUY_TO_CLOSE,
            ),
            OptionLegRequest(
                symbol=long_symbol,
                ratio_qty=1.0,
                side=OrderSide.SELL,
                position_intent=PositionIntent.SELL_TO_CLOSE,
            ),
        ]

        if limit_debit is not None:
            alpaca_limit_price = round(abs(limit_debit), 2)
            order_req = LimitOrderRequest(
                qty=qty,
                order_class=OrderClass.MLEG,
                legs=legs,
                time_in_force=alpaca_tif,
                limit_price=alpaca_limit_price,
                client_order_id=client_order_id,
            )
        else:
            order_req = MarketOrderRequest(
                qty=qty,
                order_class=OrderClass.MLEG,
                legs=legs,
                time_in_force=alpaca_tif,
                client_order_id=client_order_id,
            )

        order_res = self.trading_client.submit_order(order_data=order_req)
        logger.info(
            "Alpaca MLEG Close Spread Submitted: Buy %s / Sell %s x %d (ID: %s, Debit: $%s)",
            short_symbol,
            long_symbol,
            qty,
            order_res.id,
            limit_debit,
        )

        return {
            "id": str(order_res.id),
            "client_order_id": str(order_res.client_order_id),
            "short_symbol": short_symbol,
            "long_symbol": long_symbol,
            "qty": int(order_res.qty) if order_res.qty else qty,
            "debit": limit_debit,
            "status": str(order_res.status),
            "order_class": "mleg",
        }

    def get_stock_bars(
        self,
        symbol: str = "AAPL",
        timeframe: Optional[Any] = None,
        limit: int = 250,
    ) -> pd.DataFrame:
        """
        Fetches historical daily/hourly bars for an equity symbol.
        Returns pd.DataFrame with ['open', 'high', 'low', 'close', 'volume'].
        """
        sym = symbol.upper()
        if timeframe is None and TimeFrame is not None:
            timeframe = TimeFrame(1, TimeFrameUnit.Day)

        if self.mock_mode:
            return self._generate_synthetic_stock_bars(symbol=sym, limit=limit)

        now = datetime.now(timezone.utc)
        start_time = now - timedelta(days=int(limit * 1.6) + 30)

        try:
            if StockBarsRequest is not None and self.stock_data_client is not None:
                req = StockBarsRequest(
                    symbol_or_symbols=sym,
                    timeframe=timeframe,
                    start=start_time,
                    end=now,
                )
                bars = self.stock_data_client.get_stock_bars(req)
                df = bars.df
                if df.empty:
                    logger.warning("Empty stock bars returned by Alpaca for %s. Using synthetic fallback.", sym)
                    return self._generate_synthetic_stock_bars(symbol=sym, limit=limit)

                if isinstance(df.index, pd.MultiIndex):
                    df = df.xs(sym, level=0)

                df.columns = [c.lower() for c in df.columns]
                return df.tail(limit)
            else:
                return self._generate_synthetic_stock_bars(symbol=sym, limit=limit)
        except Exception as e:
            logger.warning("Failed to fetch stock bars for %s: %s. Using synthetic fallback.", sym, e)
            return self._generate_synthetic_stock_bars(symbol=sym, limit=limit)

    def _generate_synthetic_stock_bars(
        self, symbol: str = "AAPL", limit: int = 250, base_price: Optional[float] = None
    ) -> pd.DataFrame:
        """Generates realistic synthetic daily bars for offline testing and dry-run."""
        base_p = base_price or self.get_stock_price(symbol)
        seed = abs(hash(symbol)) % 100000
        np.random.seed(seed)
        returns = np.random.normal(0.0003, 0.015, limit)
        prices = base_p * np.exp(np.cumsum(returns))
        prices = prices * (base_p / prices[-1])

        highs = prices * (1 + np.abs(np.random.normal(0.005, 0.003, limit)))
        lows = prices * (1 - np.abs(np.random.normal(0.005, 0.003, limit)))
        opens = np.roll(prices, 1)
        opens[0] = base_p
        volumes = np.random.uniform(1000000.0, 10000000.0, limit)

        dates = [
            datetime.now(timezone.utc) - timedelta(days=limit - i)
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

    def submit_stock_order(
        self,
        symbol: str,
        side: str,
        notional: Optional[float] = None,
        qty: Optional[float] = None,
        limit_price: Optional[float] = None,
        time_in_force: str = "DAY",
        order_type: str = "MARKET",
    ) -> Dict[str, Any]:
        """
        Submits an equity market or limit order with DAY or GTC time-in-force.
        Accepts notional ($ USD) or qty (shares).
        """
        client_order_id = f"stk_{uuid.uuid4().hex[:10]}"
        side_upper = side.upper()
        tif_upper = time_in_force.upper()
        sym = symbol.upper()

        if self.mock_mode:
            current_p = limit_price or self.get_stock_price(sym)
            if notional is not None:
                calc_qty = round(notional / current_p, 4) if current_p > 0 else 1.0
                calc_notional = notional
            else:
                calc_qty = qty or 1.0
                calc_notional = round(calc_qty * current_p, 2)

            if side_upper == "BUY":
                cur_pos = self._mock_stock_positions.get(sym)
                if cur_pos:
                    new_qty = cur_pos.qty + calc_qty
                    new_avg = ((cur_pos.qty * cur_pos.avg_entry_price) + calc_notional) / new_qty
                    self._mock_stock_positions[sym] = StockPositionInfo(
                        symbol=sym,
                        qty=new_qty,
                        avg_entry_price=round(new_avg, 2),
                        current_price=current_p,
                        market_value=round(new_qty * current_p, 2),
                    )
                else:
                    self._mock_stock_positions[sym] = StockPositionInfo(
                        symbol=sym,
                        qty=calc_qty,
                        avg_entry_price=current_p,
                        current_price=current_p,
                        market_value=round(calc_qty * current_p, 2),
                    )
            elif side_upper == "SELL":
                cur_pos = self._mock_stock_positions.get(sym)
                if cur_pos:
                    new_qty = cur_pos.qty - calc_qty
                    if new_qty <= 0.0001:
                        self._mock_stock_positions.pop(sym, None)
                    else:
                        self._mock_stock_positions[sym] = StockPositionInfo(
                            symbol=sym,
                            qty=new_qty,
                            avg_entry_price=cur_pos.avg_entry_price,
                            current_price=current_p,
                            market_value=round(new_qty * current_p, 2),
                        )

            receipt = {
                "id": f"mock_stk_{uuid.uuid4().hex[:8]}",
                "client_order_id": client_order_id,
                "symbol": sym,
                "side": side_upper,
                "order_type": order_type.upper(),
                "qty": calc_qty,
                "notional": calc_notional,
                "limit_price": limit_price,
                "status": "FILLED" if order_type.upper() == "MARKET" else "NEW",
                "filled_avg_price": current_p,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            self._mock_orders.append(receipt)
            return receipt

        # Live Alpaca API order placement
        alpaca_side = OrderSide.BUY if side_upper == "BUY" else OrderSide.SELL
        tif = TimeInForce.DAY if tif_upper == "DAY" else TimeInForce.GTC

        if order_type.upper() == "LIMIT":
            if limit_price is None:
                raise ValueError("Limit order requires limit_price.")
            order_req = LimitOrderRequest(
                symbol=sym,
                qty=qty,
                notional=notional,
                limit_price=round(limit_price, 2),
                side=alpaca_side,
                time_in_force=tif,
                client_order_id=client_order_id,
            )
        else:
            order_req = MarketOrderRequest(
                symbol=sym,
                qty=qty,
                notional=notional,
                side=alpaca_side,
                time_in_force=tif,
                client_order_id=client_order_id,
            )

        order_res = self.trading_client.submit_order(order_data=order_req)
        logger.info(
            "Alpaca Stock Order Submitted: %s %s %s (ID: %s, Status: %s)",
            side_upper,
            f"${notional:.2f}" if notional else f"{qty} shares",
            sym,
            order_res.id,
            getattr(order_res, "status", "NEW"),
        )

        return {
            "id": str(getattr(order_res, "id", "")),
            "client_order_id": str(getattr(order_res, "client_order_id", "")),
            "symbol": sym,
            "side": side_upper,
            "order_type": order_type.upper(),
            "qty": float(order_res.qty) if getattr(order_res, "qty", None) else qty,
            "notional": float(order_res.notional) if getattr(order_res, "notional", None) else notional,
            "limit_price": float(order_res.limit_price) if getattr(order_res, "limit_price", None) else limit_price,
            "status": str(getattr(order_res, "status", "NEW")),
            "filled_avg_price": float(order_res.filled_avg_price) if getattr(order_res, "filled_avg_price", None) else None,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
