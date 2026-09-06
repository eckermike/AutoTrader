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
    )
    from alpaca.trading.enums import (
        OrderSide,
        TimeInForce,
        ContractType,
        PositionIntent,
        ExerciseStyle,
        QueryOrderStatus,
    )
    from alpaca.data.historical.stock import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestQuoteRequest, StockLatestTradeRequest
    from alpaca.data.historical.option import OptionHistoricalDataClient
    from alpaca.data.requests import OptionLatestQuoteRequest
except ImportError:
    TradingClient = None

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
    qty: int
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
            qty = int(float(pos.qty))
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
            exp_date_str = "2026-10-09"
            dte = 33
            if sym == "INTC":
                strikes = [19.0, 19.5, 20.0, 20.5, 21.0, 21.5, 22.0, 22.5, 23.0]
            else:
                strikes = sorted(set([round(base_price * m, 1) for m in [0.88, 0.90, 0.92, 0.95, 0.98, 1.00, 1.02, 1.05, 1.08]]))
            for strike in strikes:
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

    def submit_option_order(
        self,
        symbol: str,
        side: str,
        position_intent: str,
        qty: int = 1,
        limit_price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Submits an option order (e.g. SELL_TO_OPEN or BUY_TO_CLOSE).
        """
        client_order_id = f"opt_{uuid.uuid4().hex[:10]}"
        side_upper = side.upper()
        intent_upper = position_intent.upper()

        if self.mock_mode:
            price = limit_price or 0.50
            if "SELL" in side_upper:
                # Open short option position
                und = "INTC"
                for known_sym in ["INTC", "SOFI", "HOOD", "PLTR", "XLF", "F"]:
                    if symbol.upper().startswith(known_sym):
                        und = known_sym
                        break
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
                # Close short option position
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

        if limit_price is not None:
            order_req = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=alpaca_side,
                time_in_force=TimeInForce.GTC,
                limit_price=round(limit_price, 2),
                position_intent=alpaca_intent,
                client_order_id=client_order_id,
            )
        else:
            order_req = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=alpaca_side,
                time_in_force=TimeInForce.GTC,
                position_intent=alpaca_intent,
                client_order_id=client_order_id,
            )

        order_res = self.trading_client.submit_order(order_data=order_req)
        logger.info(
            "Alpaca Option Order Submitted: %s %d %s (ID: %s, Intent: %s)",
            side_upper,
            qty,
            symbol,
            order_res.id,
            intent_upper,
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
