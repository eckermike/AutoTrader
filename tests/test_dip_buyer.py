"""
Unit tests for Equity Mean-Reversion Dip Buyer Strategy (execution/dip_buyer_engine.py).
Tests universe scanning, oversold RSI detection, buy execution, capital gating,
HITL SGOV/FBND liquidation alert on cash shortage, +5% / RSI >= 50 profit target exit,
-5% stop-loss defense, 24-hour TTL stale order expiration, 30% tax escrow, and JSON persistence.
"""

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pandas as pd
import pytest

from config import BotConfig
from execution.dip_buyer_engine import (
    ClosedDipTrade,
    DipBuyerEngine,
    DipCandidate,
    DipPosition,
)
from notifier import TradeNotifier
from options.options_client import AlpacaOptionsClient
from tax_engine import TaxEngine


@pytest.fixture
def mock_config(tmp_path):
    tax_file = tmp_path / "test_dip_tax.json"
    dip_file = tmp_path / "test_dip_state.json"
    return BotConfig(
        ALPACA_API_KEY="MOCK_KEY",
        ALPACA_SECRET_KEY="MOCK_SECRET",
        ALPACA_PAPER=True,
        TAX_RESERVE_FILE=tax_file,
        TAX_RATE=0.30,
        DIP_ENABLED=True,
        DIP_SYMBOLS=["AAPL", "MSFT", "NVDA"],
        DIP_ORDER_SIZE_USD=1000.0,
        DIP_MAX_CAPITAL_USD=3000.0,
        DIP_RSI_THRESHOLD=30.0,
        DIP_RSI_EXIT_THRESHOLD=50.0,
        DIP_PROFIT_TARGET_PCT=0.05,
        DIP_STOP_LOSS_PCT=0.05,
        DIP_TIME_STOP_DAYS=15,
        DIP_STATE_FILE=str(dip_file),
    )


@pytest.fixture
def mock_options_client():
    client = AlpacaOptionsClient(api_key="MOCK", secret_key="MOCK", paper=True)
    client.mock_mode = True
    return client


@pytest.fixture
def tax_engine(mock_config):
    return TaxEngine(filepath=mock_config.TAX_RESERVE_FILE, tax_rate=mock_config.TAX_RATE)


@pytest.fixture
def notifier():
    return TradeNotifier(
        recipient="test@example.com",
        ntfy_topic="test_topic",
        enabled=True,
        macos_banner=False,
    )


def _generate_oversold_bars(symbol: str = "AAPL", base_price: float = 200.0) -> pd.DataFrame:
    """Generates synthetic daily bars ending with a deep selloff (RSI < 30)."""
    # 200 bars of steady uptrend, then 15 consecutive down bars
    prices = [base_price + i * 0.5 for i in range(200)]
    for i in range(1, 25):
        prices.append(prices[-1] * 0.97)  # sharp continuous decline

    df = pd.DataFrame(
        {
            "open": prices,
            "high": [p * 1.01 for p in prices],
            "low": [p * 0.99 for p in prices],
            "close": prices,
            "volume": [5000000.0] * len(prices),
        }
    )
    return df


def test_dip_buyer_scan_universe(mock_config, mock_options_client, tax_engine, notifier):
    """Verifies that the engine scans target symbols and computes RSI and SMAs."""
    engine = DipBuyerEngine(
        config=mock_config,
        options_client=mock_options_client,
        tax_engine=tax_engine,
        notifier=notifier,
    )

    candidates = engine.scan_universe()
    assert len(candidates) == 3
    for sym in ["AAPL", "MSFT", "NVDA"]:
        assert sym in candidates
        cand = candidates[sym]
        assert cand.current_price > 0
        assert 0.0 <= cand.rsi <= 100.0
        assert cand.sma_50 > 0
        assert cand.sma_200 > 0


def test_dip_buyer_buy_execution(mock_config, mock_options_client, tax_engine, notifier):
    """Verifies buy order execution and DipPosition creation when RSI < 30 and cash is available."""
    engine = DipBuyerEngine(
        config=mock_config,
        options_client=mock_options_client,
        tax_engine=tax_engine,
        notifier=notifier,
    )

    # Mock an oversold scan result
    engine.state.last_scan["AAPL"] = {
        "symbol": "AAPL",
        "current_price": 200.0,
        "rsi": 25.0,  # Oversold!
        "sma_50": 210.0,
        "sma_200": 190.0,
        "is_oversold": True,
        "is_trend_intact": True,
        "signal": "BUY",
    }

    # Set available cash
    mock_options_client._mock_cash = 10000.0

    new_positions = engine.evaluate_dip_entries(is_market_open=True)
    assert len(new_positions) == 1
    pos = new_positions[0]
    assert pos.symbol == "AAPL"
    assert pos.entry_price > 0
    assert pos.profit_target_price == round(pos.entry_price * 1.05, 2)
    assert pos.stop_loss_price == round(pos.entry_price * 0.95, 2)
    assert "AAPL" in engine.state.active_positions


def test_dip_buyer_capital_ceiling_gating(mock_config, mock_options_client, tax_engine, notifier):
    """Verifies that the strategy ceiling ($3,000 max) blocks entries once reached."""
    engine = DipBuyerEngine(
        config=mock_config,
        options_client=mock_options_client,
        tax_engine=tax_engine,
        notifier=notifier,
    )

    mock_options_client._mock_cash = 50000.0

    # Pre-populate active positions totaling $3,000
    engine.state.active_positions["MSFT"] = DipPosition(
        symbol="MSFT",
        qty=5.0,
        entry_price=400.0,
        cost_basis=2000.0,
        current_price=400.0,
        highest_price=400.0,
        profit_target_price=420.0,
        stop_loss_price=380.0,
    )
    engine.state.active_positions["NVDA"] = DipPosition(
        symbol="NVDA",
        qty=10.0,
        entry_price=100.0,
        cost_basis=1000.0,
        current_price=100.0,
        highest_price=100.0,
        profit_target_price=105.0,
        stop_loss_price=95.0,
    )

    # Try to enter AAPL with $1,000 (which would exceed $3,000 max)
    engine.state.last_scan["AAPL"] = {
        "symbol": "AAPL",
        "current_price": 200.0,
        "rsi": 22.0,
        "is_oversold": True,
        "is_trend_intact": True,
    }

    new_pos = engine.evaluate_dip_entries(is_market_open=True)
    assert len(new_pos) == 0
    assert "AAPL" not in engine.state.active_positions


def test_dip_buyer_capital_gate_rejects_and_alerts_liquidity(mock_config, mock_options_client, tax_engine, notifier):
    """Verifies that low tradable cash rejects order and triggers mobile liquidation request."""
    mock_liq = MagicMock()
    engine = DipBuyerEngine(
        config=mock_config,
        options_client=mock_options_client,
        tax_engine=tax_engine,
        notifier=notifier,
        liquidity_manager=mock_liq,
    )

    # Set cash below order size of $1,000
    mock_options_client._mock_cash = 400.0

    engine.state.last_scan["AAPL"] = {
        "symbol": "AAPL",
        "current_price": 200.0,
        "rsi": 25.0,
        "is_oversold": True,
        "is_trend_intact": True,
    }

    new_pos = engine.evaluate_dip_entries(is_market_open=True)
    assert len(new_pos) == 0
    assert "AAPL" not in engine.state.active_positions

    # Assert mobile approval request was dispatched
    mock_liq.request_liquidation_for_opportunity.assert_called_once()
    args, kwargs = mock_liq.request_liquidation_for_opportunity.call_args
    assert kwargs["target_symbol"] == "AAPL"
    assert kwargs["needed_cash"] == 1000.0


def test_dip_buyer_profit_target_exit_and_tax_escrow(mock_config, mock_options_client, tax_engine, notifier):
    """Verifies +5% profit target early exit triggers SELL and deposits 30% tax escrow."""
    engine = DipBuyerEngine(
        config=mock_config,
        options_client=mock_options_client,
        tax_engine=tax_engine,
        notifier=notifier,
    )

    # Active position entered at $200.00 (profit target = $210.00)
    engine.state.active_positions["AAPL"] = DipPosition(
        symbol="AAPL",
        qty=5.0,
        entry_price=200.0,
        cost_basis=1000.0,
        current_price=200.0,
        current_rsi=28.0,
        highest_price=200.0,
        profit_target_price=210.0,
        stop_loss_price=190.0,
    )

    # Stock price surges to $212.00 (+6% gain)
    mock_options_client.get_stock_price = MagicMock(return_value=212.00)

    initial_tax = tax_engine.current_reserve
    closed = engine.manage_active_positions(is_market_open=True)

    assert len(closed) == 1
    trade = closed[0]
    assert trade.symbol == "AAPL"
    assert trade.exit_price == 212.00
    assert trade.exit_reason == "PROFIT_TARGET"
    expected_pnl = (212.00 - 200.00) * 5.0  # +$60.00
    assert trade.realized_pnl == expected_pnl
    assert trade.tax_escrow == round(expected_pnl * 0.30, 2)  # $18.00

    # Verify tax engine reserve increased
    assert tax_engine.current_reserve == initial_tax + trade.tax_escrow
    assert "AAPL" not in engine.state.active_positions


def test_dip_buyer_rsi_rebound_exit(mock_config, mock_options_client, tax_engine, notifier):
    """Verifies that RSI rebounding to >= 50.0 exits in profit even if +5% target not yet reached."""
    engine = DipBuyerEngine(
        config=mock_config,
        options_client=mock_options_client,
        tax_engine=tax_engine,
        notifier=notifier,
    )

    engine.state.active_positions["MSFT"] = DipPosition(
        symbol="MSFT",
        qty=2.5,
        entry_price=400.0,
        cost_basis=1000.0,
        current_price=400.0,
        current_rsi=26.0,
        highest_price=400.0,
        profit_target_price=420.0,  # +5%
        stop_loss_price=380.0,
    )

    # Stock at $410.00 (+2.5% gain, below +5%) but RSI rebounded to 53.0
    mock_options_client.get_stock_price = MagicMock(return_value=410.00)
    engine.state.last_scan["MSFT"] = {"rsi": 53.0}

    closed = engine.manage_active_positions(is_market_open=True)
    assert len(closed) == 1
    assert closed[0].exit_reason == "RSI_REBOUND"
    assert closed[0].realized_pnl == (410.0 - 400.0) * 2.5  # +$25.00


def test_dip_buyer_stop_loss_exit(mock_config, mock_options_client, tax_engine, notifier):
    """Verifies -5% stop loss defense exits immediately to prevent catastrophic bleed."""
    engine = DipBuyerEngine(
        config=mock_config,
        options_client=mock_options_client,
        tax_engine=tax_engine,
        notifier=notifier,
    )

    engine.state.active_positions["NVDA"] = DipPosition(
        symbol="NVDA",
        qty=10.0,
        entry_price=100.0,
        cost_basis=1000.0,
        current_price=100.0,
        current_rsi=28.0,
        highest_price=100.0,
        profit_target_price=105.0,
        stop_loss_price=95.0,  # -5%
    )

    # Stock price drops to $94.00 (-6% loss)
    mock_options_client.get_stock_price = MagicMock(return_value=94.00)
    engine.state.last_scan["NVDA"] = {"rsi": 22.0}

    closed = engine.manage_active_positions(is_market_open=True)
    assert len(closed) == 1
    assert closed[0].exit_reason == "STOP_LOSS"
    assert closed[0].realized_pnl == (94.0 - 100.0) * 10.0  # -$60.00
    assert "NVDA" not in engine.state.active_positions


def test_dip_buyer_stale_order_cancellation(mock_config, mock_options_client, tax_engine, notifier):
    """Verifies 24-hour TTL watchdog cancels resting orders older than 24 hours."""
    engine = DipBuyerEngine(
        config=mock_config,
        options_client=mock_options_client,
        tax_engine=tax_engine,
        notifier=notifier,
    )

    mock_options_client.cancel_order = MagicMock(return_value=True)

    # Add a pending order that was created 25 hours ago
    old_time = datetime.now(timezone.utc).timestamp() - (25 * 3600)
    engine.state.pending_orders["order_stale_123"] = {
        "symbol": "AAPL",
        "created_at": old_time,
    }
    # And a fresh order created 2 hours ago
    fresh_time = datetime.now(timezone.utc).timestamp() - (2 * 3600)
    engine.state.pending_orders["order_fresh_456"] = {
        "symbol": "MSFT",
        "created_at": fresh_time,
    }

    cancelled = engine.check_stale_orders()
    assert "order_stale_123" in cancelled
    assert "order_fresh_456" not in cancelled
    assert "order_stale_123" not in engine.state.pending_orders
    assert "order_fresh_456" in engine.state.pending_orders
    mock_options_client.cancel_order.assert_called_once_with("order_stale_123")


def test_dip_buyer_state_persistence_and_reload(mock_config, mock_options_client, tax_engine, notifier):
    """Verifies that active positions, scanner cache, and closed trades persist cleanly to JSON."""
    engine1 = DipBuyerEngine(
        config=mock_config,
        options_client=mock_options_client,
        tax_engine=tax_engine,
        notifier=notifier,
    )

    pos = DipPosition(
        symbol="AAPL",
        qty=4.5,
        entry_price=220.0,
        cost_basis=990.0,
        current_price=220.0,
        highest_price=220.0,
        profit_target_price=231.0,
        stop_loss_price=209.0,
    )
    engine1.state.active_positions["AAPL"] = pos

    trade = ClosedDipTrade(
        symbol="MSFT",
        qty=2.5,
        entry_price=400.0,
        exit_price=420.0,
        entry_time="2026-09-10T10:00:00Z",
        exit_time="2026-09-12T15:00:00Z",
        holding_days=2.2,
        realized_pnl=50.0,
        realized_pnl_pct=0.05,
        tax_escrow=15.0,
        exit_reason="PROFIT_TARGET",
    )
    engine1.state.closed_trades.append(trade)
    engine1._save_state()

    # Load in new engine instance
    engine2 = DipBuyerEngine(
        config=mock_config,
        options_client=mock_options_client,
        tax_engine=tax_engine,
        notifier=notifier,
    )

    assert "AAPL" in engine2.state.active_positions
    loaded_pos = engine2.state.active_positions["AAPL"]
    assert loaded_pos.qty == 4.5
    assert loaded_pos.entry_price == 220.0
    assert len(engine2.state.closed_trades) == 1
    assert engine2.state.closed_trades[0].symbol == "MSFT"
    assert engine2.state.closed_trades[0].realized_pnl == 50.0
