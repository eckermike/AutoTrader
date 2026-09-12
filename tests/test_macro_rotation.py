"""
Unit tests for Macro Dual-Momentum / Sector Rotation Strategy (execution/macro_rotation_engine.py).
Tests universe momentum scoring, absolute trend filtering, safe-haven flight to SGOV,
buy execution, capital gating, HITL liquidity rebalancing alerts on cash shortage,
rebalance rotations with 30% tax escrow, -7% trailing stop-loss defense,
24-hour TTL stale order expiration, and atomic JSON state persistence.
"""

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pandas as pd
import pytest

from config import BotConfig
from execution.macro_rotation_engine import (
    ClosedMacroTrade,
    MacroAssetScore,
    MacroPosition,
    MacroRotationEngine,
    MacroRotationState,
)
from notifier import TradeNotifier
from options.options_client import AlpacaOptionsClient
from tax_engine import TaxEngine


@pytest.fixture
def mock_config(tmp_path):
    tax_file = tmp_path / "test_macro_tax.json"
    macro_file = tmp_path / "test_macro_state.json"
    return BotConfig(
        ALPACA_API_KEY="MOCK_KEY",
        ALPACA_SECRET_KEY="MOCK_SECRET",
        ALPACA_PAPER=True,
        TAX_RESERVE_FILE=tax_file,
        TAX_RATE=0.30,
        MACRO_ENABLED=True,
        MACRO_SYMBOLS=["QQQ", "SPY", "GLD", "VNQ", "SGOV"],
        MACRO_SAFE_HAVEN="SGOV",
        MACRO_MAX_CAPITAL_USD=5000.0,
        MACRO_TOP_N_ASSETS=2,
        MACRO_TRANCHE_SIZE_USD=2500.0,
        MACRO_LOOKBACK_SHORT_DAYS=60,
        MACRO_LOOKBACK_LONG_DAYS=120,
        MACRO_SMA_PERIOD=200,
        MACRO_TRAILING_STOP_PCT=0.07,
        MACRO_REBALANCE_CADENCE_DAYS=30,
        MACRO_STATE_FILE=str(macro_file),
    )


@pytest.fixture
def mock_options_client():
    client = AlpacaOptionsClient(api_key="MOCK", secret_key="MOCK", paper=True)
    client.mock_mode = True
    client._mock_cash = 100000.0
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


def test_macro_rotation_evaluate_universe_scores(mock_config, mock_options_client, tax_engine, notifier):
    """Verifies that the engine computes 60d, 120d, blended momentum, and 200-day SMA."""
    engine = MacroRotationEngine(
        config=mock_config,
        options_client=mock_options_client,
        tax_engine=tax_engine,
        notifier=notifier,
    )

    scores = engine.evaluate_universe()
    assert len(scores) == 5
    for sym in ["QQQ", "SPY", "GLD", "VNQ", "SGOV"]:
        assert sym in scores
        sc = scores[sym]
        assert sc.current_price > 0
        assert sc.sma_200 > 0
        assert isinstance(sc.blended_score, float)
        assert sc.asset_class != ""


def test_macro_rotation_safe_haven_flight_when_bear_market(mock_config, mock_options_client, tax_engine, notifier):
    """Verifies that when risky assets crash below SMA200 or trail cash, engine flees 100% to SGOV."""
    engine = MacroRotationEngine(
        config=mock_config,
        options_client=mock_options_client,
        tax_engine=tax_engine,
        notifier=notifier,
    )

    # Mock bars: SGOV has steady small positive return, all equities have deep selloffs below SMA200
    def mock_bear_bars(sym, limit=250):
        if sym == "SGOV":
            prices = [100.0 + i * 0.01 for i in range(limit)]
        else:
            # 200 bars at 500, then drop to 300 (below 200 SMA and negative momentum)
            prices = [500.0] * 200 + [300.0] * (limit - 200)
        return pd.DataFrame({"close": prices, "volume": [1000000.0] * limit})

    mock_options_client.get_stock_bars = MagicMock(side_effect=mock_bear_bars)

    scores = engine.evaluate_universe()
    assert engine.state.current_regime == "RISK_OFF"
    assert scores["SGOV"].is_selected is True
    assert scores["SGOV"].status == "SAFE_HAVEN"
    for risky in ["QQQ", "SPY", "GLD", "VNQ"]:
        assert scores[risky].is_selected is False


def test_macro_rotation_risk_on_top_leaders_selection(mock_config, mock_options_client, tax_engine, notifier):
    """Verifies that in a strong bull market, the top 2 momentum leaders are selected."""
    engine = MacroRotationEngine(
        config=mock_config,
        options_client=mock_options_client,
        tax_engine=tax_engine,
        notifier=notifier,
    )

    # QQQ (+30%) > GLD (+20%) > SPY (+10%) > VNQ (+5%) > SGOV (+2%), all above 200 SMA
    def mock_bull_bars(sym, limit=250):
        growth_rates = {
            "QQQ": 0.0012,
            "GLD": 0.0008,
            "SPY": 0.0005,
            "VNQ": 0.0002,
            "SGOV": 0.00005,
        }
        rate = growth_rates.get(sym, 0.0001)
        base = 100.0
        prices = [base * (1 + rate * i) for i in range(limit)]
        return pd.DataFrame({"close": prices, "volume": [1000000.0] * limit})

    mock_options_client.get_stock_bars = MagicMock(side_effect=mock_bull_bars)

    scores = engine.evaluate_universe()
    assert engine.state.current_regime == "RISK_ON"
    assert scores["QQQ"].is_selected is True
    assert scores["QQQ"].rank == 1
    assert scores["GLD"].is_selected is True
    assert scores["GLD"].rank == 2
    assert scores["SPY"].is_selected is False
    assert scores["VNQ"].is_selected is False
    assert scores["SGOV"].is_selected is False


def test_macro_rotation_rebalance_buy_execution(mock_config, mock_options_client, tax_engine, notifier):
    """Verifies that rebalancing enters the selected top 2 assets with $2,500 each and sets -7% trailing stop."""
    engine = MacroRotationEngine(
        config=mock_config,
        options_client=mock_options_client,
        tax_engine=tax_engine,
        notifier=notifier,
    )

    # Force universe to select QQQ and GLD
    def mock_bull_bars(sym, limit=250):
        rates = {"QQQ": 0.001, "GLD": 0.0008, "SPY": 0.0003, "VNQ": 0.0001, "SGOV": 0.00005}
        rate = rates.get(sym, 0.0001)
        prices = [100.0 * (1 + rate * i) for i in range(limit)]
        return pd.DataFrame({"close": prices, "volume": [1000000.0] * limit})

    mock_options_client.get_stock_bars = MagicMock(side_effect=mock_bull_bars)
    mock_options_client._mock_cash = 50000.0

    new_pos = engine.rebalance_portfolio(is_market_open=True)
    assert len(new_pos) == 2
    assert "QQQ" in engine.state.active_positions
    assert "GLD" in engine.state.active_positions

    qqq_pos = engine.state.active_positions["QQQ"]
    assert qqq_pos.cost_basis > 0
    assert qqq_pos.trailing_stop_price == round(qqq_pos.entry_price * 0.93, 2)  # -7%
    assert engine.state.current_regime == "RISK_ON"
    assert engine.state.last_rebalance_timestamp > 0


def test_macro_rotation_capital_ceiling_gating(mock_config, mock_options_client, tax_engine, notifier):
    """Verifies that the $5,000 maximum strategy ceiling blocks new purchases once reached."""
    engine = MacroRotationEngine(
        config=mock_config,
        options_client=mock_options_client,
        tax_engine=tax_engine,
        notifier=notifier,
    )

    # Pre-allocate $5,000 across two positions
    engine.state.active_positions["QQQ"] = MacroPosition(
        symbol="QQQ",
        qty=5.0,
        entry_price=500.0,
        cost_basis=2500.0,
        current_price=500.0,
        trailing_stop_price=465.0,
    )
    engine.state.active_positions["GLD"] = MacroPosition(
        symbol="GLD",
        qty=10.0,
        entry_price=250.0,
        cost_basis=2500.0,
        current_price=250.0,
        trailing_stop_price=232.5,
    )

    # Mock evaluate_universe so QQQ, GLD, and SPY are all targets
    mock_scores = {
        "QQQ": MacroAssetScore(symbol="QQQ", asset_class="Equities", current_price=500.0, return_60d=10.0, return_120d=20.0, blended_score=15.0, sma_200=450.0, is_above_sma200=True, is_selected=True),
        "GLD": MacroAssetScore(symbol="GLD", asset_class="Real Assets", current_price=250.0, return_60d=8.0, return_120d=16.0, blended_score=12.0, sma_200=220.0, is_above_sma200=True, is_selected=True),
        "SPY": MacroAssetScore(symbol="SPY", asset_class="Equities", current_price=560.0, return_60d=6.0, return_120d=12.0, blended_score=9.0, sma_200=500.0, is_above_sma200=True, is_selected=True),
    }
    engine.evaluate_universe = MagicMock(return_value=mock_scores)

    # Attempt rebalance
    new_pos = engine.rebalance_portfolio(is_market_open=True)
    # Since $5,000 is fully utilized and QQQ + GLD are still targets, SPY is blocked by ceiling
    assert len(new_pos) == 0
    assert "SPY" not in engine.state.active_positions


def test_macro_rotation_capital_gate_rejects_and_alerts_liquidity(mock_config, mock_options_client, tax_engine, notifier):
    """Verifies that insufficient tradable cash rejects order and dispatches HITL SGOV liquidation alert."""
    mock_liq = MagicMock()
    engine = MacroRotationEngine(
        config=mock_config,
        options_client=mock_options_client,
        tax_engine=tax_engine,
        notifier=notifier,
        liquidity_manager=mock_liq,
    )

    # Set cash to $1,000 (below $2,500 tranche size)
    mock_options_client._mock_cash = 1000.0

    def mock_bull_bars(sym, limit=250):
        rates = {"QQQ": 0.001, "GLD": 0.0008, "SPY": 0.0003, "VNQ": 0.0001, "SGOV": 0.00005}
        prices = [100.0 * (1 + rates.get(sym, 0.0001) * i) for i in range(limit)]
        return pd.DataFrame({"close": prices, "volume": [1000000.0] * limit})

    mock_options_client.get_stock_bars = MagicMock(side_effect=mock_bull_bars)

    new_pos = engine.rebalance_portfolio(is_market_open=True)
    assert len(new_pos) == 0
    assert len(engine.state.active_positions) == 0

    # Verify HITL liquidation alert was dispatched
    mock_liq.request_liquidation_for_opportunity.assert_called()
    call_args = mock_liq.request_liquidation_for_opportunity.call_args[1]
    assert call_args["needed_cash"] == 2500.0
    assert "Macro Dual-Momentum" in call_args["opportunity_type"]


def test_macro_rotation_rebalance_sell_and_tax_escrow(mock_config, mock_options_client, tax_engine, notifier):
    """Verifies that demoted positions are sold at rebalance and 30% of profit is escrowed."""
    engine = MacroRotationEngine(
        config=mock_config,
        options_client=mock_options_client,
        tax_engine=tax_engine,
        notifier=notifier,
    )

    # Pre-populate held position in VNQ entered at $80
    engine.state.active_positions["VNQ"] = MacroPosition(
        symbol="VNQ",
        qty=25.0,
        entry_price=80.0,
        cost_basis=2000.0,
        current_price=80.0,
        trailing_stop_price=74.4,
    )

    # Now VNQ price is $100 (realized gain = $500)
    mock_options_client.get_stock_price = MagicMock(return_value=100.0)

    # Mock new universe where QQQ and GLD win, so VNQ is demoted
    def mock_rotation_bars(sym, limit=250):
        rates = {"QQQ": 0.002, "GLD": 0.0015, "SPY": 0.0005, "VNQ": 0.0001, "SGOV": 0.00005}
        prices = [100.0 * (1 + rates.get(sym, 0.0001) * i) for i in range(limit)]
        return pd.DataFrame({"close": prices, "volume": [1000000.0] * limit})

    mock_options_client.get_stock_bars = MagicMock(side_effect=mock_rotation_bars)
    mock_options_client._mock_cash = 50000.0

    initial_tax = tax_engine.current_reserve
    engine.rebalance_portfolio(is_market_open=True)

    # Assert VNQ was liquidated
    assert "VNQ" not in engine.state.active_positions
    assert len(engine.state.closed_trades) >= 1
    vnq_trade = [t for t in engine.state.closed_trades if t.symbol == "VNQ"][0]
    expected_pnl = (100.0 - 80.0) * 25.0  # +$500.00
    assert vnq_trade.realized_pnl == expected_pnl
    assert vnq_trade.tax_escrow == round(expected_pnl * 0.30, 2)  # $150.00
    assert tax_engine.current_reserve == initial_tax + 150.00


def test_macro_rotation_trailing_stop_loss_defense(mock_config, mock_options_client, tax_engine, notifier):
    """Verifies that an active position is closed immediately when price breaches the -7% trailing stop."""
    engine = MacroRotationEngine(
        config=mock_config,
        options_client=mock_options_client,
        tax_engine=tax_engine,
        notifier=notifier,
    )

    # Active QQQ entered at $500, peaked at $500, trailing stop at $465 (-7%)
    engine.state.active_positions["QQQ"] = MacroPosition(
        symbol="QQQ",
        qty=5.0,
        entry_price=500.0,
        cost_basis=2500.0,
        current_price=500.0,
        highest_price=500.0,
        trailing_stop_price=465.0,
    )

    # Price drops to $460 (breaching stop)
    mock_options_client.get_stock_price = MagicMock(return_value=460.0)

    closed = engine.manage_active_positions(is_market_open=True)
    assert len(closed) == 1
    trade = closed[0]
    assert trade.symbol == "QQQ"
    assert trade.exit_reason == "TRAILING_STOP"
    assert trade.realized_pnl == (460.0 - 500.0) * 5.0  # -$200.00
    assert "QQQ" not in engine.state.active_positions


def test_macro_rotation_stale_order_cancellation(mock_config, mock_options_client, tax_engine, notifier):
    """Verifies that orders older than 24h are cancelled by the TTL watchdog."""
    engine = MacroRotationEngine(
        config=mock_config,
        options_client=mock_options_client,
        tax_engine=tax_engine,
        notifier=notifier,
    )

    mock_options_client.cancel_order = MagicMock(return_value=True)
    old_time = datetime.now(timezone.utc).timestamp() - (25 * 3600)
    fresh_time = datetime.now(timezone.utc).timestamp() - (2 * 3600)

    engine.state.pending_orders["order_stale_macro"] = {"symbol": "QQQ", "created_at": old_time}
    engine.state.pending_orders["order_fresh_macro"] = {"symbol": "GLD", "created_at": fresh_time}

    cancelled = engine.check_stale_orders()
    assert "order_stale_macro" in cancelled
    assert "order_fresh_macro" not in cancelled
    assert "order_stale_macro" not in engine.state.pending_orders
    assert "order_fresh_macro" in engine.state.pending_orders


def test_macro_rotation_state_persistence_and_reload(mock_config, mock_options_client, tax_engine, notifier):
    """Verifies that state persists to macro_rotation_state.json and reloads cleanly."""
    engine1 = MacroRotationEngine(
        config=mock_config,
        options_client=mock_options_client,
        tax_engine=tax_engine,
        notifier=notifier,
    )

    engine1.state.current_regime = "RISK_ON"
    engine1.state.active_positions["QQQ"] = MacroPosition(
        symbol="QQQ",
        qty=5.0,
        entry_price=480.0,
        cost_basis=2400.0,
        current_price=490.0,
        highest_price=490.0,
        trailing_stop_price=455.7,
        asset_class="Tech / Growth Equities",
    )
    engine1.state.closed_trades.append(
        ClosedMacroTrade(
            symbol="GLD",
            qty=10.0,
            entry_price=220.0,
            exit_price=235.0,
            entry_time="2026-08-01T10:00:00Z",
            exit_time="2026-09-01T10:00:00Z",
            holding_days=31.0,
            realized_pnl=150.0,
            realized_pnl_pct=0.068,
            tax_escrow=45.0,
            exit_reason="REBALANCE_ROTATION",
        )
    )
    engine1._save_state()

    # Load in new engine instance
    engine2 = MacroRotationEngine(
        config=mock_config,
        options_client=mock_options_client,
        tax_engine=tax_engine,
        notifier=notifier,
    )

    assert engine2.state.current_regime == "RISK_ON"
    assert "QQQ" in engine2.state.active_positions
    assert engine2.state.active_positions["QQQ"].qty == 5.0
    assert len(engine2.state.closed_trades) == 1
    assert engine2.state.closed_trades[0].symbol == "GLD"
    assert engine2.state.closed_trades[0].realized_pnl == 150.0
