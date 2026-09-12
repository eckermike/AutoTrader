"""
Unit tests for Statistical Pairs Trading Engine (execution/pairs_trading_engine.py).
Tests spread ratio calculation, rolling z-score divergence, market-neutral simultaneous
order execution, capital gating, HITL liquidity rebalancing requests on cash shortage,
mean reversion profit-taking with 30% tax escrow withholding, downside divergence
stop-loss, holding duration time stop, 24-hour TTL stale order expiration, and atomic JSON state persistence.
"""

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from config import BotConfig
from execution.pairs_trading_engine import (
    ActivePairPosition,
    ClosedPairTrade,
    PairLeg,
    PairSpreadMetrics,
    PairsTradingEngine,
    PairsTradingState,
)
from notifier import TradeNotifier
from options.options_client import AlpacaOptionsClient
from tax_engine import TaxEngine


@pytest.fixture
def mock_config(tmp_path):
    tax_file = tmp_path / "test_pairs_tax.json"
    pairs_file = tmp_path / "test_pairs_state.json"
    return BotConfig(
        ALPACA_API_KEY="MOCK_KEY",
        ALPACA_SECRET_KEY="MOCK_SECRET",
        ALPACA_PAPER=True,
        TAX_RESERVE_FILE=tax_file,
        TAX_RATE=0.30,
        PAIRS_ENABLED=True,
        PAIRS_LIST=[("XOM", "CVX"), ("KO", "PEP"), ("GOOGL", "MSFT")],
        PAIRS_MAX_CAPITAL_USD=5000.0,
        PAIRS_ALLOCATION_PER_PAIR_USD=2500.0,
        PAIRS_LOOKBACK_DAYS=30,
        PAIRS_ENTRY_ZSCORE=2.0,
        PAIRS_EXIT_ZSCORE=0.5,
        PAIRS_STOP_LOSS_ZSCORE=3.5,
        PAIRS_TIME_STOP_DAYS=20,
        PAIRS_STATE_FILE=str(pairs_file),
        PAIRS_TIME_IN_FORCE="DAY",
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
    notifier_mock = TradeNotifier(
        recipient="test@example.com",
        ntfy_topic="test_topic",
        enabled=True,
        macos_banner=False,
    )
    notifier_mock.notify_pair_trade_open = MagicMock()
    notifier_mock.notify_pair_trade_close = MagicMock()
    return notifier_mock


def test_pairs_trading_spread_calculation(mock_config, mock_options_client, tax_engine, notifier):
    """Verifies calculation of price ratio, rolling mean, std dev, and z-score for a pair."""
    engine = PairsTradingEngine(
        config=mock_config,
        options_client=mock_options_client,
        tax_engine=tax_engine,
        notifier=notifier,
    )

    metrics = engine.calculate_pair_spread("XOM", "CVX")
    assert metrics is not None
    assert metrics.symbol_a == "XOM"
    assert metrics.symbol_b == "CVX"
    assert metrics.price_a == 115.00
    assert metrics.price_b == 150.00
    assert metrics.ratio > 0
    assert metrics.mean_ratio > 0
    assert metrics.std_ratio > 0
    assert isinstance(metrics.zscore, float)
    assert metrics.signal in ["BUY_A_SELL_B", "SELL_A_BUY_B", "NEUTRAL"]


def test_pairs_trading_entry_signal_logic(mock_config, mock_options_client, tax_engine, notifier):
    """Verifies that z >= 2.0 signals SELL_A_BUY_B and z <= -2.0 signals BUY_A_SELL_B."""
    engine = PairsTradingEngine(
        config=mock_config,
        options_client=mock_options_client,
        tax_engine=tax_engine,
        notifier=notifier,
    )

    # Mock historical bars to force specific mean and std
    dates = pd.date_range(end=datetime.now(), periods=45, freq="D")
    bars_a = pd.DataFrame({"close": [100.0, 102.0, 98.0, 101.0, 99.0] * 9}, index=dates)
    bars_b = pd.DataFrame({"close": [100.0] * 45}, index=dates)
    mock_options_client.get_stock_bars = MagicMock(
        side_effect=lambda sym, **kwargs: bars_a if sym == "XOM" else bars_b
    )
    mock_options_client.get_stock_historical_bars = mock_options_client.get_stock_bars

    # 1. Neutral test (Price ratio matches historical mean)
    mock_options_client.get_stock_price = MagicMock(side_effect=lambda sym: 100.0)
    metrics_neutral = engine.calculate_pair_spread("XOM", "CVX")
    assert metrics_neutral.signal == "NEUTRAL"

    # 2. Overvalued A test: Ratio rises high (Price A = 110, Price B = 100 -> ratio 1.10 vs mean 1.0)
    mock_options_client.get_stock_price = MagicMock(side_effect=lambda sym: 110.0 if sym == "XOM" else 100.0)
    metrics_high = engine.calculate_pair_spread("XOM", "CVX")
    assert metrics_high.zscore >= 2.0
    assert metrics_high.signal == "SELL_A_BUY_B"  # Short A, Long B

    # 3. Undervalued A test: Ratio drops low (Price A = 90, Price B = 100 -> ratio 0.90 vs mean 1.0)
    mock_options_client.get_stock_price = MagicMock(side_effect=lambda sym: 90.0 if sym == "XOM" else 100.0)
    metrics_low = engine.calculate_pair_spread("XOM", "CVX")
    assert metrics_low.zscore <= -2.0
    assert metrics_low.signal == "BUY_A_SELL_B"  # Long A, Short B


def test_pairs_trading_order_execution(mock_config, mock_options_client, tax_engine, notifier):
    """Verifies simultaneous market-neutral order execution for Long and Short legs ($1,250 each)."""
    engine = PairsTradingEngine(
        config=mock_config,
        options_client=mock_options_client,
        tax_engine=tax_engine,
        notifier=notifier,
    )

    metrics_map = {
        "XOM_CVX": PairSpreadMetrics(
            symbol_a="XOM",
            symbol_b="CVX",
            price_a=115.00,
            price_b=150.00,
            ratio=0.7667,
            mean_ratio=0.7000,
            std_ratio=0.0250,
            zscore=2.67,
            signal="SELL_A_BUY_B",
            status="OPPORTUNITY",
        )
    }

    engine.evaluate_entries(metrics_map)

    assert "XOM_CVX" in engine.state.active_positions
    pos = engine.state.active_positions["XOM_CVX"]
    assert pos.symbol_a == "XOM"
    assert pos.symbol_b == "CVX"
    assert pos.leg_a.side == "SELL"  # Short XOM
    assert pos.leg_b.side == "BUY"   # Long CVX
    assert round(pos.leg_a.cost_basis, 0) == 1250.0
    assert round(pos.leg_b.cost_basis, 0) == 1250.0
    assert round(pos.total_notional, 0) == 2500.0

    notifier.notify_pair_trade_open.assert_called_once()


def test_pairs_trading_capital_gating(mock_config, mock_options_client, tax_engine, notifier):
    """Verifies that trade entry is blocked when tradable cash is less than $2,500."""
    engine = PairsTradingEngine(
        config=mock_config,
        options_client=mock_options_client,
        tax_engine=tax_engine,
        notifier=notifier,
    )

    # Drain tradable cash
    tax_engine.get_tradable_cash = MagicMock(return_value=1200.0)

    metrics_map = {
        "XOM_CVX": PairSpreadMetrics(
            symbol_a="XOM",
            symbol_b="CVX",
            price_a=115.00,
            price_b=150.00,
            ratio=0.7667,
            mean_ratio=0.7000,
            std_ratio=0.0250,
            zscore=2.67,
            signal="SELL_A_BUY_B",
            status="OPPORTUNITY",
        )
    }

    engine.evaluate_entries(metrics_map)
    assert len(engine.state.active_positions) == 0
    notifier.notify_pair_trade_open.assert_not_called()


def test_pairs_trading_hitl_liquidity_alert(mock_config, mock_options_client, tax_engine, notifier):
    """Verifies that an interactive HITL liquidity liquidation alert is sent when cash is low."""
    mock_liquidity_manager = MagicMock()
    engine = PairsTradingEngine(
        config=mock_config,
        options_client=mock_options_client,
        tax_engine=tax_engine,
        notifier=notifier,
        liquidity_manager=mock_liquidity_manager,
    )

    tax_engine.get_tradable_cash = MagicMock(return_value=800.0)

    metrics_map = {
        "KO_PEP": PairSpreadMetrics(
            symbol_a="KO",
            symbol_b="PEP",
            price_a=68.00,
            price_b=172.00,
            ratio=0.3953,
            mean_ratio=0.4500,
            std_ratio=0.0200,
            zscore=-2.73,
            signal="BUY_A_SELL_B",
            status="OPPORTUNITY",
        )
    }

    engine.evaluate_entries(metrics_map)
    assert len(engine.state.active_positions) == 0
    mock_liquidity_manager.request_liquidation_for_opportunity.assert_called_once()


def test_pairs_trading_strategy_ceiling(mock_config, mock_options_client, tax_engine, notifier):
    """Verifies that new pair entries are blocked when the $5,000 cumulative ceiling is reached."""
    engine = PairsTradingEngine(
        config=mock_config,
        options_client=mock_options_client,
        tax_engine=tax_engine,
        notifier=notifier,
    )

    # Pre-populate 2 active positions totaling $5,000
    pos1 = ActivePairPosition(
        pair_id="XOM_CVX",
        symbol_a="XOM",
        symbol_b="CVX",
        leg_a=PairLeg(symbol="XOM", side="SELL", qty=10.87, entry_price=115.0, cost_basis=1250.0),
        leg_b=PairLeg(symbol="CVX", side="BUY", qty=8.33, entry_price=150.0, cost_basis=1250.0),
        entry_ratio=0.7667,
        entry_zscore=2.5,
        total_notional=2500.0,
    )
    pos2 = ActivePairPosition(
        pair_id="KO_PEP",
        symbol_a="KO",
        symbol_b="PEP",
        leg_a=PairLeg(symbol="KO", side="BUY", qty=18.38, entry_price=68.0, cost_basis=1250.0),
        leg_b=PairLeg(symbol="PEP", side="SELL", qty=7.27, entry_price=172.0, cost_basis=1250.0),
        entry_ratio=0.3953,
        entry_zscore=-2.5,
        total_notional=2500.0,
    )
    engine.state.active_positions["XOM_CVX"] = pos1
    engine.state.active_positions["KO_PEP"] = pos2

    # Attempt 3rd entry
    metrics_map = {
        "GOOGL_MSFT": PairSpreadMetrics(
            symbol_a="GOOGL",
            symbol_b="MSFT",
            price_a=165.00,
            price_b=420.00,
            ratio=0.3929,
            mean_ratio=0.4500,
            std_ratio=0.0200,
            zscore=-2.85,
            signal="BUY_A_SELL_B",
            status="OPPORTUNITY",
        )
    }

    engine.evaluate_entries(metrics_map)
    assert "GOOGL_MSFT" not in engine.state.active_positions
    assert len(engine.state.active_positions) == 2


def test_pairs_trading_mean_reversion_exit_and_tax_escrow(mock_config, mock_options_client, tax_engine, notifier):
    """Verifies that when z-score converges to <= 0.50, trade liquidates for profit and allocates 30% tax escrow."""
    engine = PairsTradingEngine(
        config=mock_config,
        options_client=mock_options_client,
        tax_engine=tax_engine,
        notifier=notifier,
    )

    # Active trade: Short XOM @ 120, Long CVX @ 140. Total notional $2,500. Entry z = 2.5
    pos = ActivePairPosition(
        pair_id="XOM_CVX",
        symbol_a="XOM",
        symbol_b="CVX",
        leg_a=PairLeg(symbol="XOM", side="SELL", qty=10.42, entry_price=120.0, cost_basis=1250.4),
        leg_b=PairLeg(symbol="CVX", side="BUY", qty=8.93, entry_price=140.0, cost_basis=1250.2),
        entry_ratio=0.8571,
        entry_zscore=2.5,
        total_notional=2500.6,
    )
    engine.state.active_positions["XOM_CVX"] = pos

    # Mean reversion: XOM falls to 110 (Short gain: (120-110)*10.42 = +$104.20)
    # CVX rises to 145 (Long gain: (145-140)*8.93 = +$44.65)
    # Combined profit: +$148.85
    # Live zscore converges to 0.30 (<= 0.50 exit threshold)
    metrics_map = {
        "XOM_CVX": PairSpreadMetrics(
            symbol_a="XOM",
            symbol_b="CVX",
            price_a=110.00,
            price_b=145.00,
            ratio=0.7586,
            mean_ratio=0.7500,
            std_ratio=0.0250,
            zscore=0.34,
            signal="NEUTRAL",
            status="ACTIVE",
        )
    }

    initial_tax_reserve = tax_engine.current_reserve
    engine.manage_active_pairs(metrics_map)

    # Position should be closed
    assert "XOM_CVX" not in engine.state.active_positions
    assert len(engine.state.closed_trades) == 1

    closed = engine.state.closed_trades[0]
    assert closed.pair_id == "XOM_CVX"
    assert closed.exit_reason == "MEAN_REVERSION"
    assert closed.net_realized_pnl > 0
    assert closed.tax_escrow == round(closed.net_realized_pnl * 0.30, 2)
    assert tax_engine.current_reserve == pytest.approx(initial_tax_reserve + closed.tax_escrow, abs=0.02)

    notifier.notify_pair_trade_close.assert_called_once()


def test_pairs_trading_divergence_stop_loss(mock_config, mock_options_client, tax_engine, notifier):
    """Verifies that when z-score diverges beyond |z| >= 3.50, downside stop-loss triggers."""
    engine = PairsTradingEngine(
        config=mock_config,
        options_client=mock_options_client,
        tax_engine=tax_engine,
        notifier=notifier,
    )

    pos = ActivePairPosition(
        pair_id="KO_PEP",
        symbol_a="KO",
        symbol_b="PEP",
        leg_a=PairLeg(symbol="KO", side="BUY", qty=18.0, entry_price=70.0, cost_basis=1260.0),
        leg_b=PairLeg(symbol="PEP", side="SELL", qty=7.5, entry_price=168.0, cost_basis=1260.0),
        entry_ratio=0.4167,
        entry_zscore=-2.2,
        total_notional=2520.0,
    )
    engine.state.active_positions["KO_PEP"] = pos

    # Extreme divergence: ratio drops further, zscore hits -3.70 (<= -3.50)
    metrics_map = {
        "KO_PEP": PairSpreadMetrics(
            symbol_a="KO",
            symbol_b="PEP",
            price_a=62.00,
            price_b=180.00,
            ratio=0.3444,
            mean_ratio=0.4500,
            std_ratio=0.0280,
            zscore=-3.77,
            signal="BUY_A_SELL_B",
            status="ACTIVE",
        )
    }

    engine.manage_active_pairs(metrics_map)

    assert "KO_PEP" not in engine.state.active_positions
    assert len(engine.state.closed_trades) == 1
    assert engine.state.closed_trades[0].exit_reason == "DIVERGENCE_STOP"


def test_pairs_trading_time_stop(mock_config, mock_options_client, tax_engine, notifier):
    """Verifies that an active pair trade held >= 20 calendar days is liquidated via time stop."""
    engine = PairsTradingEngine(
        config=mock_config,
        options_client=mock_options_client,
        tax_engine=tax_engine,
        notifier=notifier,
    )

    past_time = (datetime.now(timezone.utc) - timedelta(days=22)).isoformat()
    pos = ActivePairPosition(
        pair_id="GOOGL_MSFT",
        symbol_a="GOOGL",
        symbol_b="MSFT",
        leg_a=PairLeg(symbol="GOOGL", side="BUY", qty=7.5, entry_price=165.0, cost_basis=1237.5),
        leg_b=PairLeg(symbol="MSFT", side="SELL", qty=3.0, entry_price=420.0, cost_basis=1260.0),
        entry_ratio=0.3929,
        entry_zscore=-2.1,
        entry_time=past_time,
        total_notional=2497.5,
    )
    engine.state.active_positions["GOOGL_MSFT"] = pos

    # zscore remains unresolved at -1.20 (between exit 0.50 and stop 3.50)
    metrics_map = {
        "GOOGL_MSFT": PairSpreadMetrics(
            symbol_a="GOOGL",
            symbol_b="MSFT",
            price_a=166.00,
            price_b=418.00,
            ratio=0.3971,
            mean_ratio=0.4200,
            std_ratio=0.0190,
            zscore=-1.21,
            signal="NEUTRAL",
            status="ACTIVE",
        )
    }

    engine.manage_active_pairs(metrics_map)

    assert "GOOGL_MSFT" not in engine.state.active_positions
    assert len(engine.state.closed_trades) == 1
    assert engine.state.closed_trades[0].exit_reason == "TIME_STOP"


def test_pairs_trading_check_stale_orders(mock_config, mock_options_client, tax_engine, notifier):
    """Verifies that pending limit orders older than 24 hours are cancelled."""
    engine = PairsTradingEngine(
        config=mock_config,
        options_client=mock_options_client,
        tax_engine=tax_engine,
        notifier=notifier,
    )

    now = time.time()
    engine.state.pending_orders = {
        "order_recent": {"placed_at": now - 3600},       # 1 hr old
        "order_stale": {"placed_at": now - (25 * 3600)}, # 25 hrs old
    }

    mock_options_client.cancel_order = MagicMock(return_value=True)

    cancelled = engine.check_stale_orders()
    assert cancelled == 1
    assert "order_recent" in engine.state.pending_orders
    assert "order_stale" not in engine.state.pending_orders
    mock_options_client.cancel_order.assert_called_once_with("order_stale")


def test_pairs_trading_state_persistence(mock_config, mock_options_client, tax_engine, notifier):
    """Verifies atomic state saving and reloading from JSON."""
    engine = PairsTradingEngine(
        config=mock_config,
        options_client=mock_options_client,
        tax_engine=tax_engine,
        notifier=notifier,
    )

    pos = ActivePairPosition(
        pair_id="XOM_CVX",
        symbol_a="XOM",
        symbol_b="CVX",
        leg_a=PairLeg(symbol="XOM", side="SELL", qty=10.0, entry_price=115.0, cost_basis=1150.0),
        leg_b=PairLeg(symbol="CVX", side="BUY", qty=8.0, entry_price=150.0, cost_basis=1200.0),
        entry_ratio=0.7667,
        entry_zscore=2.4,
        total_notional=2350.0,
    )
    engine.state.active_positions["XOM_CVX"] = pos
    engine.state.total_realized_pnl = 350.0
    engine.state.total_tax_escrow = 105.0
    engine._save_state()

    assert Path(mock_config.PAIRS_STATE_FILE).exists()

    # Load in fresh instance
    engine2 = PairsTradingEngine(
        config=mock_config,
        options_client=mock_options_client,
        tax_engine=tax_engine,
        notifier=notifier,
    )
    assert "XOM_CVX" in engine2.state.active_positions
    assert engine2.state.active_positions["XOM_CVX"].symbol_a == "XOM"
    assert engine2.state.total_realized_pnl == 350.0
    assert engine2.state.total_tax_escrow == 105.0
