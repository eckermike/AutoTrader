"""
Unit and integration tests for Defined-Risk Option Spreads Engine.
Validates:
1. Multi-leg contract pair selection (short put + long put $5 lower).
2. Alpaca MLEG order execution and negative limit credit formatting.
3. 50% Profit Target Buy-to-Close early exit and TaxEngine integration.
4. Stop-loss protection trigger at 2.5x credit.
5. Portfolio capital ceiling and TaxEngine hard capital gating.
6. Persistent JSON state save/load cycle.
"""

import json
import os
import pytest
from unittest.mock import MagicMock, patch

from config import BotConfig
from notifier import TradeNotifier
from options.options_client import AlpacaOptionsClient, OptionContractInfo
from options.spread_engine import (
    SpreadEngine,
    SpreadPortfolioManager,
    SpreadPosition,
)
from tax_engine import TaxEngine


@pytest.fixture
def mock_config(tmp_path):
    """Provides a test configuration using temporary files."""
    tax_file = str(tmp_path / "test_tax_reserve.json")
    spread_file = str(tmp_path / "test_spreads_state.json")
    return BotConfig(
        ALPACA_API_KEY="MOCK_KEY",
        ALPACA_SECRET_KEY="MOCK_SECRET",
        ALPACA_PAPER=True,
        TAX_RESERVE_FILE=tax_file,
        TAX_RATE=0.30,
        SPREAD_ENABLED=True,
        SPREAD_SYMBOLS=["SPY", "QQQ", "IWM"],
        SPREAD_WIDTH_USD=5.0,
        SPREAD_TARGET_DELTA=0.20,  # ~20 delta (~80% probability OTM)
        SPREAD_MIN_CREDIT_USD=0.50,
        SPREAD_PROFIT_TARGET_PCT=0.50,
        SPREAD_STOP_LOSS_RATIO=2.50,
        SPREAD_MAX_CAPITAL_USD=5000.0,
        SPREAD_ORDER_QTY=1,
        SPREAD_STATE_FILE=spread_file,
        ALERT_ENABLED=False,
    )


@pytest.fixture
def mock_options_client():
    """Mock AlpacaOptionsClient with realistic option chain responses."""
    client = AlpacaOptionsClient("MOCK", "MOCK", paper=True, mock_mode=True)
    return client


@pytest.fixture
def tax_engine(mock_config):
    """Initializes a fresh TaxEngine."""
    return TaxEngine(filepath=mock_config.TAX_RESERVE_FILE, tax_rate=mock_config.TAX_RATE)


@pytest.fixture
def notifier(mock_config):
    """Initializes a mock TradeNotifier."""
    return TradeNotifier(recipient="test@example.com", enabled=False)


def test_spread_contract_selection(mock_config, mock_options_client, tax_engine, notifier):
    """Verifies that the engine selects a valid pair with matching expiry and $5 strike width."""
    engine = SpreadEngine(
        config=mock_config,
        options_client=mock_options_client,
        tax_engine=tax_engine,
        notifier=notifier,
        symbol="SPY",
    )

    pair = engine.select_spread_contracts(current_price=560.0)
    assert pair is not None
    short_c = pair["short"]
    long_c = pair["long"]

    assert short_c.strike_price < 560.0
    assert long_c.strike_price == short_c.strike_price - mock_config.SPREAD_WIDTH_USD
    assert short_c.expiration_date == long_c.expiration_date
    assert short_c.contract_type == "put"
    assert long_c.contract_type == "put"


def test_spread_mleg_order_submission(mock_config, mock_options_client, tax_engine, notifier):
    """Verifies opening a new credit spread and checking position tracking."""
    engine = SpreadEngine(
        config=mock_config,
        options_client=mock_options_client,
        tax_engine=tax_engine,
        notifier=notifier,
        symbol="SPY",
    )

    res = engine.step(current_spread_collateral=0.0)
    assert res["status"] == "OPENED"
    assert engine.active_spread is not None
    assert engine.active_spread.underlying == "SPY"
    assert engine.active_spread.collateral_locked == 500.0
    assert engine.active_spread.entry_credit > 0
    assert engine.active_spread.status == "ACTIVE"


def test_spread_50_percent_profit_early_exit(mock_config, mock_options_client, tax_engine, notifier):
    """Verifies that when spread price drops to 50% of credit, the position closes and allocates tax."""
    engine = SpreadEngine(
        config=mock_config,
        options_client=mock_options_client,
        tax_engine=tax_engine,
        notifier=notifier,
        symbol="SPY",
    )

    # Open spread with $1.00 credit
    engine.step(current_spread_collateral=0.0)
    assert engine.active_spread is not None
    engine.active_spread.entry_credit = 1.00
    engine.active_spread.max_profit = 100.00

    # Mock quote where buyback cost is $0.45 (<= $0.50 target)
    with patch.object(mock_options_client, "get_spread_quote", return_value={"close_debit": 0.45}):
        res = engine.step(current_spread_collateral=500.0)

    assert res["status"] == "CLOSED"
    assert res["reason"] == "PROFIT_TARGET_50_PCT"
    assert res["realized_pnl"] == 55.00  # $100 entry - $45 close = $55 profit
    assert engine.active_spread is None

    # Verify TaxEngine recorded the profit and allocated 30%
    assert tax_engine.total_realized_profit == 55.00
    assert tax_engine.tax_reserve == pytest.approx(16.50, 0.01)  # 30% of $55.00


def test_spread_stop_loss_guard(mock_config, mock_options_client, tax_engine, notifier):
    """Verifies that when spread expands to 2.5x credit, stop-loss triggers."""
    engine = SpreadEngine(
        config=mock_config,
        options_client=mock_options_client,
        tax_engine=tax_engine,
        notifier=notifier,
        symbol="SPY",
    )

    engine.step(current_spread_collateral=0.0)
    engine.active_spread.entry_credit = 0.80

    # Mock quote where buyback cost is $2.10 (>= 2.5x $0.80 = $2.00)
    with patch.object(mock_options_client, "get_spread_quote", return_value={"close_debit": 2.10}):
        res = engine.step(current_spread_collateral=500.0)

    assert res["status"] == "CLOSED"
    assert res["reason"] == "STOP_LOSS_GUARD"
    assert res["realized_pnl"] < 0  # Loss


def test_spread_capital_ceiling_gating(mock_config, mock_options_client, tax_engine, notifier):
    """Verifies that no new spread opens if portfolio collateral budget is exceeded."""
    engine = SpreadEngine(
        config=mock_config,
        options_client=mock_options_client,
        tax_engine=tax_engine,
        notifier=notifier,
        symbol="SPY",
    )

    # Budget is $5,000; pass $4,800 existing collateral -> next $500 spread would exceed $5,000
    res = engine.step(current_spread_collateral=4800.0)
    assert res["status"] == "BUDGET_CAPPED"
    assert engine.active_spread is None


def test_spread_portfolio_manager_lifecycle(mock_config, mock_options_client, tax_engine, notifier):
    """Verifies SpreadPortfolioManager coordination across SPY, QQQ, and IWM, and state saving."""
    manager = SpreadPortfolioManager(
        config=mock_config,
        options_client=mock_options_client,
        tax_engine=tax_engine,
        notifier=notifier,
    )

    # Step all symbols during market hours
    results = manager.step_all(is_market_open=True)
    assert "SPY" in results
    assert "QQQ" in results
    assert "IWM" in results

    metrics = manager.get_summary_metrics()
    assert metrics["active_count"] >= 1
    assert metrics["collateral_locked"] > 0

    # Verify state was saved to JSON
    assert os.path.exists(mock_config.SPREAD_STATE_FILE)

    # Reload into a new manager and verify state persistence
    manager2 = SpreadPortfolioManager(
        config=mock_config,
        options_client=mock_options_client,
        tax_engine=tax_engine,
        notifier=notifier,
    )
    metrics2 = manager2.get_summary_metrics()
    assert metrics2["active_count"] == metrics["active_count"]
    assert metrics2["collateral_locked"] == metrics["collateral_locked"]
