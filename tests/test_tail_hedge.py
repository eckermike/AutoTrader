"""
Unit tests for Black Swan Tail-Risk Crash Hedge Strategy (options/tail_hedge_engine.py).
Tests contract discovery, buy execution, 250% monetization exit, 21 DTE roll defense,
30% tax escrow segregation, monthly budget capping, and JSON persistence.
"""

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from config import BotConfig
from notifier import TradeNotifier
from options.options_client import AlpacaOptionsClient, OptionContractInfo
from options.tail_hedge_engine import TailHedgeEngine, TailHedgePosition
from tax_engine import TaxEngine


@pytest.fixture
def mock_config(tmp_path):
    tax_file = tmp_path / "test_hedge_tax.json"
    hedge_file = tmp_path / "test_hedge_state.json"
    return BotConfig(
        ALPACA_API_KEY="MOCK_KEY",
        ALPACA_SECRET_KEY="MOCK_SECRET",
        ALPACA_PAPER=True,
        TAX_RESERVE_FILE=tax_file,
        TAX_RATE=0.30,
        HEDGE_ENABLED=True,
        HEDGE_UNDERLYING="SPY",
        HEDGE_TARGET_DTE_MIN=45,
        HEDGE_TARGET_DTE_MAX=90,
        HEDGE_OTM_PCT=0.15,
        HEDGE_MAX_COST_PER_CONTRACT_USD=1.00,
        HEDGE_MONTHLY_BUDGET_USD=150.00,
        HEDGE_PROFIT_TARGET_PCT=2.50,
        HEDGE_ROLL_DTE=21,
        HEDGE_STATE_FILE=str(hedge_file),
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


def test_tail_hedge_contract_selection(mock_config, mock_options_client, tax_engine, notifier):
    """Verifies that the engine selects an OTM put ~15% below spot price with price <= max cost."""
    engine = TailHedgeEngine(
        config=mock_config,
        options_client=mock_options_client,
        tax_engine=tax_engine,
        notifier=notifier,
    )

    current_price = 540.0
    contract = engine.select_hedge_contract(current_price)

    assert contract is not None
    assert contract.contract_type == "put"
    assert contract.strike_price < current_price
    # ~15% OTM strike: 540 * 0.85 = ~459
    assert 440.0 <= contract.strike_price <= 480.0
    assert contract.days_to_expiration >= mock_config.HEDGE_TARGET_DTE_MIN


def test_tail_hedge_buy_order_execution(mock_config, mock_options_client, tax_engine, notifier):
    """Verifies stepping when flat submits a buy order, initializes active hedge, and updates budget."""
    engine = TailHedgeEngine(
        config=mock_config,
        options_client=mock_options_client,
        tax_engine=tax_engine,
        notifier=notifier,
    )

    with patch.object(notifier, "notify_tail_hedge_open") as mock_notify:
        res = engine.step(is_market_open=True)

        assert res["status"] == "OPENED"
        assert engine.active_hedge is not None
        assert engine.active_hedge.status == "ACTIVE"
        assert engine.active_hedge.qty == 1
        assert engine.active_hedge.entry_total_cost > 0.0
        assert engine.monthly_spent_usd == engine.active_hedge.entry_total_cost
        assert mock_notify.called


def test_tail_hedge_budget_ceiling_gating(mock_config, mock_options_client, tax_engine, notifier):
    """Verifies that no new hedge is purchased if the monthly budget is already exhausted."""
    engine = TailHedgeEngine(
        config=mock_config,
        options_client=mock_options_client,
        tax_engine=tax_engine,
        notifier=notifier,
    )

    # Exhaust monthly budget
    engine.monthly_spent_usd = 150.00
    res = engine.step(is_market_open=True)

    assert res["status"] == "MONTHLY_BUDGET_REACHED"
    assert engine.active_hedge is None


def test_tail_hedge_monetization_at_250_percent(mock_config, mock_options_client, tax_engine, notifier):
    """Verifies that when option price surges +250%, engine sells to close and records 30% tax."""
    engine = TailHedgeEngine(
        config=mock_config,
        options_client=mock_options_client,
        tax_engine=tax_engine,
        notifier=notifier,
    )

    # Open hedge at $0.50/sh ($50 total)
    engine.step(is_market_open=True)
    assert engine.active_hedge is not None
    engine.active_hedge.entry_price_per_share = 0.50
    engine.active_hedge.entry_total_cost = 50.0
    engine.active_hedge.target_monetization_price = 1.75  # +250%

    # Simulate market crash: quote spikes to $2.00/sh ($200 market value = +$150 gain)
    with patch.object(mock_options_client, "get_option_quote", return_value={"bid_price": 2.00, "ask_price": 2.10}), \
         patch.object(notifier, "notify_tail_hedge_monetized") as mock_monetize_alert:
        
        res = engine.step(is_market_open=True)

        assert res["status"] == "CLOSED"
        assert res["reason"] == "MONETIZATION_PROFIT_TARGET"
        assert res["realized_pnl"] == 150.0
        assert engine.active_hedge is None
        assert len(engine.closed_hedges) == 1
        assert mock_monetize_alert.called

        # Verify Tax Escrow: 30% of $150 = $45 allocated
        assert tax_engine.tax_reserve == 45.0
        assert tax_engine.total_realized_profit == 150.0


def test_tail_hedge_21_dte_roll_defense(mock_config, mock_options_client, tax_engine, notifier):
    """Verifies that when DTE reaches <= 21, the contract is closed to preserve capital before theta crush."""
    engine = TailHedgeEngine(
        config=mock_config,
        options_client=mock_options_client,
        tax_engine=tax_engine,
        notifier=notifier,
    )

    engine.step(is_market_open=True)
    assert engine.active_hedge is not None

    # Set expiration date to 15 days in future
    future_date = (date.today() + timedelta(days=15)).strftime("%Y-%m-%d")
    engine.active_hedge.expiration_date = future_date

    with patch.object(mock_options_client, "get_option_quote", return_value={"bid_price": 0.20, "ask_price": 0.25}), \
         patch.object(notifier, "notify_tail_hedge_rolled") as mock_roll_alert:
        
        res = engine.step(is_market_open=True)

        assert res["status"] == "CLOSED"
        assert res["reason"] == "THETA_DEFENSE_ROLL_21_DTE"
        assert engine.active_hedge is None
        assert mock_roll_alert.called


def test_tail_hedge_state_persistence_and_reload(mock_config, mock_options_client, tax_engine, notifier):
    """Verifies that active hedge and monthly spending persist across restarts."""
    engine1 = TailHedgeEngine(
        config=mock_config,
        options_client=mock_options_client,
        tax_engine=tax_engine,
        notifier=notifier,
    )
    engine1.step(is_market_open=True)
    assert engine1.active_hedge is not None
    saved_symbol = engine1.active_hedge.contract_symbol
    spent = engine1.monthly_spent_usd

    # Instantiate new engine pointing to same state file
    engine2 = TailHedgeEngine(
        config=mock_config,
        options_client=mock_options_client,
        tax_engine=tax_engine,
        notifier=notifier,
    )
    assert engine2.active_hedge is not None
    assert engine2.active_hedge.contract_symbol == saved_symbol
    assert engine2.monthly_spent_usd == spent


def test_tail_hedge_capital_gate_rejects_if_insufficient_cash(mock_config, mock_options_client, tax_engine, notifier):
    """Verifies that if tradable cash is insufficient, entry is rejected and liquidity manager notified."""
    mock_liquidity_mgr = MagicMock()
    engine = TailHedgeEngine(
        config=mock_config,
        options_client=mock_options_client,
        tax_engine=tax_engine,
        notifier=notifier,
        liquidity_manager=mock_liquidity_mgr,
    )

    # Mock zero tradable cash
    with patch.object(tax_engine, "get_tradable_cash", return_value=0.0):
        res = engine.step(is_market_open=True)
        assert res["status"] == "CAPITAL_GATE_REJECTED"
        assert engine.active_hedge is None
        assert mock_liquidity_mgr.request_liquidation_for_opportunity.called
