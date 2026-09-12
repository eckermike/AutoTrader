"""
Unit tests for Autonomous LiquidityManager & interactive ntfy approval engine.
Includes tests for:
- Layer 1: Time-To-Live (4h TTL) expiration window.
- Layer 2: Pre-execution viability and slippage reassessment gate.
"""

import time
from unittest.mock import MagicMock, patch
import pytest

from config import BotConfig
from execution.liquidity_manager import LiquidityManager, LiquidityState, PendingApproval
from notifier import TradeNotifier


@pytest.fixture
def mock_config(tmp_path):
    state_path = tmp_path / "test_liquidity_state.json"
    return BotConfig(
        ALPACA_API_KEY="MOCK_KEY",
        ALPACA_SECRET_KEY="MOCK_SECRET",
        ALPACA_PAPER=True,
        NTFY_TOPIC="test_topic",
        NTFY_ACTION_TOPIC="test_actions_topic",
        LIQUIDITY_RESERVE_ENABLED=True,
        SGOV_ALLOCATION_USD=20000.0,
        FBND_ALLOCATION_USD=20000.0,
        LIQUIDITY_STATE_FILE=str(state_path),
        APPROVAL_ON_LIQUIDATION_ONLY=True,
        APPROVAL_TTL_HOURS=4.0,
        APPROVAL_MAX_SLIPPAGE_PCT=0.005,
    )


@pytest.fixture
def mock_notifier():
    notifier = MagicMock(spec=TradeNotifier)
    return notifier


@pytest.fixture
def mock_trading_client():
    client = MagicMock()
    # Mock order submission
    def fake_submit(order_req):
        mock_order = MagicMock()
        mock_order.id = f"ORDER-{order_req.symbol}-123"
        mock_order.status = "ACCEPTED"
        mock_order.symbol = order_req.symbol
        return mock_order

    client.submit_order.side_effect = fake_submit
    return client


def test_liquidity_manager_dispatch_approval_request(mock_config, mock_notifier, mock_trading_client):
    manager = LiquidityManager(
        config=mock_config,
        trading_client=mock_trading_client,
        notifier=mock_notifier,
        state_file=mock_config.LIQUIDITY_STATE_FILE,
    )

    dispatched = manager.dispatch_5050_bond_request()
    assert dispatched is True
    assert manager.state.pending_approval is not None
    assert manager.state.pending_approval.status == "PENDING"
    assert manager.state.pending_approval.sgov_amount == 20000.0
    assert manager.state.pending_approval.fbnd_amount == 20000.0

    # Verify notifier was called
    mock_notifier.notify_approval_request.assert_called_once()
    args, kwargs = mock_notifier.notify_approval_request.call_args
    assert "Capital Allocation Request" in kwargs["proposal_title"]
    assert kwargs["action_topic"] == "test_actions_topic"
    assert kwargs["approve_body"] == "APPROVE_5050_BONDS"

    # Second call should not re-dispatch while pending
    second_call = manager.dispatch_5050_bond_request()
    assert second_call is False


def test_liquidity_manager_approval_execution(mock_config, mock_notifier, mock_trading_client):
    manager = LiquidityManager(
        config=mock_config,
        trading_client=mock_trading_client,
        notifier=mock_notifier,
        state_file=mock_config.LIQUIDITY_STATE_FILE,
    )

    manager.dispatch_5050_bond_request()
    assert manager.state.pending_approval is not None

    # Mock polling response returning APPROVE_5050_BONDS
    with patch.object(manager, "poll_action_topic", return_value=["APPROVE_5050_BONDS"]):
        result = manager.step()
        assert result == "APPROVED"

    # Verify orders were submitted
    assert mock_trading_client.submit_order.call_count == 2
    symbols_ordered = [call[0][0].symbol for call in mock_trading_client.submit_order.call_args_list]
    assert "SGOV" in symbols_ordered
    assert "FBND" in symbols_ordered

    # Verify state is cleared and recorded in history
    assert manager.state.pending_approval is None
    assert len(manager.state.history) == 1
    assert manager.state.history[0]["status"] == "EXECUTED"
    assert len(manager.state.history[0]["order_ids"]) == 2

    # Verify resolution notification
    mock_notifier.notify_approval_resolution.assert_called_once()
    assert mock_notifier.notify_approval_resolution.call_args[1]["approved"] is True


def test_liquidity_manager_rejection(mock_config, mock_notifier, mock_trading_client):
    manager = LiquidityManager(
        config=mock_config,
        trading_client=mock_trading_client,
        notifier=mock_notifier,
        state_file=mock_config.LIQUIDITY_STATE_FILE,
    )

    manager.dispatch_5050_bond_request()
    assert manager.state.pending_approval is not None

    # Mock polling response returning REJECT_5050_BONDS
    with patch.object(manager, "poll_action_topic", return_value=["REJECT_5050_BONDS"]):
        result = manager.step()
        assert result == "REJECTED"

    # Verify NO orders were submitted
    assert mock_trading_client.submit_order.call_count == 0

    # Verify state is cleared and history recorded
    assert manager.state.pending_approval is None
    assert len(manager.state.history) == 1
    assert manager.state.history[0]["status"] == "REJECTED"

    # Verify cancellation notification
    mock_notifier.notify_approval_resolution.assert_called_once()
    assert mock_notifier.notify_approval_resolution.call_args[1]["approved"] is False


def test_liquidity_manager_ttl_expiration(mock_config, mock_notifier, mock_trading_client):
    """Test Layer 1 TTL expiration: requests older than 4.0 hours are automatically expired."""
    manager = LiquidityManager(
        config=mock_config,
        trading_client=mock_trading_client,
        notifier=mock_notifier,
        state_file=mock_config.LIQUIDITY_STATE_FILE,
    )

    manager.dispatch_5050_bond_request()
    assert manager.state.pending_approval is not None

    # Simulate 4.5 hours passing
    manager.state.pending_approval.created_timestamp = time.time() - (4.5 * 3600)
    manager._save_state()

    result = manager.step()
    assert result == "EXPIRED"

    # Verify state is cleared and recorded as EXPIRED
    assert manager.state.pending_approval is None
    assert len(manager.state.history) == 1
    assert manager.state.history[0]["status"] == "EXPIRED"

    # Verify notification sent to user
    mock_notifier.notify_approval_resolution.assert_called_once()
    assert "Approval Expired" in mock_notifier.notify_approval_resolution.call_args[1]["title"]


def test_liquidity_manager_layer2_slippage_failure(mock_config, mock_notifier, mock_trading_client):
    """Test Layer 2 Pre-Execution Viability: price moves >0.5% aborts execution."""
    manager = LiquidityManager(
        config=mock_config,
        trading_client=mock_trading_client,
        notifier=mock_notifier,
        state_file=mock_config.LIQUIDITY_STATE_FILE,
    )

    # Dispatch opportunity request with reference price $100.00
    manager.request_liquidation_for_opportunity(
        needed_cash=10000.0,
        target_symbol="PLTR",
        opportunity_type="OPTION_WHEEL_PUT",
        current_price=100.00,
    )
    assert manager.state.pending_approval is not None

    # Mock live quote showing price moved to $102.00 (+2.0% slippage > 0.5% max)
    mock_quote = MagicMock()
    mock_quote.ask_price = 102.00
    mock_trading_client.get_latest_quote.return_value = mock_quote

    # User taps approve
    with patch.object(manager, "poll_action_topic", return_value=["APPROVE_LIQUIDATION"]):
        result = manager.step()
        assert result == "ABORTED_VIABILITY"

    # Verify NO broker orders were submitted
    assert mock_trading_client.submit_order.call_count == 0

    # Verify state recorded as ABORTED_VIABILITY
    assert manager.state.pending_approval is None
    assert len(manager.state.history) == 1
    assert manager.state.history[0]["status"] == "ABORTED_VIABILITY"

    # Verify abort alert sent to user
    mock_notifier.notify_approval_resolution.assert_called_once()
    assert "Execution Aborted" in mock_notifier.notify_approval_resolution.call_args[1]["title"]


def test_liquidity_manager_layer2_slippage_success(mock_config, mock_notifier, mock_trading_client):
    """Test Layer 2 Pre-Execution Viability: price moves within 0.5% passes and executes."""
    manager = LiquidityManager(
        config=mock_config,
        trading_client=mock_trading_client,
        notifier=mock_notifier,
        state_file=mock_config.LIQUIDITY_STATE_FILE,
    )

    # Dispatch opportunity request with reference price $100.00
    manager.request_liquidation_for_opportunity(
        needed_cash=10000.0,
        target_symbol="PLTR",
        opportunity_type="OPTION_WHEEL_PUT",
        current_price=100.00,
    )
    assert manager.state.pending_approval is not None

    # Mock live quote showing price moved slightly to $100.20 (+0.2% slippage <= 0.5% max)
    mock_quote = MagicMock()
    mock_quote.ask_price = 100.20
    mock_trading_client.get_latest_quote.return_value = mock_quote

    # User taps approve
    with patch.object(manager, "poll_action_topic", return_value=["APPROVE_LIQUIDATION"]):
        result = manager.step()
        assert result == "APPROVED"

    # Verify liquidation order submitted to sell SGOV
    assert mock_trading_client.submit_order.call_count == 1
    order_req = mock_trading_client.submit_order.call_args[0][0]
    assert order_req.symbol == "SGOV"
    assert order_req.notional == 10000.0

    # Verify state recorded as EXECUTED
    assert manager.state.pending_approval is None
    assert len(manager.state.history) == 1
    assert manager.state.history[0]["status"] == "EXECUTED"

    # Verify execution confirmation sent
    mock_notifier.notify_approval_resolution.assert_called_once()
    assert "Liquidation Executed" in mock_notifier.notify_approval_resolution.call_args[1]["title"]


def test_liquidity_manager_dividend_auto_escrow(mock_config, mock_notifier, mock_trading_client, tmp_path):
    """Test autonomous dividend capture: 30% of incoming cash dividends are auto-escrowed into TaxEngine."""
    from tax_engine import TaxEngine
    tax_file = tmp_path / "tax_reserve_test.json"
    tax_engine = TaxEngine(filepath=tax_file, tax_rate=0.30)

    manager = LiquidityManager(
        config=mock_config,
        trading_client=mock_trading_client,
        notifier=mock_notifier,
        tax_engine=tax_engine,
        state_file=mock_config.LIQUIDITY_STATE_FILE,
    )

    # Mock Alpaca dividend activity
    mock_div = MagicMock()
    mock_div.id = "DIV-ACT-12345"
    mock_div.symbol = "SGOV"
    mock_div.net_amount = 85.00
    mock_trading_client.get_activities.return_value = [mock_div]

    # Run step
    with patch.object(manager, "poll_action_topic", return_value=[]):
        manager.step()

    # Verify dividend was recorded in tax_engine
    assert tax_engine.current_reserve == 25.50  # 30% of $85.00
    assert tax_engine.state.total_realized_profit == 85.00
    assert "DIV-ACT-12345" in manager.state.processed_dividend_ids

    # Verify push notification sent
    mock_notifier.send_ntfy.assert_called_once()
    assert "Dividend Tax Escrow" in mock_notifier.send_ntfy.call_args[1]["title"]

    # Second step should be idempotent and not duplicate
    with patch.object(manager, "poll_action_topic", return_value=[]):
        manager.step()

    assert tax_engine.current_reserve == 25.50
    assert tax_engine.state.trade_count == 1


def test_wheel_capital_shortage_triggers_liquidation_request(mock_config, mock_notifier, tmp_path):
    """Verifies that when Wheel needs collateral exceeding cash, it dispatches an SGOV liquidation request."""
    from options.options_client import AlpacaOptionsClient
    from options.wheel_engine import WheelEngine
    from tax_engine import TaxEngine

    tax_engine = TaxEngine(filepath=tmp_path / "tax_res.json", tax_rate=0.30)
    options_client = AlpacaOptionsClient("MOCK", "MOCK", paper=True, mock_mode=True)
    mock_liq = MagicMock(spec=LiquidityManager)

    engine = WheelEngine(
        config=mock_config,
        options_client=options_client,
        tax_engine=tax_engine,
        notifier=mock_notifier,
        symbol="INTC",
        liquidity_manager=mock_liq,
    )

    # Step with 0 cash available
    engine.step(total_cash=0.0, available_tradable_cash=0.0)

    # Verify liquidation request dispatched to user's phone
    mock_liq.request_liquidation_for_opportunity.assert_called_once()
    kwargs = mock_liq.request_liquidation_for_opportunity.call_args[1]
    assert kwargs["target_symbol"] == "INTC"
    assert kwargs["needed_cash"] > 0
    assert "Option Wheel" in kwargs["opportunity_type"]


def test_spread_capital_shortage_triggers_liquidation_request(mock_config, mock_notifier, tmp_path):
    """Verifies that when Defined-Risk Spreads need collateral exceeding cash, it dispatches an SGOV liquidation request."""
    from options.options_client import AlpacaOptionsClient
    from options.spread_engine import SpreadEngine
    from tax_engine import TaxEngine

    tax_engine = TaxEngine(filepath=tmp_path / "tax_res.json", tax_rate=0.30)
    options_client = AlpacaOptionsClient("MOCK", "MOCK", paper=True, mock_mode=True)
    mock_liq = MagicMock(spec=LiquidityManager)

    engine = SpreadEngine(
        config=mock_config,
        options_client=options_client,
        tax_engine=tax_engine,
        notifier=mock_notifier,
        symbol="SPY",
        liquidity_manager=mock_liq,
    )

    # Mock tax_engine returning $0 tradable cash
    with patch.object(tax_engine, "get_tradable_cash", return_value=0.0):
        engine.step(current_spread_collateral=0.0)

    mock_liq.request_liquidation_for_opportunity.assert_called_once()
    kwargs = mock_liq.request_liquidation_for_opportunity.call_args[1]
    assert kwargs["target_symbol"] == "SPY"
    assert kwargs["needed_cash"] == 500.0  # $5.00 * 100
    assert "Defined-Risk Spread" in kwargs["opportunity_type"]


