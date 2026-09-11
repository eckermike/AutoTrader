"""
Unit tests for Autonomous LiquidityManager & interactive ntfy approval engine.
"""

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
