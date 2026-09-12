"""
Unit tests for TradeNotifier.
Tests iMessage formatting, alert payloads, and error resilience.
"""

from unittest.mock import patch, MagicMock
import pytest
from notifier import TradeNotifier


def test_notifier_formatting_buy():
    notifier = TradeNotifier(recipient="test@icloud.com", ntfy_topic="test_topic", enabled=True, macos_banner=False)
    
    with patch.object(notifier, "send_imessage") as mock_imsg, patch.object(notifier, "send_ntfy") as mock_ntfy:
        mock_imsg.return_value = True
        mock_ntfy.return_value = True
        
        notifier.notify_buy(
            symbol="BTC/USD",
            price=80000.0,
            notional=500.0,
            qty=0.00625,
            composite_score=0.650,
            tech_score=0.700,
            vol_score=0.550,
            sent_score=0.700,
            tradable_cash=95000.0,
            tax_reserve=1500.0,
        )
        
        assert mock_ntfy.called
        assert mock_imsg.called
        msg = mock_ntfy.call_args[1]["message"]
        assert "BUY ORDER EXECUTED" in msg
        assert "BTC/USD" in msg
        assert "$80,000.00" in msg
        assert "Composite: +0.650" in msg
        assert "Tradable Cash: $95,000.00" in msg


def test_notifier_formatting_sell():
    notifier = TradeNotifier(recipient="test@icloud.com", ntfy_topic="test_topic", enabled=True, macos_banner=False)
    
    with patch.object(notifier, "send_imessage") as mock_imsg, patch.object(notifier, "send_ntfy") as mock_ntfy:
        mock_imsg.return_value = True
        mock_ntfy.return_value = True
        
        notifier.notify_sell(
            symbol="BTC/USD",
            exit_price=85000.0,
            qty=0.00625,
            reason="Trailing Stop Triggered",
            gross_pnl=31.25,
            tax_allocated=9.38,
            tax_credit=0.0,
            reserve_after=1509.38,
        )
        
        assert mock_ntfy.called
        assert mock_imsg.called
        msg = mock_ntfy.call_args[1]["message"]
        assert "SELL ORDER EXECUTED" in msg
        assert "BTC/USD" in msg
        assert "$85,000.00" in msg
        assert "Gross Realized PnL: +$31.25" in msg
        assert "Tax Allocated (30%): +$9.38" in msg


def test_notifier_disabled():
    notifier = TradeNotifier(recipient="test@icloud.com", enabled=False)
    # When disabled, should return False immediately
    assert notifier.send_imessage("Test message") is False


def test_notifier_spread_open_and_close():
    notifier = TradeNotifier(recipient="test@icloud.com", ntfy_topic="test_topic", enabled=True, macos_banner=False)

    with patch.object(notifier, "send_imessage", return_value=True) as mock_imsg, \
         patch.object(notifier, "send_ntfy", return_value=True) as mock_ntfy:

        # Test Spread Open Alert
        notifier.notify_spread_open(
            underlying="SPY",
            short_strike=530.0,
            long_strike=525.0,
            expiration="2026-10-16",
            dte=34,
            net_credit=0.85,
            collateral_locked=500.0,
            max_profit=85.0,
            target_exit_profit=42.50,
        )
        assert mock_ntfy.called
        msg = mock_ntfy.call_args[1]["message"]
        assert "SPREAD: BULL PUT OPENED" in msg
        assert "SPY" in msg
        assert "$530.00P" in msg
        assert "+$85.00" in msg

        # Test Spread Close Alert (Profit Target)
        notifier.notify_spread_close(
            underlying="SPY",
            short_strike=530.0,
            long_strike=525.0,
            reason="PROFIT_TARGET_50_PCT",
            realized_pnl=42.50,
            tax_allocated=12.75,
            tax_reserve_after=127.50,
        )
        close_msg = mock_ntfy.call_args[1]["message"]
        assert "SPREAD: POSITION CLOSED" in close_msg
        assert "+$42.50" in close_msg
        assert "PROFIT_TARGET_50_PCT" in close_msg
        assert "Tax Escrow (30% Allocated): +$12.75" in close_msg
        assert "Tax Reserve Balance: $127.50" in close_msg
