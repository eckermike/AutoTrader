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
