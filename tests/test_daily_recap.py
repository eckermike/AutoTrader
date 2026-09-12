"""
Unit tests for End-of-Day Daily Recap & Zero-Trade Diagnostic Notifications.
Tests TradeNotifier message formatting, TradingDaemon diagnostic assembly,
and daily schedule idempotency.
"""

from unittest.mock import MagicMock, patch
import pytest
from config import BotConfig
from main import TradingDaemon
from notifier import TradeNotifier


def test_notifier_formatting_daily_recap_zero_trades():
    notifier = TradeNotifier(
        recipient="test@example.com",
        ntfy_topic="test_topic",
        enabled=True,
        macos_banner=False,
    )

    with patch.object(notifier, "send_ntfy", return_value=True) as mock_ntfy, \
         patch.object(notifier, "send_imessage", return_value=True) as mock_imsg:
        
        notifier.notify_daily_recap(
            date_str="2026-09-08",
            trades_count=0,
            crypto_diagnostics=[
                "AVAX/USD: Active position (61.71 units, +1.3%). 5% trailing stop active.",
                "BTC/USD: Peak score +0.160 (Trigger: +0.60). Market below momentum breakout threshold.",
            ],
            wheel_diagnostics=[
                "INTC: Short Put active (INTC261002P00089000). Waiting for 50% profit decay.",
                "F: Limit order pending in order book (F261002P00013500).",
            ],
            spread_diagnostics=[
                "SPY: Active Bull Put (530P/525P, exp 2026-10-16). PnL: +$25.00 (50% towards 50% target).",
                "QQQ: Idle / Staging next high-probability setup.",
            ],
            cash=100277.49,
            tradable_cash=100162.64,
            tax_reserve=114.85,
        )

        assert mock_ntfy.called
        assert mock_imsg.called

        msg = mock_ntfy.call_args[1]["message"]
        assert "AUTOTRADER DAILY BRIEFING — 2026-09-08" in msg
        assert "0 New Orders" in msg
        assert "WHY NO TRADES WERE TRIGGERED TODAY" in msg
        assert "AVAX/USD: Active position" in msg
        assert "BTC/USD: Peak score +0.160" in msg
        assert "INTC: Short Put active" in msg
        assert "Defined-Risk Option Spreads" in msg
        assert "SPY: Active Bull Put" in msg
        assert "Total Cash:    $100,277.49" in msg
        assert "Tradable Cash: $100,162.64" in msg
        assert "Tax Escrow:    $114.85" in msg


def test_notifier_formatting_daily_recap_with_trades():
    notifier = TradeNotifier(
        recipient="test@example.com",
        ntfy_topic="test_topic",
        enabled=True,
        macos_banner=False,
    )

    with patch.object(notifier, "send_ntfy", return_value=True) as mock_ntfy, \
         patch.object(notifier, "send_imessage", return_value=True) as mock_imsg:
        
        notifier.notify_daily_recap(
            date_str="2026-09-08",
            trades_count=3,
            crypto_diagnostics=["LINK/USD: Buy order executed"],
            wheel_diagnostics=["INTC: Put contract filled"],
            cash=100000.0,
            tradable_cash=99000.0,
            tax_reserve=300.0,
        )

        msg = mock_ntfy.call_args[1]["message"]
        assert "3 executed" in msg
        assert "ACTIVITY & STATUS BREAKDOWN" in msg


def test_daemon_daily_recap_assembly_and_idempotency(tmp_path):
    tax_file = tmp_path / "recap_tax.json"
    config = BotConfig(
        ALPACA_PAPER=True,
        TAX_RESERVE_FILE=tax_file,
        TARGET_SYMBOLS=["BTC/USD", "AVAX/USD"],
        DAILY_RECAP_ENABLED=True,
        DAILY_RECAP_HOUR=17,
        DAILY_RECAP_MINUTE=0,
    )

    daemon = TradingDaemon(config=config, dry_run=True)
    # Give AVAX an active position and BTC a peak score
    daemon.crypto_positions["AVAX/USD"] = {
        "qty": 50.0,
        "entry_price": 8.0,
        "peak_price": 8.5,
    }
    daemon.daily_max_composite_scores["BTC/USD"] = 0.25

    with patch.object(daemon.notifier, "notify_daily_recap") as mock_notify:
        # Force dispatch
        dispatched = daemon.check_and_dispatch_daily_recap(
            account_cash=100000.0,
            tradable_cash=99500.0,
            force=True,
        )
        assert dispatched is True
        assert mock_notify.called

        # Verify arguments passed to notifier
        call_kwargs = mock_notify.call_args[1]
        assert call_kwargs["trades_count"] == 0
        crypto_diag = call_kwargs["crypto_diagnostics"]
        assert any("AVAX/USD" in d for d in crypto_diag)
        assert any("BTC/USD" in d for d in crypto_diag)

        # Calling again without force should be blocked by idempotency (same day)
        second_dispatch = daemon.check_and_dispatch_daily_recap(
            account_cash=100000.0,
            tradable_cash=99500.0,
            force=False,
        )
        assert second_dispatch is False
