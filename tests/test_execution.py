"""
Unit tests for Alpaca broker client and execution daemon.
Tests strict paper trading enforcement, order submission, fractional sizing,
trailing stop-loss evaluation, and end-to-end cycle execution.
"""

import pytest
from config import BotConfig
from execution.alpaca_client import AlpacaPaperClient, OrderResult
from main import TradingDaemon


def test_strict_paper_trading_enforcement():
    # Attempting to initialize with paper=False must raise ValueError
    with pytest.raises(ValueError) as exc_info:
        AlpacaPaperClient(
            api_key="TEST_KEY",
            secret_key="TEST_SECRET",
            paper=False,
        )
    assert "Live trading is strictly disallowed" in str(exc_info.value)


def test_config_strict_paper_enforcement():
    # Attempting to set ALPACA_PAPER=False in BotConfig must raise ValueError
    with pytest.raises(ValueError) as exc_info:
        BotConfig(ALPACA_PAPER=False)
    assert "CRITICAL RISK VIOLATION" in str(exc_info.value)


def test_alpaca_mock_order_submission():
    client = AlpacaPaperClient(
        api_key="MOCK_KEY",
        secret_key="MOCK_SECRET",
        paper=True,
        mock_mode=True,
    )
    
    account_init = client.get_account()
    assert account_init.cash == 100000.00
    
    # Buy $500 of BTC/USD
    order = client.submit_market_order(
        symbol="BTC/USD",
        side="BUY",
        notional=500.0,
        estimated_price=50000.0,
    )
    
    assert isinstance(order, OrderResult)
    assert order.side == "BUY"
    assert order.time_in_force == "GTC"
    assert order.notional == 500.0
    assert order.qty == 0.01  # 500 / 50000
    
    # Verify account cash reduced
    account_after = client.get_account()
    assert account_after.cash == 99500.00
    
    # Verify position exists
    pos = client.get_crypto_position("BTC/USD")
    assert pos is not None
    assert pos.qty == 0.01
    assert pos.avg_entry_price == 50000.0
    
    # Close position
    close_res = client.close_crypto_position("BTC/USD")
    assert close_res is not None
    assert close_res.side == "SELL"
    
    # Verify position is flat
    pos_flat = client.get_crypto_position("BTC/USD")
    assert pos_flat is None


def test_daemon_full_cycle_dry_run(tmp_path):
    test_tax_file = tmp_path / "daemon_tax_reserve.json"
    config = BotConfig(
        ALPACA_PAPER=True,
        TAX_RESERVE_FILE=test_tax_file,
        CYCLE_INTERVAL_SECONDS=5,
    )
    
    daemon = TradingDaemon(config=config, dry_run=True)
    
    # Execute a single cycle
    daemon.run_cycle()
    
    assert daemon.cycle_count == 1
    assert test_tax_file.exists()
    assert daemon.tax_engine.current_reserve >= 0.0


def test_daemon_trailing_stop_loss_trigger(tmp_path):
    test_tax_file = tmp_path / "stop_tax_reserve.json"
    config = BotConfig(
        ALPACA_PAPER=True,
        TAX_RESERVE_FILE=test_tax_file,
        TRAILING_STOP_LOSS_PCT=0.05,  # 5% trailing stop
        BUY_TRIGGER_SCORE=0.60,
        SELL_TRIGGER_SCORE=-0.40,
    )
    
    daemon = TradingDaemon(config=config, dry_run=True)
    
    # Simulate an open position that rallied to $70,000 and dropped to $65,000 (-7.1% drawdown)
    daemon.position_qty = 0.05
    daemon.position_entry_price = 60000.0
    daemon.position_peak_price = 70000.0
    
    current_price = 65000.0  # (70000 - 65000) / 70000 = 7.14% > 5% stop
    
    tech, vol, sent, comp, signal_type, exit_reason = daemon.evaluate_signals(current_price)
    
    assert signal_type == "SELL"
    assert exit_reason is not None
    assert "Trailing Stop-Loss Triggered" in exit_reason


def test_daemon_multi_crypto_scanning(tmp_path):
    test_tax_file = tmp_path / "multi_crypto_tax.json"
    basket = ["BTC/USD", "ETH/USD", "SOL/USD", "LINK/USD"]
    config = BotConfig(
        ALPACA_PAPER=True,
        TAX_RESERVE_FILE=test_tax_file,
        TARGET_SYMBOLS=basket,
        WHEEL_ENABLED=False,
    )
    daemon = TradingDaemon(config=config, dry_run=True)
    daemon.run_cycle()

    assert daemon.cycle_count == 1
    # Verify all symbols have been tracked in crypto_positions
    for sym in basket:
        assert sym in daemon.crypto_positions


def test_daemon_multi_crypto_position_isolation(tmp_path):
    test_tax_file = tmp_path / "isolation_tax.json"
    config = BotConfig(
        ALPACA_PAPER=True,
        TAX_RESERVE_FILE=test_tax_file,
        TRAILING_STOP_LOSS_PCT=0.05,
        TARGET_SYMBOLS=["BTC/USD", "LINK/USD"],
        WHEEL_ENABLED=False,
    )
    daemon = TradingDaemon(config=config, dry_run=True)
    # Set LINK/USD into a trailing stop condition
    daemon.crypto_positions["LINK/USD"] = {
        "qty": 50.0,
        "entry_price": 20.0,
        "peak_price": 25.0,
    }
    # BTC/USD has no position
    daemon.crypto_positions["BTC/USD"] = {
        "qty": 0.0,
        "entry_price": None,
        "peak_price": None,
    }

    # Evaluate LINK at 22.0 (drawdown = (25-22)/25 = 12% > 5%)
    _, _, _, _, link_signal, link_reason = daemon.evaluate_signals(22.0, symbol="LINK/USD")
    assert link_signal == "SELL"
    assert "Trailing Stop-Loss Triggered" in (link_reason or "")

    # Evaluate BTC at 65000.0 (BTC has 0 qty, so trailing stop should not trigger)
    _, _, _, _, btc_signal, btc_reason = daemon.evaluate_signals(65000.0, symbol="BTC/USD")
    assert btc_reason != link_reason

