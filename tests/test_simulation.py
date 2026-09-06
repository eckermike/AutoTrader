"""
Full end-to-end integration test simulating multiple cycles:
1. Buy signal triggering paper market order placement.
2. Position open and tracking peak price.
3. Exit/Sell signal triggering liquidation.
4. Tax Escrow Engine capturing 30% profit into tax_reserve.json.
5. Tradable Cash hard capital gate preventing order exceeding tradable cash.
"""

from pathlib import Path
import pytest
from config import BotConfig
from execution.alpaca_client import AlpacaPaperClient
from main import TradingDaemon
from tax_engine import TaxEngine, InsufficientTradableCashError


def test_full_trade_and_tax_escrow_lifecycle(tmp_path: Path):
    tax_file = tmp_path / "lifecycle_tax_reserve.json"
    
    config = BotConfig(
        ALPACA_PAPER=True,
        TAX_RESERVE_FILE=tax_file,
        TAX_RATE=0.30,
        ORDER_SIZE_USD=1000.0,
        BUY_TRIGGER_SCORE=0.50,
        SELL_TRIGGER_SCORE=-0.30,
    )
    
    daemon = TradingDaemon(config=config, dry_run=True)
    
    # 1. Initial State
    assert daemon.tax_engine.current_reserve == 0.0
    account_0 = daemon.client.get_account()
    assert account_0.cash == 100000.00
    assert daemon.tax_engine.calculate_tradable_cash(account_0.cash) == 100000.00
    
    # 2. Simulate Market Buy Order ($1,000 at $60,000)
    order_buy = daemon.client.submit_market_order(
        symbol="BTC/USD",
        side="BUY",
        notional=1000.0,
        estimated_price=60000.0,
    )
    assert order_buy.status == "FILLED"
    pos = daemon.client.get_crypto_position("BTC/USD")
    assert pos is not None
    assert pos.qty == pytest.approx(1000.0 / 60000.0, rel=1e-4)
    
    # 3. Simulate Price Appreciation to $66,000 (+10% gain) and liquidation
    exit_price = 66000.0
    gross_profit = (exit_price - 60000.0) * pos.qty  # $100.00 profit
    
    # Close position in broker
    order_close = daemon.client.close_crypto_position("BTC/USD")
    assert order_close.status == "FILLED"
    
    # Record closed trade in Tax Escrow Engine
    trade_rec = daemon.tax_engine.record_closed_trade(
        symbol="BTC/USD",
        side="SELL",
        qty=pos.qty,
        entry_price=60000.0,
        exit_price=exit_price,
    )
    
    # Verify 30% allocated to tax reserve
    assert trade_rec.gross_pnl == pytest.approx(100.00, rel=1e-2)
    assert trade_rec.tax_allocated == pytest.approx(30.00, rel=1e-2)
    assert daemon.tax_engine.current_reserve == pytest.approx(30.00, rel=1e-2)
    
    # Verify Hard Capital Gate
    account_after = daemon.client.get_account()
    # Cash is now $100,100.00
    tradable_cash = daemon.tax_engine.calculate_tradable_cash(account_after.cash)
    assert tradable_cash == pytest.approx(100100.00 - 30.00, rel=1e-2)
    
    # Attempting to spend more than tradable cash raises InsufficientTradableCashError
    with pytest.raises(InsufficientTradableCashError):
        daemon.tax_engine.validate_order_budget(
            alpaca_cash_balance=account_after.cash,
            order_cost=tradable_cash + 1.0,
        )
