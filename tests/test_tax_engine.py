"""
Unit tests for Virtual Tax Escrow Engine.
Tests tax allocation on gains, credits on losses (floored at $0.00),
hard capital gate enforcement, and atomic JSON persistence.
"""

import json
from pathlib import Path
import pytest
from tax_engine import TaxEngine, InsufficientTradableCashError


@pytest.fixture
def temp_tax_file(tmp_path: Path) -> Path:
    return tmp_path / "test_tax_reserve.json"


def test_initial_state_empty(temp_tax_file: Path):
    engine = TaxEngine(filepath=temp_tax_file, tax_rate=0.30)
    assert engine.current_reserve == 0.0
    assert engine.calculate_tradable_cash(1000.0) == 1000.0
    assert temp_tax_file.exists()


def test_profitable_trade_allocation(temp_tax_file: Path):
    engine = TaxEngine(filepath=temp_tax_file, tax_rate=0.30)
    
    # Buy 1 BTC at $60,000, sell at $65,000 -> Gross Profit: $5,000
    record = engine.record_closed_trade(
        symbol="BTC/USD",
        side="SELL",
        qty=1.0,
        entry_price=60000.0,
        exit_price=65000.0,
        fee=0.0,
    )
    
    assert record.gross_pnl == 5000.0
    assert record.tax_allocated == 1500.0  # 30% of $5,000
    assert record.tax_credit == 0.0
    assert engine.current_reserve == 1500.0
    
    # Capital gate check: Account has $10,000 cash -> Tradable cash is $8,500
    assert engine.calculate_tradable_cash(10000.0) == 8500.0


def test_losing_trade_credit_and_floor(temp_tax_file: Path):
    engine = TaxEngine(filepath=temp_tax_file, tax_rate=0.30)
    
    # 1. First profitable trade: +$1,000 -> Tax reserve becomes $300
    engine.record_closed_trade(
        symbol="BTC/USD",
        side="SELL",
        qty=1.0,
        entry_price=10000.0,
        exit_price=11000.0,
    )
    assert engine.current_reserve == 300.0
    
    # 2. Losing trade: -$500 -> Tax credit: $150 -> New reserve: $150
    record2 = engine.record_closed_trade(
        symbol="BTC/USD",
        side="SELL",
        qty=1.0,
        entry_price=11000.0,
        exit_price=10500.0,
    )
    assert record2.gross_pnl == -500.0
    assert record2.tax_credit == 150.0
    assert engine.current_reserve == 150.0
    
    # 3. Big losing trade: -$2,000 -> Potential credit: $600 -> Floor reserve at $0.00
    record3 = engine.record_closed_trade(
        symbol="BTC/USD",
        side="SELL",
        qty=1.0,
        entry_price=10500.0,
        exit_price=8500.0,
    )
    assert record3.gross_pnl == -2000.0
    assert record3.tax_credit == 150.0  # Only $150 was available in reserve to credit
    assert engine.current_reserve == 0.0  # Strict floor at $0.00


def test_hard_capital_gate_rejection(temp_tax_file: Path):
    engine = TaxEngine(filepath=temp_tax_file, tax_rate=0.30)
    
    # Reserve $4,000 in escrow
    engine.record_closed_trade(
        symbol="BTC/USD",
        side="SELL",
        qty=1.0,
        entry_price=50000.0,
        exit_price=63333.33,
    )
    assert engine.current_reserve == 4000.0
    
    total_cash = 5000.0
    tradable_cash = engine.calculate_tradable_cash(total_cash)
    assert tradable_cash == 1000.0
    
    # Order within tradable cash ($800 <= $1000) passes
    validated = engine.validate_order_budget(alpaca_cash_balance=total_cash, order_cost=800.0)
    assert validated == 1000.0
    
    # Order exceeding tradable cash ($1200 > $1000) fails with exception
    with pytest.raises(InsufficientTradableCashError) as exc_info:
        engine.validate_order_budget(alpaca_cash_balance=total_cash, order_cost=1200.0)
    
    assert "Requested $1200.00 exceeds Tradable Cash $1000.00" in str(exc_info.value)


def test_json_state_persistence_and_reload(temp_tax_file: Path):
    engine1 = TaxEngine(filepath=temp_tax_file, tax_rate=0.30)
    engine1.record_closed_trade(
        symbol="BTC/USD",
        side="SELL",
        qty=0.5,
        entry_price=50000.0,
        exit_price=60000.0,
    )
    expected_reserve = engine1.current_reserve
    assert expected_reserve == 1500.0
    
    # Instantiate a second engine pointing to the same file
    engine2 = TaxEngine(filepath=temp_tax_file, tax_rate=0.30)
    assert engine2.current_reserve == expected_reserve
    assert engine2.state.trade_count == 1
    assert len(engine2.state.trade_history) == 1
    assert engine2.state.trade_history[0].symbol == "BTC/USD"
