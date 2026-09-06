"""
Unit tests for INTC Option Wheel Strategy Engine.
Tests state machine transitions (CSP -> Monitoring -> CC -> Monitoring),
contract selection filters, 50% profit target early exits,
hard capital gate enforcement, and tax escrow integration.
"""

from pathlib import Path
import pytest
from config import BotConfig
from notifier import TradeNotifier
from options.options_client import (
    AlpacaOptionsClient,
    OptionPositionInfo,
    StockPositionInfo,
)
from options.wheel_engine import WheelEngine, WheelPortfolioManager, WheelState
from tax_engine import TaxEngine


@pytest.fixture
def test_setup(tmp_path: Path):
    tax_file = tmp_path / "wheel_tax_reserve.json"
    config = BotConfig(
        ALPACA_PAPER=True,
        TAX_RESERVE_FILE=tax_file,
        WHEEL_ENABLED=True,
        WHEEL_SYMBOL="INTC",
        WHEEL_TARGET_DTE_MIN=21,
        WHEEL_TARGET_DTE_MAX=45,
        WHEEL_PROFIT_TARGET_PCT=0.50,
        WHEEL_CONTRACTS=1,
    )
    tax_engine = TaxEngine(filepath=tax_file, tax_rate=0.30)
    options_client = AlpacaOptionsClient(
        api_key="MOCK_KEY",
        secret_key="MOCK_SECRET",
        paper=True,
        mock_mode=True,
    )
    notifier = TradeNotifier(recipient=None, enabled=False, macos_banner=False)
    engine = WheelEngine(
        config=config,
        options_client=options_client,
        tax_engine=tax_engine,
        notifier=notifier,
    )
    return engine, options_client, tax_engine


def test_initial_state_is_cash_secured_put(test_setup):
    engine, client, tax = test_setup
    state = engine.evaluate_state()
    assert state == WheelState.CASH_SECURED_PUT


def test_put_contract_selection(test_setup):
    engine, client, tax = test_setup
    stock_price = client.get_stock_price("INTC")  # 21.50 in mock mode
    contract = engine.select_put_contract(stock_price)
    
    assert contract is not None
    assert contract.contract_type == "put"
    assert contract.strike_price < stock_price  # Must be OTM
    assert 21 <= contract.days_to_expiration <= 45


def test_csp_order_execution_and_tax_escrow(test_setup):
    engine, client, tax = test_setup
    
    # Step: Execute CSP
    status = engine.step(total_cash=100000.0)
    
    assert status.state == WheelState.CASH_SECURED_PUT
    assert engine.active_contract_symbol is not None
    assert engine.active_contract_premium is not None
    
    # Verify tax escrow captured 30% of option premium
    total_premium = engine.active_contract_premium * 100 * engine.config.WHEEL_CONTRACTS
    expected_tax = total_premium * 0.30
    assert tax.current_reserve == pytest.approx(expected_tax, rel=1e-2)
    assert tax.state.trade_count == 1
    
    # Next evaluation state must be MONITORING_PUT
    next_state = engine.evaluate_state()
    assert next_state == WheelState.MONITORING_PUT


def test_csp_50_percent_profit_target_early_exit(test_setup):
    engine, client, tax = test_setup
    
    # 1. Open Put contract
    engine.step(total_cash=100000.0)
    put_symbol = engine.active_contract_symbol
    initial_premium = engine.active_contract_premium
    
    # 2. Simulate option price decay: option drops to 40% of original price (60% profit > 50% target)
    pos = client._mock_option_positions[put_symbol]
    pos.current_price = initial_premium * 0.40
    
    # 3. Step: should trigger BUY_TO_CLOSE
    status = engine.step(total_cash=100000.0)
    assert status.state == WheelState.MONITORING_PUT
    
    # Position should now be closed in broker
    assert put_symbol not in client._mock_option_positions
    assert engine.active_contract_symbol is None
    
    # Next state returns to CASH_SECURED_PUT
    assert engine.evaluate_state() == WheelState.CASH_SECURED_PUT


def test_covered_call_selection_and_execution(test_setup):
    engine, client, tax = test_setup
    
    # Simulate owning 100 shares of INTC at cost basis $20.00
    client._mock_stock_positions["INTC"] = StockPositionInfo(
        symbol="INTC",
        qty=100,
        avg_entry_price=20.0,
        current_price=21.50,
        market_value=2150.0,
    )
    engine.cost_basis = 20.0
    
    # State should be COVERED_CALL
    assert engine.evaluate_state() == WheelState.COVERED_CALL
    
    # Step: should sell covered call
    status = engine.step(total_cash=100000.0)
    assert status.state == WheelState.COVERED_CALL
    assert engine.active_contract_symbol is not None
    assert "C" in engine.active_contract_symbol
    
    # Next state is MONITORING_CALL
    assert engine.evaluate_state() == WheelState.MONITORING_CALL


def test_capital_gate_rejects_csp_if_insufficient_cash(test_setup):
    engine, client, tax = test_setup
    
    # Artificial small cash balance ($500) where collateral ($2,000) exceeds cash
    status = engine.step(total_cash=500.0)
    
    # Order should be rejected by capital gate, so no option opened
    assert engine.active_contract_symbol is None
    assert len(client._mock_option_positions) == 0


def test_multi_asset_portfolio_manager_initialization(tmp_path: Path):
    tax_file = tmp_path / "portfolio_tax_reserve.json"
    config = BotConfig(
        ALPACA_PAPER=True,
        TAX_RESERVE_FILE=tax_file,
        WHEEL_ENABLED=True,
        WHEEL_SYMBOLS=["INTC", "F", "SOFI", "HOOD", "PLTR", "XLF"],
    )
    tax_engine = TaxEngine(filepath=tax_file, tax_rate=0.30)
    options_client = AlpacaOptionsClient(api_key="MOCK", secret_key="MOCK", paper=True, mock_mode=True)
    notifier = TradeNotifier(recipient=None, enabled=False, macos_banner=False)

    portfolio = WheelPortfolioManager(
        config=config,
        options_client=options_client,
        tax_engine=tax_engine,
        notifier=notifier,
    )

    assert set(portfolio.symbols) == {"INTC", "F", "SOFI", "HOOD", "PLTR", "XLF"}
    assert len(portfolio.engines) == 6
    for sym in portfolio.symbols:
        assert portfolio.engines[sym].symbol == sym
        assert portfolio.engines[sym].evaluate_state() == WheelState.CASH_SECURED_PUT


def test_multi_asset_portfolio_step_all(tmp_path: Path):
    tax_file = tmp_path / "portfolio_tax_reserve_step.json"
    config = BotConfig(
        ALPACA_PAPER=True,
        TAX_RESERVE_FILE=tax_file,
        WHEEL_ENABLED=True,
        WHEEL_SYMBOLS=["INTC", "F", "SOFI"],
    )
    tax_engine = TaxEngine(filepath=tax_file, tax_rate=0.30)
    options_client = AlpacaOptionsClient(api_key="MOCK", secret_key="MOCK", paper=True, mock_mode=True)
    notifier = TradeNotifier(recipient=None, enabled=False, macos_banner=False)

    portfolio = WheelPortfolioManager(
        config=config,
        options_client=options_client,
        tax_engine=tax_engine,
        notifier=notifier,
    )

    # Step across portfolio with $100k
    statuses = portfolio.step(total_cash=100000.0)

    assert len(statuses) == 3
    for sym in ["INTC", "F", "SOFI"]:
        assert sym in statuses
        assert statuses[sym].state == WheelState.CASH_SECURED_PUT
        assert portfolio.engines[sym].active_contract_symbol is not None
        assert sym in portfolio.engines[sym].active_contract_symbol


def test_multi_asset_portfolio_capital_gating(tmp_path: Path):
    tax_file = tmp_path / "portfolio_tax_gate.json"
    config = BotConfig(
        ALPACA_PAPER=True,
        TAX_RESERVE_FILE=tax_file,
        WHEEL_ENABLED=True,
        WHEEL_SYMBOLS=["INTC", "F", "SOFI"],
    )
    tax_engine = TaxEngine(filepath=tax_file, tax_rate=0.30)
    options_client = AlpacaOptionsClient(api_key="MOCK", secret_key="MOCK", paper=True, mock_mode=True)
    notifier = TradeNotifier(recipient=None, enabled=False, macos_banner=False)

    portfolio = WheelPortfolioManager(
        config=config,
        options_client=options_client,
        tax_engine=tax_engine,
        notifier=notifier,
    )

    # With only $2,200 total cash:
    # INTC collateral is ~20 * 100 = $2,000 -> succeeds
    # Remaining cash drops to ~$200 -> F and SOFI collateral will exceed remaining and be rejected
    statuses = portfolio.step(total_cash=2200.0)

    assert statuses["INTC"].collateral_locked > 0
    assert portfolio.engines["INTC"].active_contract_symbol is not None
    # F and SOFI should have been rejected due to lack of remaining cash
    assert portfolio.engines["F"].active_contract_symbol is None
    assert portfolio.engines["SOFI"].active_contract_symbol is None
