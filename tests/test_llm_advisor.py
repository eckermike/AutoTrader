"""
Tests for LLM Portfolio Advisor and Quantitative Intelligence Engine.
Verifies daily briefing generation, HITL reserve liquidation advising,
fundamental divergence gating, falling knife defense, and fund Q&A.
"""

import json
from pathlib import Path
import pytest
from intelligence.llm_advisor import LLMAdvisor, get_llm_advisor


@pytest.fixture
def mock_advisor():
    return LLMAdvisor(provider="mock")


def test_llm_advisor_singleton():
    adv1 = get_llm_advisor()
    adv2 = get_llm_advisor()
    assert adv1 is adv2


def test_generate_daily_briefing_fallback(mock_advisor, tmp_path, monkeypatch):
    test_briefing_file = tmp_path / "daily_briefing.json"
    import intelligence.llm_advisor as advisor_module
    monkeypatch.setattr(advisor_module, "BRIEFING_FILE", test_briefing_file)

    briefing = mock_advisor.generate_daily_briefing(
        date_str="2026-09-12",
        trades_count=2,
        cash=100000.0,
        tradable_cash=70000.0,
        tax_reserve=30000.0,
        diagnostics={"Wheel": ["AAPL CSP active"], "Pairs": ["XOM/CVX active"]},
    )

    assert "AutoTrader" in briefing
    assert "2 active execution(s)" in briefing
    assert "100,000.00" in briefing
    assert "30,000.00" in briefing

    # Verify file was written
    assert test_briefing_file.exists()
    data = json.loads(test_briefing_file.read_text())
    assert data["date"] == "2026-09-12"
    assert data["trades_count"] == 2
    assert data["tax_reserve"] == 30000.0


def test_generate_daily_briefing_zero_trades(mock_advisor, tmp_path, monkeypatch):
    test_briefing_file = tmp_path / "daily_briefing.json"
    import intelligence.llm_advisor as advisor_module
    monkeypatch.setattr(advisor_module, "BRIEFING_FILE", test_briefing_file)

    briefing = mock_advisor.generate_daily_briefing(
        date_str="2026-09-12",
        trades_count=0,
        cash=50000.0,
        tradable_cash=35000.0,
        tax_reserve=15000.0,
    )
    assert "defensively positioned" in briefing or "0 new orders" in briefing


def test_recommend_liquidation_tranche(mock_advisor):
    # Case 1: Ample SGOV -> Recommends SGOV
    rec = mock_advisor.recommend_liquidation_tranche(
        needed_cash=1000.0,
        target_symbol="XOM",
        opportunity_type="Pairs Trading",
        sgov_balance=20000.0,
        fbnd_balance=20000.0,
    )
    assert rec["recommended_tranche"] == "SGOV"
    assert "SGOV" in rec["reasoning"]

    # Case 2: Depleted SGOV -> Recommends FBND
    rec2 = mock_advisor.recommend_liquidation_tranche(
        needed_cash=5000.0,
        target_symbol="SPY",
        opportunity_type="Dip Buyer",
        sgov_balance=200.0,
        fbnd_balance=20000.0,
    )
    assert rec2["recommended_tranche"] == "FBND"
    assert "FBND" in rec2["reasoning"]


def test_verify_pair_divergence(mock_advisor):
    # Clean divergence -> Approved
    res = mock_advisor.verify_pair_divergence(
        symbol_a="XOM",
        symbol_b="CVX",
        z_score=2.25,
        recent_headlines=["Crude oil inventories rise slightly", "OPEC maintains target production"],
    )
    assert res["safe_to_trade"] is True
    assert res["risk_level"] == "LOW"

    # Hazardous divergence -> Blocked
    res_hazard = mock_advisor.verify_pair_divergence(
        symbol_a="XOM",
        symbol_b="CVX",
        z_score=2.85,
        recent_headlines=["CVX faces SEC fraud investigation and possible bankruptcy restructuring"],
    )
    assert res_hazard["safe_to_trade"] is False
    assert res_hazard["risk_level"] == "HIGH"
    assert "hazard" in res_hazard["reasoning"].lower()


def test_verify_dip_candidate(mock_advisor):
    # Routine oversold dip -> Safe
    res = mock_advisor.verify_dip_candidate(
        symbol="SPY",
        rsi_val=28.5,
        drop_pct=-4.2,
        recent_headlines=["Tech pullback cools broad index"],
    )
    assert res["safe_to_trade"] is True
    assert res["risk_level"] == "LOW"

    # Severe falling knife (> 30% crash) -> Unsafe
    res_knife = mock_advisor.verify_dip_candidate(
        symbol="ABC",
        rsi_val=14.0,
        drop_pct=-35.0,
        recent_headlines=["Accounting irregularities revealed"],
    )
    assert res_knife["safe_to_trade"] is False
    assert res_knife["risk_level"] == "HIGH"


def test_answer_fund_query_scenarios(mock_advisor):
    snapshot = {
        "portfolio": {"cash": 100000.0, "tradable_cash": 65000.0},
        "tax_engine": {
            "tax_reserve": 25000.0,
            "total_tax_allocated": 30000.0,
            "total_tax_credits": 5000.0,
            "net_realized_pnl": 83333.33,
        },
        "spreads": {"active_count": 2, "collateral_locked": 1000.0},
        "tail_hedge": {"has_active_hedge": True, "monthly_spent": 140.0, "monthly_budget_remaining": 10.0},
        "dip_buyer": {"active_positions_count": 1, "capital_deployed_usd": 1000.0, "win_rate_pct": 100.0},
        "macro_rotation": {"regime": "RISK_ON", "active_positions_count": 3, "capital_deployed_usd": 3000.0},
        "pairs_trading": {"active_positions_count": 1, "allocated_capital_usd": 2000.0, "total_realized_pnl": 120.0, "win_rate_pct": 100.0},
    }

    # Tax query
    tax_resp = mock_advisor.answer_fund_query("What is our tax escrow reserve balance?", snapshot)
    assert "25,000.00" in tax_resp
    assert "Tax Escrow" in tax_resp

    # Cash query
    cash_resp = mock_advisor.answer_fund_query("How much tradable cash is available?", snapshot)
    assert "65,000.00" in cash_resp

    # Pairs query
    pairs_resp = mock_advisor.answer_fund_query("Tell me about our pairs trading performance", snapshot)
    assert "**1** active" in pairs_resp
    assert "2,000.00" in pairs_resp

    # Tail hedge query
    hedge_resp = mock_advisor.answer_fund_query("Is our black swan tail hedge active?", snapshot)
    assert "ACTIVE" in hedge_resp
