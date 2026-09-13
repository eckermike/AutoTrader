"""
LLM Portfolio Advisor & Quantitative Intelligence Engine.
Provides modular integration with:
- Local Ollama (qwen2.5:7b, llama3.1:8b)
- Google Gemini API
- Robust Rule-Based Deterministic Fallback for offline execution & testing
"""

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests

logger = logging.getLogger("intelligence.llm_advisor")

BASE_DIR = Path(__file__).resolve().parent.parent
BRIEFING_FILE = BASE_DIR / "daily_briefing.json"


class LLMAdvisor:
    """
    Central LLM Intelligence Engine supporting:
    1. Daily 5:00 PM Wall Street PM Executive Briefing.
    2. Smart HITL Reserve Liquidation Advisor (SGOV vs FBND).
    3. Pairs Divergence Fundamental Sanity Gate.
    4. Dip Buyer Falling Knife Guard.
    5. Natural Language Fund Q&A (Ask AutoTrader AI).
    """

    def __init__(
        self,
        provider: str = "ollama",
        ollama_base_url: str = "http://localhost:11434",
        ollama_model: str = "qwen2.5:7b",
        gemini_api_key: Optional[str] = None,
        timeout_seconds: float = 8.0,
    ):
        self.provider = os.getenv("LLM_PROVIDER", provider).strip().lower()
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", ollama_base_url).rstrip("/")
        self.ollama_model = os.getenv("OLLAMA_MODEL", ollama_model)
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", gemini_api_key)
        self.timeout_seconds = timeout_seconds

    def _call_llm_raw(self, prompt: str, system_instruction: Optional[str] = None, json_format: bool = False) -> Optional[str]:
        """Calls the configured LLM provider and returns the raw string response."""
        if self.provider == "mock":
            return None

        # Try Ollama first
        if self.provider == "ollama":
            endpoint = f"{self.ollama_base_url}/api/generate"
            payload: Dict[str, Any] = {
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2},
            }
            if system_instruction:
                payload["system"] = system_instruction
            if json_format:
                payload["format"] = "json"

            try:
                resp = requests.post(endpoint, json=payload, timeout=self.timeout_seconds)
                if resp.status_code == 200:
                    data = resp.json()
                    return data.get("response", "").strip()
                logger.warning("Ollama returned status %d. Falling back.", resp.status_code)
            except Exception as e:
                logger.debug("Ollama request failed: %s. Falling back to rule-based logic.", e)

        # Try Gemini if API key is provided
        if (self.provider == "gemini" or (self.provider == "ollama" and self.gemini_api_key)) and self.gemini_api_key:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={self.gemini_api_key}"
            headers = {"Content-Type": "application/json"}
            payload = {
                "contents": [{"parts": [{"text": f"{system_instruction}\n\n{prompt}" if system_instruction else prompt}]}],
                "generationConfig": {
                    "temperature": 0.2,
                    "response_mime_type": "application/json" if json_format else "text/plain",
                },
            }
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout_seconds)
                if resp.status_code == 200:
                    data = resp.json()
                    parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])
                    return parts[0].get("text", "").strip()
            except Exception as e:
                logger.debug("Gemini request failed: %s. Falling back.", e)

        return None

    # =========================================================================
    # 1. Daily Executive Briefing
    # =========================================================================
    def generate_daily_briefing(
        self,
        date_str: str,
        trades_count: int,
        cash: float,
        tradable_cash: float,
        tax_reserve: float,
        diagnostics: Optional[Dict[str, List[str]]] = None,
    ) -> str:
        """
        Generates a concise 2-paragraph Wall Street style Executive Summary for the 5:00 PM recap.
        Saves result to daily_briefing.json.
        """
        diag_str = ""
        if diagnostics:
            for strat, lines in diagnostics.items():
                if lines:
                    diag_str += f"\n- {strat}: " + "; ".join(lines[:2])

        prompt = (
            f"You are the Lead Portfolio Manager at AutoTrader Capital Partners. Write a crisp, 2-paragraph "
            f"Executive Briefing for our daily market recap on {date_str}.\n\n"
            f"Key metrics:\n"
            f"- Trades executed today: {trades_count}\n"
            f"- Total Cash: ${cash:,.2f} | Tradable Cash: ${tradable_cash:,.2f}\n"
            f"- 30% Tax Escrow Reserved: ${tax_reserve:,.2f}\n"
            f"Strategy Statuses:{diag_str or ' All strategies nominal'}\n\n"
            f"Tone: Institutional, disciplined, risk-conscious hedge fund memo. Highlight capital preservation, "
            f"tax discipline, and active risk posture. Return plain text only (no markdown headings)."
        )

        response = self._call_llm_raw(prompt, system_instruction="You are an elite quantitative hedge fund manager.")

        if not response:
            # Deterministic Fallback Summary
            if trades_count > 0:
                p1 = (
                    f"AutoTrader closed the session with {trades_count} active execution(s) while maintaining disciplined "
                    f"risk across all 5 strategies. Total fund liquidity stands at ${cash:,.2f}, with ${tradable_cash:,.2f} "
                    f"in active tradable cash ready for systematic deployment."
                )
            else:
                p1 = (
                    f"AutoTrader remained defensively positioned today with 0 new orders triggered, adhering strictly to "
                    f"selective entry thresholds across credit spreads, tail hedges, dip buyer, macro rotation, and pairs trading. "
                    f"Portfolio liquidity remains robust at ${cash:,.2f}."
                )
            p2 = (
                f"The 30% Automated Tax Escrow is fully capitalized at ${tax_reserve:,.2f}, sequestering tax liabilities "
                f"from tradable assets. Tail-risk hedges and collateral gates remain actively monitored for the next trading cycle."
            )
            response = f"{p1}\n\n{p2}"

        # Clean up formatting
        cleaned_briefing = response.strip()

        # Save to disk
        try:
            briefing_payload = {
                "date": date_str,
                "generated_at": datetime.now().isoformat(),
                "provider": self.provider,
                "trades_count": trades_count,
                "cash": round(cash, 2),
                "tradable_cash": round(tradable_cash, 2),
                "tax_reserve": round(tax_reserve, 2),
                "summary": cleaned_briefing,
            }
            with open(BRIEFING_FILE, "w", encoding="utf-8") as f:
                json.dump(briefing_payload, f, indent=2)
        except Exception as e:
            logger.warning("Could not persist daily_briefing.json: %s", e)

        return cleaned_briefing

    # =========================================================================
    # 2. Smart HITL Reserve Liquidation Advisor (SGOV vs FBND)
    # =========================================================================
    def recommend_liquidation_tranche(
        self,
        needed_cash: float,
        target_symbol: str,
        opportunity_type: str,
        sgov_balance: float = 20000.0,
        fbnd_balance: float = 20000.0,
    ) -> Dict[str, Any]:
        """
        Advises whether to liquidate SGOV (ultra-short treasury) or FBND (core total bond).
        Returns a structured recommendation dictionary.
        """
        prompt = (
            f"We need to liquidate ${needed_cash:,.2f} of bond reserves to fund a high-conviction {opportunity_type} "
            f"trade on {target_symbol}. Our current reserves are: SGOV (0-3mo Treasuries) = ${sgov_balance:,.2f}, "
            f"FBND (Total Bond Market duration) = ${fbnd_balance:,.2f}.\n"
            f"Decide which tranche to sell and provide 1 concise sentence explaining why from a fixed-income duration and yield perspective.\n"
            f"Respond with JSON ONLY adhering to: {{\"recommended_tranche\": \"SGOV\"|\"FBND\", \"reasoning\": \"...\"}}"
        )

        raw = self._call_llm_raw(prompt, json_format=True)
        if raw:
            try:
                parsed = json.loads(raw)
                tranche = parsed.get("recommended_tranche", "SGOV").upper()
                if tranche not in ("SGOV", "FBND"):
                    tranche = "SGOV"
                return {
                    "recommended_tranche": tranche,
                    "reasoning": parsed.get("reasoning", "Optimal liquidity selection."),
                    "provider": self.provider,
                }
            except Exception:
                pass

        # Deterministic Heuristic:
        # Prefer SGOV for smaller/routine liquidity needs to preserve FBND's duration yield;
        # if SGOV balance is depleted below needed cash, choose FBND.
        if sgov_balance >= needed_cash:
            return {
                "recommended_tranche": "SGOV",
                "reasoning": (
                    "Liquidate ultra-short cash (SGOV) to lock in principal while preserving FBND's "
                    "intermediate duration and regular coupon yield."
                ),
                "provider": "heuristic",
            }
        else:
            return {
                "recommended_tranche": "FBND",
                "reasoning": (
                    "Liquidate core bond tranche (FBND) because SGOV balance is insufficient "
                    "for the required trade allocation."
                ),
                "provider": "heuristic",
            }

    # =========================================================================
    # 3. Fundamental Divergence & Falling Knife Guards
    # =========================================================================
    def verify_pair_divergence(
        self,
        symbol_a: str,
        symbol_b: str,
        z_score: float,
        recent_headlines: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Examines whether a pair divergence (|z| >= 2.0) is a mean-reverting statistical
        anomaly or a catastrophic structural rupture.
        """
        headlines_text = "\n".join(f"- {h}" for h in (recent_headlines or [])) or "No material negative news detected."
        prompt = (
            f"Analyze a statistical pairs trading divergence between {symbol_a} and {symbol_b}.\n"
            f"Current rolling Z-score: {z_score:.2f} (Entry threshold |z| >= 2.0).\n"
            f"Headlines:\n{headlines_text}\n\n"
            f"Determine if this divergence is: 1) A routine statistical dislocation safe to arbitrage, or "
            f"2) A catastrophic structural breakdown (fraud, bankruptcy, terminal event) that should NOT be traded.\n"
            f"Output JSON ONLY: {{\"safe_to_trade\": true|false, \"risk_level\": \"LOW\"|\"MEDIUM\"|\"HIGH\", \"reasoning\": \"...\"}}"
        )

        raw = self._call_llm_raw(prompt, json_format=True)
        if raw:
            try:
                parsed = json.loads(raw)
                return {
                    "safe_to_trade": bool(parsed.get("safe_to_trade", True)),
                    "risk_level": str(parsed.get("risk_level", "LOW")).upper(),
                    "reasoning": str(parsed.get("reasoning", "Validated divergence.")),
                    "provider": self.provider,
                }
            except Exception:
                pass

        # Deterministic Heuristic:
        # If headlines contain extreme insolvency/fraud keywords, flag unsafe; otherwise approve.
        hazard_keywords = ["bankruptcy", "insolvency", "sec fraud", "indictment", "chapter 11", "delisting"]
        combined_text = " ".join(recent_headlines or []).lower()
        has_hazard = any(k in combined_text for k in hazard_keywords)

        if has_hazard:
            return {
                "safe_to_trade": False,
                "risk_level": "HIGH",
                "reasoning": "Severe structural hazard detected in headline disclosures. Statistical arbitrage blocked.",
                "provider": "heuristic",
            }

        return {
            "safe_to_trade": True,
            "risk_level": "LOW",
            "reasoning": f"No structural impairment detected. Z-score {z_score:.2f} reflects standard mean-reversion opportunity.",
            "provider": "heuristic",
        }

    def verify_dip_candidate(
        self,
        symbol: str,
        rsi_val: float,
        drop_pct: float,
        recent_headlines: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Examines whether an oversold dip (RSI <= 30) is a high-probability bounce or a falling knife.
        """
        prompt = (
            f"Assess an equity dip buy candidate for {symbol}: RSI(14)={rsi_val:.1f}, 5-day drop={drop_pct:.1f}%.\n"
            f"Headlines:\n" + ("\n".join(recent_headlines or ["No major news"])) + "\n"
            f"Is this a viable mean-reversion bounce or a toxic falling knife?\n"
            f"Output JSON: {{\"safe_to_trade\": true|false, \"risk_level\": \"LOW\"|\"HIGH\", \"reasoning\": \"...\"}}"
        )

        raw = self._call_llm_raw(prompt, json_format=True)
        if raw:
            try:
                parsed = json.loads(raw)
                return {
                    "safe_to_trade": bool(parsed.get("safe_to_trade", True)),
                    "risk_level": str(parsed.get("risk_level", "LOW")).upper(),
                    "reasoning": str(parsed.get("reasoning", "Dip candidate evaluated.")),
                    "provider": self.provider,
                }
            except Exception:
                pass

        # Heuristic fallback
        if drop_pct < -30.0:  # > 30% single-week crash is high risk
            return {
                "safe_to_trade": False,
                "risk_level": "HIGH",
                "reasoning": f"Severe sudden drawdown ({drop_pct:.1f}%) exceeds mean-reversion safety parameters.",
                "provider": "heuristic",
            }

        return {
            "safe_to_trade": True,
            "risk_level": "LOW",
            "reasoning": f"RSI {rsi_val:.1f} indicates classic oversold condition with intact trend structure.",
            "provider": "heuristic",
        }

    # =========================================================================
    # 4. Natural Language "Chat with your Fund" Q&A
    # =========================================================================
    def answer_fund_query(self, user_query: str, fund_snapshot: Dict[str, Any]) -> str:
        """
        Answers natural language investor questions regarding fund health, tax escrow, and strategy states.
        """
        tax_data = fund_snapshot.get("tax_engine", {})
        portfolio = fund_snapshot.get("portfolio", {})
        spreads = fund_snapshot.get("spreads", {})
        hedge = fund_snapshot.get("tail_hedge", {})
        dip = fund_snapshot.get("dip_buyer", {})
        macro = fund_snapshot.get("macro_rotation", {})
        pairs = fund_snapshot.get("pairs_trading", {})

        context = (
            f"AutoTrader Capital Fund Snapshot:\n"
            f"- Total Cash: ${portfolio.get('cash', 0.0):,.2f}\n"
            f"- Tradable Cash: ${portfolio.get('tradable_cash', 0.0):,.2f}\n"
            f"- 30% Tax Escrow Balance: ${tax_data.get('tax_reserve', 0.0):,.2f}\n"
            f"- Net Realized P&L: ${tax_data.get('net_realized_pnl', 0.0):,.2f}\n"
            f"- Active Strategies:\n"
            f"  * Option Spreads: {spreads.get('active_count', 0)} active, ${spreads.get('collateral_locked', 0.0):,.2f} collateral\n"
            f"  * Black Swan Tail Hedge: active={hedge.get('has_active_hedge', False)}, budget=${hedge.get('monthly_budget', 150.0):,.2f}\n"
            f"  * Dip Buyer: {dip.get('active_positions_count', 0)} active, deployed=${dip.get('capital_deployed_usd', 0.0):,.2f}\n"
            f"  * Macro Rotation: regime={macro.get('regime', 'STANDBY')}, {macro.get('active_positions_count', 0)} active\n"
            f"  * Pairs Trading: {pairs.get('active_positions_count', 0)} active pairs, allocated=${pairs.get('allocated_capital_usd', 0.0):,.2f}\n"
        )

        prompt = (
            f"Fund Snapshot Context:\n{context}\n\n"
            f"User Question: {user_query}\n\n"
            f"Guidelines:\n"
            f"- If the question is about fund performance, cash, tax, or trading strategies, answer directly and accurately using the figures from Context Data. Use bold for key numbers.\n"
            f"- If the question is general knowledge, banter, or off-topic, answer directly, accurately, and naturally in first person. Do NOT include robotic disclaimers about context data or mention that the question is outside fund scope.\n"
            f"- Keep answers concise, helpful, and under 3 paragraphs."
        )

        system_instruction = (
            "You are AutoTrader AI, an intelligent quantitative co-pilot and trading assistant for AutoTrader Capital Partners. "
            "Speak directly in a confident, friendly, and professional first-person voice ('I', 'we'). Never use robotic meta-commentary."
        )

        response = self._call_llm_raw(prompt, system_instruction=system_instruction)
        if response:
            return response.strip()

        # Deterministic Q&A heuristics for common questions when offline:
        q_lower = user_query.lower()
        if "tax" in q_lower or "escrow" in q_lower:
            return (
                f"🏛️ **Tax Escrow Status**: We currently have **${tax_data.get('tax_reserve', 0.0):,.2f}** safely sequestered "
                f"in our 30% Tax Escrow Reserve. Total tax allocated to date is **${tax_data.get('total_tax_allocated', 0.0):,.2f}** "
                f"against **${tax_data.get('total_tax_credits', 0.0):,.2f}** in loss credits, yielding a net realized P&L of "
                f"**${tax_data.get('net_realized_pnl', 0.0):,.2f}**."
            )
        elif "cash" in q_lower or "balance" in q_lower or "liquidity" in q_lower:
            return (
                f"💰 **Liquidity Overview**: Total cash is **${portfolio.get('cash', 0.0):,.2f}**, of which "
                f"**${portfolio.get('tradable_cash', 0.0):,.2f}** is active tradable cash after deducting tax reserves and collateral."
            )
        elif "pair" in q_lower or "pairs" in q_lower:
            return (
                f"⚖️ **Pairs Trading Overview**: Currently **{pairs.get('active_positions_count', 0)}** active market-neutral pairs "
                f"deployed with **${pairs.get('allocated_capital_usd', 0.0):,.2f}** allocated out of **${pairs.get('max_capital_usd', 5000.0):,.2f}** "
                f"capital ceiling. Realized P&L is **${pairs.get('total_realized_pnl', 0.0):,.2f}** with a win rate of **{pairs.get('win_rate_pct', 0.0):.1f}%**."
            )
        elif "hedge" in q_lower or "black swan" in q_lower or "tail" in q_lower:
            return (
                f"🦅 **Tail Hedge Overview**: Tail risk hedge is **{'ACTIVE' if hedge.get('has_active_hedge') else 'STANDBY'}**. "
                f"Monthly budget spent is **${hedge.get('monthly_spent', 0.0):,.2f}** with **${hedge.get('monthly_budget_remaining', 150.0):,.2f}** "
                f"remaining for crash protection."
            )
        elif "macro" in q_lower or "sector" in q_lower:
            return (
                f"🧭 **Macro Rotation Overview**: Current regime is **{macro.get('regime', 'STANDBY')}** with "
                f"**{macro.get('active_positions_count', 0)}** sector positions deployed (${macro.get('capital_deployed_usd', 0.0):,.2f})."
            )
        elif "dip" in q_lower:
            return (
                f"🎯 **Dip Buyer Overview**: Currently **{dip.get('active_positions_count', 0)}** oversold dip positions "
                f"active with **${dip.get('capital_deployed_usd', 0.0):,.2f}** deployed. Win rate is **{dip.get('win_rate_pct', 0.0):.1f}%**."
            )
        else:
            return (
                f"📊 **AutoTrader Portfolio Status**: Total cash is **${portfolio.get('cash', 0.0):,.2f}** "
                f"(Tradable: **${portfolio.get('tradable_cash', 0.0):,.2f}**), with **${tax_data.get('tax_reserve', 0.0):,.2f}** "
                f"in tax escrow. We have **{spreads.get('active_count', 0)}** option spreads, "
                f"**{dip.get('active_positions_count', 0)}** dip positions, "
                f"**{macro.get('active_positions_count', 0)}** macro sectors, and "
                f"**{pairs.get('active_positions_count', 0)}** statistical pairs actively monitored."
            )


# Global singleton instance
_GLOBAL_LLM_ADVISOR: Optional[LLMAdvisor] = None


def get_llm_advisor() -> LLMAdvisor:
    """Returns the singleton LLMAdvisor instance."""
    global _GLOBAL_LLM_ADVISOR
    if _GLOBAL_LLM_ADVISOR is None:
        _GLOBAL_LLM_ADVISOR = LLMAdvisor()
    return _GLOBAL_LLM_ADVISOR
