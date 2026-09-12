"""
Intelligence package for AutoTrader.
Provides LLM-powered portfolio commentary, HITL rebalancing recommendations,
and fundamental divergence safety checks.
"""

from intelligence.llm_advisor import (
    LLMAdvisor,
    get_llm_advisor,
)

__all__ = ["LLMAdvisor", "get_llm_advisor"]
