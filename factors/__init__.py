"""
Tri-Factor Decision Engine package:
- Technical Factor (RSI 14, SMA 50/200 via pandas-ta)
- Volume/Regime Factor (24h volume delta, 7d volatility baseline)
- Sentiment Factor (Ollama local LLM NLP pipeline with Gemini/FinBERT/Mock interfaces)
"""

from factors.technical import TechnicalFactor, TechnicalResult
from factors.volume_regime import VolumeRegimeFactor, VolumeRegimeResult
from factors.sentiment import (
    BaseSentimentAnalyzer,
    SentimentResult,
    OllamaSentimentAnalyzer,
    GeminiSentimentAnalyzer,
    FinBERTSentimentAnalyzer,
    MockSentimentAnalyzer,
    create_sentiment_analyzer,
)

__all__ = [
    "TechnicalFactor",
    "TechnicalResult",
    "VolumeRegimeFactor",
    "VolumeRegimeResult",
    "BaseSentimentAnalyzer",
    "SentimentResult",
    "OllamaSentimentAnalyzer",
    "GeminiSentimentAnalyzer",
    "FinBERTSentimentAnalyzer",
    "MockSentimentAnalyzer",
    "create_sentiment_analyzer",
]
