"""
Unit tests for Tri-Factor Decision Engine:
- Technical Factor (RSI 14, SMA 50/200 via pandas-ta)
- Volume/Regime Factor (24h volume delta, 7d volatility baseline)
- Sentiment Factor (Ollama JSON schema, fallback, and mock analyzers)
"""

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from factors.technical import TechnicalFactor, TechnicalResult
from factors.volume_regime import VolumeRegimeFactor, VolumeRegimeResult
from factors.sentiment import (
    SentimentResult,
    OllamaSentimentAnalyzer,
    MockSentimentAnalyzer,
    create_sentiment_analyzer,
)


@pytest.fixture
def sample_bars() -> pd.DataFrame:
    """Generates 250 bars of synthetic OHLCV data."""
    np.random.seed(123)
    n = 250
    returns = np.random.normal(0.0005, 0.01, n)
    prices = 50000.0 * np.exp(np.cumsum(returns))
    volumes = np.random.uniform(100.0, 500.0, n)
    
    return pd.DataFrame(
        {
            "open": prices,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "close": prices,
            "volume": volumes,
        }
    )


def test_technical_factor_bounds_and_structure(sample_bars: pd.DataFrame):
    factor = TechnicalFactor()
    result = factor.evaluate(sample_bars)
    
    assert isinstance(result, TechnicalResult)
    assert -1.0 <= result.score <= 1.0
    assert 0.0 <= result.rsi <= 100.0
    assert result.sma_50 > 0
    assert result.sma_200 is not None and result.sma_200 > 0
    assert isinstance(result.golden_cross, bool)


def test_technical_factor_bullish_trend():
    # Construct a strong uptrend
    n = 250
    prices = np.linspace(30000, 60000, n)
    df = pd.DataFrame({
        "open": prices,
        "high": prices * 1.01,
        "low": prices * 0.99,
        "close": prices,
        "volume": np.full(n, 200.0),
    })
    
    factor = TechnicalFactor()
    result = factor.evaluate(df)
    assert result.score > 0.40  # Should be strongly bullish
    assert result.golden_cross is True
    assert result.rsi > 50.0


def test_technical_factor_bearish_trend():
    # Construct a strong downtrend
    n = 250
    prices = np.linspace(60000, 30000, n)
    df = pd.DataFrame({
        "open": prices,
        "high": prices * 1.01,
        "low": prices * 0.99,
        "close": prices,
        "volume": np.full(n, 200.0),
    })
    
    factor = TechnicalFactor()
    result = factor.evaluate(df)
    assert result.score < -0.40  # Should be strongly bearish
    assert result.golden_cross is False
    assert result.rsi < 50.0


def test_volume_regime_bounds(sample_bars: pd.DataFrame):
    factor = VolumeRegimeFactor(bars_per_day=24, baseline_days=7)
    result = factor.evaluate(sample_bars)
    
    assert isinstance(result, VolumeRegimeResult)
    assert -1.0 <= result.score <= 1.0
    assert result.volume_24h > 0
    assert result.volatility_24h >= 0
    assert result.volatility_7d >= 0


def test_volume_regime_bullish_surge():
    # 250 bars: first 226 bars low volume & flat, last 24 bars 10x volume & 20% price rally
    n = 250
    prices = np.full(n, 50000.0)
    prices[-24:] = np.linspace(50000.0, 60000.0, 24)
    
    volumes = np.full(n, 100.0)
    volumes[-24:] = 1000.0  # 10x surge
    
    df = pd.DataFrame({
        "open": prices,
        "high": prices * 1.01,
        "low": prices * 0.99,
        "close": prices,
        "volume": volumes,
    })
    
    factor = VolumeRegimeFactor(bars_per_day=24, baseline_days=7)
    result = factor.evaluate(df)
    assert result.score > 0.30  # High volume up-move is bullish
    assert result.volume_delta_pct > 0


def test_volume_regime_bearish_surge():
    # 250 bars: last 24 bars 10x volume & 20% price collapse
    n = 250
    prices = np.full(n, 50000.0)
    prices[-24:] = np.linspace(50000.0, 40000.0, 24)
    
    volumes = np.full(n, 100.0)
    volumes[-24:] = 1000.0  # Panic selling volume
    
    df = pd.DataFrame({
        "open": prices,
        "high": prices * 1.01,
        "low": prices * 0.99,
        "close": prices,
        "volume": volumes,
    })
    
    factor = VolumeRegimeFactor(bars_per_day=24, baseline_days=7)
    result = factor.evaluate(df)
    assert result.score < -0.30  # High volume sell-off is bearish


def test_sentiment_result_schema_validation():
    # Valid model
    res = SentimentResult(
        sentiment_score=0.75,
        confidence=0.90,
        provider="ollama",
        model="qwen2.5:7b",
        reasoning="Strong positive inflows",
    )
    assert res.sentiment_score == 0.75
    assert res.confidence == 0.90
    
    # Out of range sentiment_score should fail validation
    with pytest.raises(ValidationError):
        SentimentResult(
            sentiment_score=1.5,
            confidence=0.90,
            provider="ollama",
            model="qwen2.5:7b",
        )


def test_mock_sentiment_analyzer():
    analyzer = MockSentimentAnalyzer()
    
    bull_res = analyzer.analyze("Massive institutional bull rally breaks all resistance!")
    assert bull_res.sentiment_score > 0.50
    
    bear_res = analyzer.analyze("Severe market crash and panic selling triggers dump!")
    assert bear_res.sentiment_score < -0.50


def test_ollama_offline_fallback():
    # Point to an unused local port to simulate offline Ollama
    analyzer = OllamaSentimentAnalyzer(base_url="http://localhost:59999", timeout_seconds=1.0)
    res = analyzer.analyze("Some bitcoin news")
    
    # Must return neutral score without crashing
    assert res.sentiment_score == 0.0
    assert res.confidence == 0.0
    assert "Neutral fallback" in (res.reasoning or "")


def test_sentiment_factory():
    analyzer = create_sentiment_analyzer(provider="mock")
    assert isinstance(analyzer, MockSentimentAnalyzer)
    
    ollama_analyzer = create_sentiment_analyzer(provider="ollama")
    assert isinstance(ollama_analyzer, OllamaSentimentAnalyzer)
