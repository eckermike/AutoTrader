"""
NLP Sentiment Factor Engine.
Provides a modular architecture supporting:
- Ollama local LLM HTTP endpoint (default: qwen2.5:7b or llama3.1:8b) with strict JSON output schema.
- Google Gemini API.
- HuggingFace FinBERT.
- Mock / Synthetic analyzer for offline dry-runs and automated testing.
"""

import abc
import json
import logging
import re
from typing import Any, Dict, Optional
import requests
from pydantic import BaseModel, Field

logger = logging.getLogger("factors.sentiment")


class SentimentResult(BaseModel):
    """Structured sentiment scoring output with strict validation bounds."""

    sentiment_score: float = Field(
        ge=-1.0,
        le=1.0,
        description="Sentiment score: -1.0 (strongly bearish) to +1.0 (strongly bullish)",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Model confidence level: 0.0 to 1.0",
    )
    provider: str
    model: str
    reasoning: Optional[str] = Field(
        default=None,
        description="Brief summary of sentiment justification",
    )


class BaseSentimentAnalyzer(abc.ABC):
    """Abstract Base Class for modular sentiment analyzers."""

    @abc.abstractmethod
    def analyze(self, market_text: str) -> SentimentResult:
        """
        Analyzes market text/headlines and produces a normalized SentimentResult.
        """
        pass


class OllamaSentimentAnalyzer(BaseSentimentAnalyzer):
    """
    Connects to local Ollama HTTP endpoint (http://localhost:11434/api/generate).
    Enforces strict JSON schema: {"sentiment_score": float, "confidence": float}.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5:7b",
        timeout_seconds: float = 10.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    def analyze(self, market_text: str) -> SentimentResult:
        """
        Sends prompt to Ollama with format='json' and parses structured sentiment.
        Gracefully handles offline instances by returning a neutral score.
        """
        endpoint = f"{self.base_url}/api/generate"
        prompt = (
            "You are an expert quantitative crypto market analyst. "
            "Evaluate the following cryptocurrency headlines, news, and market commentary. "
            "Determine the prevailing sentiment and output valid JSON ONLY, strictly conforming "
            "to this JSON schema:\n"
            "{\n"
            '  "sentiment_score": <float between -1.0 (strongly bearish) and 1.0 (strongly bullish)>,\n'
            '  "confidence": <float between 0.0 (no confidence) and 1.0 (absolute certainty)>,\n'
            '  "reasoning": "<1-sentence summary>"\n'
            "}\n\n"
            f"Market Content:\n{market_text}\n"
        )

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.1,
            },
        }

        try:
            response = requests.post(
                endpoint,
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_state = None  # Clear any state
            if response.status_code != 200:
                logger.warning(
                    "Ollama returned HTTP %d: %s. Defaulting to neutral sentiment.",
                    response.status_code,
                    response.text[:100],
                )
                return self._neutral_fallback(f"HTTP {response.status_code}")

            res_json = response.json()
            raw_text = res_json.get("response", "{}")
            parsed = json.loads(raw_text)

            score = float(parsed.get("sentiment_score", 0.0))
            score_clamped = max(-1.0, min(1.0, score))
            confidence = float(parsed.get("confidence", 0.5))
            conf_clamped = max(0.0, min(1.0, confidence))
            reasoning = parsed.get("reasoning", "Parsed from Ollama JSON output")

            return SentimentResult(
                sentiment_score=round(score_clamped, 4),
                confidence=round(conf_clamped, 4),
                provider="ollama",
                model=self.model,
                reasoning=reasoning,
            )

        except requests.exceptions.RequestException as e:
            logger.warning(
                "Ollama endpoint unreachable at %s (%s). Defaulting to neutral sentiment.",
                self.base_url,
                e,
            )
            return self._neutral_fallback(f"Connection failed: {e}")
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(
                "Failed to parse JSON response from Ollama (%s). Defaulting to neutral.",
                e,
            )
            return self._neutral_fallback(f"JSON parse error: {e}")

    def _neutral_fallback(self, reason: str) -> SentimentResult:
        """Returns neutral sentiment score when Ollama is unavailable."""
        return SentimentResult(
            sentiment_score=0.0,
            confidence=0.0,
            provider="ollama",
            model=self.model,
            reasoning=f"Neutral fallback ({reason})",
        )


class GeminiSentimentAnalyzer(BaseSentimentAnalyzer):
    """
    Connects to Google Gemini API via REST endpoint with structured JSON output.
    """

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        self.api_key = api_key
        self.model = model

    def analyze(self, market_text: str) -> SentimentResult:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": (
                                "Evaluate cryptocurrency market sentiment. Output strictly JSON adhering to:\n"
                                '{"sentiment_score": <float -1.0 to 1.0>, "confidence": <float 0.0 to 1.0>, "reasoning": "<str>"}\n\n'
                                f"Text:\n{market_text}"
                            )
                        }
                    ]
                }
            ],
            "generationConfig": {
                "response_mime_type": "application/json",
                "temperature": 0.1,
            },
        }

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=10.0)
            if resp.status_code == 200:
                data = resp.json()
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                parsed = json.loads(text)
                return SentimentResult(
                    sentiment_score=float(parsed.get("sentiment_score", 0.0)),
                    confidence=float(parsed.get("confidence", 0.5)),
                    provider="gemini",
                    model=self.model,
                    reasoning=parsed.get("reasoning", "Gemini analysis"),
                )
        except Exception as e:
            logger.warning("Gemini sentiment analysis failed (%s). Returning neutral.", e)

        return SentimentResult(
            sentiment_score=0.0,
            confidence=0.0,
            provider="gemini",
            model=self.model,
            reasoning="Neutral fallback",
        )


class FinBERTSentimentAnalyzer(BaseSentimentAnalyzer):
    """
    Pluggable FinBERT sentiment analyzer (via local HuggingFace transformers pipeline).
    """

    def __init__(self, model_name: str = "ProsusAI/finbert"):
        self.model_name = model_name
        self._pipeline = None

    def _get_pipeline(self):
        if self._pipeline is None:
            try:
                from transformers import pipeline
                self._pipeline = pipeline("sentiment-analysis", model=self.model_name)
            except ImportError:
                logger.warning(
                    "transformers library not installed. Install with `pip install transformers torch` to use FinBERT."
                )
                self._pipeline = False
        return self._pipeline

    def analyze(self, market_text: str) -> SentimentResult:
        pipe = self._get_pipeline()
        if not pipe:
            return SentimentResult(
                sentiment_score=0.0,
                confidence=0.0,
                provider="finbert",
                model=self.model_name,
                reasoning="FinBERT dependencies not installed; neutral fallback.",
            )

        try:
            res = pipe(market_text[:512])[0]
            label = res.get("label", "neutral").lower()
            conf = float(res.get("score", 0.5))

            if label == "positive":
                score = conf
            elif label == "negative":
                score = -conf
            else:
                score = 0.0

            return SentimentResult(
                sentiment_score=round(score, 4),
                confidence=round(conf, 4),
                provider="finbert",
                model=self.model_name,
                reasoning=f"FinBERT label: {label}",
            )
        except Exception as e:
            logger.warning("FinBERT evaluation error (%s).", e)
            return SentimentResult(
                sentiment_score=0.0,
                confidence=0.0,
                provider="finbert",
                model=self.model_name,
                reasoning=f"Error: {e}",
            )


class MockSentimentAnalyzer(BaseSentimentAnalyzer):
    """
    Mock sentiment analyzer for deterministic testing, dry-runs, and offline execution.
    """

    def __init__(self, default_score: float = 0.65, default_confidence: float = 0.85):
        self.default_score = default_score
        self.default_confidence = default_confidence

    def analyze(self, market_text: str) -> SentimentResult:
        # Rule-based heuristics if text contains bullish/bearish keywords
        lower = market_text.lower()
        score = self.default_score
        if "bear" in lower or "crash" in lower or "plunge" in lower or "dump" in lower:
            score = -0.75
        elif "bull" in lower or "surge" in lower or "rally" in lower or "breakout" in lower:
            score = 0.75

        return SentimentResult(
            sentiment_score=score,
            confidence=self.default_confidence,
            provider="mock",
            model="synthetic-v1",
            reasoning=f"Keyword heuristic evaluation (score={score:.2f})",
        )


def create_sentiment_analyzer(
    provider: str = "ollama",
    ollama_base_url: str = "http://localhost:11434",
    ollama_model: str = "qwen2.5:7b",
    gemini_api_key: Optional[str] = None,
) -> BaseSentimentAnalyzer:
    """
    Factory function to instantiate the configured sentiment analyzer.
    """
    provider_clean = provider.strip().lower()

    if provider_clean == "ollama":
        return OllamaSentimentAnalyzer(
            base_url=ollama_base_url,
            model=ollama_model,
        )
    elif provider_clean == "gemini":
        if not gemini_api_key:
            logger.warning("GEMINI_API_KEY not set; defaulting to MockSentimentAnalyzer.")
            return MockSentimentAnalyzer()
        return GeminiSentimentAnalyzer(api_key=gemini_api_key)
    elif provider_clean == "finbert":
        return FinBERTSentimentAnalyzer()
    elif provider_clean == "mock":
        return MockSentimentAnalyzer()
    else:
        logger.warning(
            "Unknown sentiment provider '%s'. Defaulting to Ollama analyzer.", provider
        )
        return OllamaSentimentAnalyzer(
            base_url=ollama_base_url,
            model=ollama_model,
        )
