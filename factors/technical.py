"""
Deterministic Technical Factor Engine.
Computes RSI(14) and 50/200-period Simple Moving Averages using pandas-ta.
Normalizes indicators to produce a deterministic float score between -1.0 (strongly bearish)
and +1.0 (strongly bullish).
"""

import logging
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
import pandas_ta as ta
from pydantic import BaseModel, Field

logger = logging.getLogger("factors.technical")


class TechnicalResult(BaseModel):
    """Normalized technical factor scoring output with detailed indicator values."""

    score: float = Field(
        ge=-1.0,
        le=1.0,
        description="Deterministic score between -1.0 (bearish) and 1.0 (bullish)",
    )
    current_price: float
    rsi: float
    sma_50: float
    sma_200: Optional[float] = None
    golden_cross: bool = False
    details: Dict[str, Any] = Field(default_factory=dict)


class TechnicalFactor:
    """
    Computes deterministic technical indicators on OHLCV market bars.
    Requires pandas-ta for RSI(14), SMA(50), and SMA(200).
    """

    def __init__(
        self,
        rsi_length: int = 14,
        sma_fast_length: int = 50,
        sma_slow_length: int = 200,
    ):
        self.rsi_length = rsi_length
        self.sma_fast_length = sma_fast_length
        self.sma_slow_length = sma_slow_length

    def evaluate(self, df: pd.DataFrame) -> TechnicalResult:
        """
        Calculates indicators and produces a normalized score in [-1.0, 1.0].
        
        Parameters:
            df: DataFrame containing at least a 'close' column (case-insensitive).
                Ideally contains at least 200 rows for complete SMA(200) calculation.
        """
        if df.empty:
            raise ValueError("Input DataFrame is empty; cannot evaluate technical indicators.")

        # Standardize column names to lowercase
        df_clean = df.copy()
        df_clean.columns = [c.lower() for c in df_clean.columns]

        if "close" not in df_clean.columns:
            raise ValueError("DataFrame must contain a 'close' column.")

        close_series = df_clean["close"].astype(float)
        current_price = float(close_series.iloc[-1])

        # 1. Calculate RSI(14)
        rsi_series = ta.rsi(close_series, length=self.rsi_length)
        if rsi_series is None or rsi_series.dropna().empty:
            logger.warning("RSI calculation yielded insufficient data; defaulting to neutral 50.0")
            current_rsi = 50.0
        else:
            current_rsi = float(rsi_series.iloc[-1])
            if np.isnan(current_rsi):
                current_rsi = 50.0

        # 2. Calculate SMA(50)
        sma_fast_series = ta.sma(close_series, length=self.sma_fast_length)
        if sma_fast_series is None or sma_fast_series.dropna().empty:
            sma_fast = current_price
        else:
            sma_fast = float(sma_fast_series.iloc[-1])
            if np.isnan(sma_fast):
                sma_fast = current_price

        # 3. Calculate SMA(200)
        sma_slow_series = ta.sma(close_series, length=self.sma_slow_length)
        if sma_slow_series is None or sma_slow_series.dropna().empty:
            sma_slow = sma_fast
        else:
            sma_slow = float(sma_slow_series.iloc[-1])
            if np.isnan(sma_slow):
                sma_slow = sma_fast

        # --- Normalized Scoring Engine ---
        # Component A: RSI Score [-1.0, 1.0]
        # Neutral RSI is 50.0. Scale deviation: RSI 75 -> +1.0, RSI 25 -> -1.0.
        rsi_delta = current_rsi - 50.0
        rsi_score = float(np.clip(rsi_delta / 25.0, -1.0, 1.0))

        # Component B: Fast SMA Trend Score [-1.0, 1.0]
        # Evaluates % distance from current price to SMA(50)
        pct_from_sma_fast = (current_price - sma_fast) / max(sma_fast, 1e-6)
        sma_fast_score = float(np.tanh(pct_from_sma_fast * 20.0))

        # Component C: Regime & Trend Cross Score [-1.0, 1.0]
        # Golden Cross (SMA 50 > SMA 200) vs Death Cross (SMA 50 < SMA 200)
        golden_cross = sma_fast >= sma_slow
        pct_fast_vs_slow = (sma_fast - sma_slow) / max(sma_slow, 1e-6)
        pct_price_vs_slow = (current_price - sma_slow) / max(sma_slow, 1e-6)
        sma_slow_score = float(np.tanh((pct_price_vs_slow + pct_fast_vs_slow) * 10.0))

        # Composite Technical Score:
        # 40% RSI Momentum + 30% Fast SMA + 30% Trend/Cross
        composite_technical = (
            0.40 * rsi_score + 0.30 * sma_fast_score + 0.30 * sma_slow_score
        )
        final_score = float(np.clip(composite_technical, -1.0, 1.0))

        details = {
            "rsi_score": round(rsi_score, 4),
            "sma_fast_score": round(sma_fast_score, 4),
            "sma_slow_score": round(sma_slow_score, 4),
            "pct_from_sma50": round(pct_from_sma_fast * 100, 2),
            "pct_from_sma200": round(pct_price_vs_slow * 100, 2),
        }

        logger.debug(
            "Technical Evaluation: Price=%.2f | RSI=%.2f (score=%.2f) | SMA50=%.2f | SMA200=%.2f | Score=%.3f",
            current_price,
            current_rsi,
            rsi_score,
            sma_fast,
            sma_slow,
            final_score,
        )

        return TechnicalResult(
            score=round(final_score, 4),
            current_price=round(current_price, 2),
            rsi=round(current_rsi, 2),
            sma_50=round(sma_fast, 2),
            sma_200=round(sma_slow, 2),
            golden_cross=golden_cross,
            details=details,
        )
