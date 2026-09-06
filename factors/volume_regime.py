"""
Volume & Volatility Regime Factor Engine.
Calculates 24-hour volume delta and evaluates recent return volatility against a 7-day baseline.
Produces a normalized float score between -1.0 (strongly bearish) and +1.0 (strongly bullish).
"""

import logging
from typing import Any, Dict
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger("factors.volume_regime")


class VolumeRegimeResult(BaseModel):
    """Normalized volume/regime factor scoring output."""

    score: float = Field(
        ge=-1.0,
        le=1.0,
        description="Normalized score between -1.0 (bearish volume/regime) and 1.0 (bullish volume/regime)",
    )
    volume_24h: float
    volume_prev_24h: float
    volume_delta_pct: float
    volatility_24h: float
    volatility_7d: float
    volatility_ratio: float
    details: Dict[str, Any] = Field(default_factory=dict)


class VolumeRegimeFactor:
    """
    Evaluates 24-hour volume expansion/contraction and 7-day volatility baseline.
    Standardized for hourly cryptocurrency bars (24 bars/day, 168 bars/week).
    """

    def __init__(self, bars_per_day: int = 24, baseline_days: int = 7):
        self.bars_per_day = bars_per_day
        self.baseline_bars = bars_per_day * baseline_days

    def evaluate(self, df: pd.DataFrame) -> VolumeRegimeResult:
        """
        Computes 24h volume delta and volatility regime metrics.
        
        Parameters:
            df: DataFrame containing 'close' and 'volume' columns.
        """
        if df.empty:
            raise ValueError("Input DataFrame is empty; cannot evaluate volume regime.")

        # Standardize column names
        df_clean = df.copy()
        df_clean.columns = [c.lower() for c in df_clean.columns]

        if "volume" not in df_clean.columns or "close" not in df_clean.columns:
            raise ValueError("DataFrame must contain both 'close' and 'volume' columns.")

        close = df_clean["close"].astype(float)
        volume = df_clean["volume"].astype(float)
        total_bars = len(df_clean)

        # Handle bar count adaptability
        bpd = min(self.bars_per_day, max(1, total_bars // 2))
        
        # 1. 24-Hour Volume Delta
        vol_curr_24h = float(volume.iloc[-bpd:].sum())
        if total_bars >= 2 * bpd:
            vol_prev_24h = float(volume.iloc[-2 * bpd : -bpd].sum())
        else:
            vol_prev_24h = vol_curr_24h

        vol_prev_safe = max(vol_prev_24h, 1.0)
        volume_delta_pct = (vol_curr_24h - vol_prev_24h) / vol_prev_safe

        # 2. Volatility Analysis (Log returns)
        log_returns = np.log(close / close.shift(1)).dropna()
        
        if len(log_returns) >= bpd:
            volatility_24h = float(log_returns.iloc[-bpd:].std())
        else:
            volatility_24h = float(log_returns.std()) if not log_returns.empty else 0.01

        baseline_window = min(len(log_returns), self.baseline_bars)
        if baseline_window > 0:
            volatility_7d = float(log_returns.iloc[-baseline_window:].std())
        else:
            volatility_7d = volatility_24h

        # Fallback for near-zero volatility
        volatility_7d_safe = max(volatility_7d, 1e-5)
        volatility_ratio = volatility_24h / volatility_7d_safe

        # 3. 24-Hour Price Movement Context
        current_price = float(close.iloc[-1])
        price_24h_ago = float(close.iloc[-bpd]) if total_bars >= bpd else float(close.iloc[0])
        price_delta_pct = (current_price - price_24h_ago) / max(price_24h_ago, 1e-6)

        # --- Normalized Scoring Engine ---
        # A. Directional Volume Flow:
        # High volume on up move = bullish confirmation (+)
        # High volume on down move = bearish distribution (-)
        # Low volume on flat move = neutral
        signed_volume_intensity = float(np.tanh(volume_delta_pct * 1.5)) * np.sign(price_delta_pct)
        
        # B. Price Trend Velocity
        price_velocity_score = float(np.tanh(price_delta_pct * 25.0))

        # C. Volatility Multiplier
        # Compression (ratio < 0.8) dampens extremes
        # Expansion (ratio > 1.2) amplifies signal
        vol_amplifier = float(np.clip(volatility_ratio, 0.6, 1.8))

        raw_score = (0.55 * signed_volume_intensity + 0.45 * price_velocity_score) * (vol_amplifier / 1.2)
        final_score = float(np.clip(raw_score, -1.0, 1.0))

        details = {
            "volume_delta_pct": round(volume_delta_pct * 100, 2),
            "price_delta_pct": round(price_delta_pct * 100, 2),
            "signed_volume_intensity": round(signed_volume_intensity, 4),
            "price_velocity_score": round(price_velocity_score, 4),
            "volatility_ratio": round(volatility_ratio, 3),
        }

        logger.debug(
            "Volume/Regime Eval: Vol24h=%.1f | VolDelta=%.1f%% | PriceDelta=%.2f%% | VolRatio=%.2f | Score=%.3f",
            vol_curr_24h,
            volume_delta_pct * 100,
            price_delta_pct * 100,
            volatility_ratio,
            final_score,
        )

        return VolumeRegimeResult(
            score=round(final_score, 4),
            volume_24h=round(vol_curr_24h, 2),
            volume_prev_24h=round(vol_prev_24h, 2),
            volume_delta_pct=round(volume_delta_pct, 4),
            volatility_24h=round(volatility_24h, 6),
            volatility_7d=round(volatility_7d, 6),
            volatility_ratio=round(volatility_ratio, 4),
            details=details,
        )
