"""
Centralized configuration for the cryptocurrency paper trading bot.
Uses Pydantic Settings for type validation, environment variable parsing, and risk threshold enforcement.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any, List, Optional, Union
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class BotConfig(BaseSettings):
    """Configuration model for cryptocurrency paper trading bot."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Alpaca Paper Trading Credentials ---
    ALPACA_API_KEY: str = Field(
        default="MOCK_API_KEY",
        description="Alpaca API Key ID (paper trading)",
    )
    ALPACA_SECRET_KEY: str = Field(
        default="MOCK_SECRET_KEY",
        description="Alpaca Secret Key (paper trading)",
    )
    ALPACA_PAPER: bool = Field(
        default=True,
        description="Must be strictly True for paper trading",
    )
    ALPACA_BASE_URL: str = Field(
        default="https://paper-api.alpaca.markets",
        description="Alpaca Paper API Base URL",
    )

    # --- Target Symbol & Daemon Parameters ---
    TARGET_SYMBOL: str = Field(
        default="BTC/USD",
        description="Default cryptocurrency pair for single-asset trading",
    )
    TARGET_SYMBOLS: Union[List[str], str] = Field(
        default=["BTC/USD", "ETH/USD", "SOL/USD", "LINK/USD", "AVAX/USD", "DOGE/USD"],
        description="List of target cryptocurrency pairs for multi-crypto portfolio scanner",
    )
    CYCLE_INTERVAL_SECONDS: int = Field(
        default=60,
        ge=5,
        le=86400,
        description="Scheduled daemon loop interval in seconds",
    )
    ORDER_SIZE_USD: float = Field(
        default=500.0,
        gt=0.0,
        description="Target dollar amount allocated per paper buy order",
    )
    MAX_POSITION_USD: float = Field(
        default=500.0,
        gt=0.0,
        description="Maximum cumulative position notional value in USD per crypto asset",
    )

    # --- Tri-Factor Weights ---
    WEIGHT_TECHNICAL: float = Field(
        default=0.40,
        ge=0.0,
        le=1.0,
        description="Weight assigned to deterministic technical factor",
    )
    WEIGHT_VOLUME: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        description="Weight assigned to volume/regime factor",
    )
    WEIGHT_SENTIMENT: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        description="Weight assigned to NLP sentiment factor",
    )

    # --- Signal Thresholds & Risk Management ---
    BUY_TRIGGER_SCORE: float = Field(
        default=0.60,
        ge=-1.0,
        le=1.0,
        description="Minimum composite score to trigger buy order",
    )
    SELL_TRIGGER_SCORE: float = Field(
        default=-0.40,
        ge=-1.0,
        le=1.0,
        description="Maximum composite score to trigger exit/sell order",
    )
    TRAILING_STOP_LOSS_PCT: float = Field(
        default=0.05,
        gt=0.0,
        lt=1.0,
        description="Trailing stop-loss percentage from peak price (e.g. 0.05 = 5%)",
    )

    # --- Virtual Tax Escrow Engine ---
    TAX_RATE: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        description="Percentage of net realized profit allocated to virtual tax escrow",
    )
    TAX_RESERVE_FILE: Path = Field(
        default=Path("tax_reserve.json"),
        description="Persistent JSON file tracking virtual tax reserve",
    )

    # --- Sentiment Factor Engine ---
    SENTIMENT_PROVIDER: str = Field(
        default="ollama",
        description="Sentiment provider: 'ollama', 'gemini', 'finbert', or 'mock'",
    )
    OLLAMA_BASE_URL: str = Field(
        default="http://localhost:11434",
        description="Base URL for local Ollama HTTP endpoint",
    )
    OLLAMA_MODEL: str = Field(
        default="qwen2.5:7b",
        description="Ollama model name (e.g., qwen2.5:7b or llama3.1:8b)",
    )
    GEMINI_API_KEY: Optional[str] = Field(
        default=None,
        description="Optional Gemini API key if using Gemini provider",
    )

    # --- Notifications & Alerts ---
    ALERT_ENABLED: bool = Field(
        default=True,
        description="Whether to dispatch trade alerts",
    )
    ALERT_RECIPIENT: Optional[str] = Field(
        default="eckermike87@icloud.com",
        description="Apple ID email or phone number for iMessage trade alerts",
    )
    NTFY_TOPIC: Optional[str] = Field(
        default="eckermike87",
        description="ntfy.sh topic for instant iOS PWA and Apple Watch push notifications",
    )
    ALERT_MACOS_BANNER: bool = Field(
        default=True,
        description="Whether to display local macOS desktop notification banners",
    )
    DAILY_RECAP_ENABLED: bool = Field(
        default=True,
        description="Whether to send an automated end-of-day summary notification",
    )
    DAILY_RECAP_HOUR: int = Field(
        default=17,
        ge=0,
        le=23,
        description="Hour of day (0-23, local time) to send daily recap (17 = 5:00 PM)",
    )
    DAILY_RECAP_MINUTE: int = Field(
        default=0,
        ge=0,
        le=59,
        description="Minute of hour (0-59) to send daily recap",
    )
    DAILY_RECAP_ONLY_ZERO_TRADES: bool = Field(
        default=False,
        description="If True, only sends recap if zero trades were executed that day",
    )
    NTFY_ACTION_TOPIC: Optional[str] = Field(
        default="eckermike87-actions",
        description="ntfy.sh topic for receiving two-way action button approvals from iOS",
    )

    # --- Liquidity Reserve & Cash Management ---
    LIQUIDITY_RESERVE_ENABLED: bool = Field(
        default=True,
        description="Whether the autonomous liquidity and cash yield manager is active",
    )
    LIQUIDITY_SYMBOLS: Union[List[str], str] = Field(
        default=["SGOV", "FBND"],
        description="Tickers used for cash-yield parking (e.g. SGOV, FBND)",
    )
    SGOV_ALLOCATION_USD: float = Field(
        default=20000.0,
        ge=0.0,
        description="Target notional allocation into SGOV (ultra-short Treasury)",
    )
    FBND_ALLOCATION_USD: float = Field(
        default=20000.0,
        ge=0.0,
        description="Target notional allocation into FBND (total bond ETF)",
    )
    LIQUIDITY_STATE_FILE: str = Field(
        default="liquidity_state.json",
        description="Persistent state file tracking pending approvals and bond holdings",
    )
    APPROVAL_ON_LIQUIDATION_ONLY: bool = Field(
        default=True,
        description="If True, interactive mobile approval is only required when selling parked funds (SGOV/FBND)",
    )
    APPROVAL_TTL_HOURS: float = Field(
        default=4.0,
        ge=0.25,
        le=24.0,
        description="Maximum approval window in hours before a trade request expires (Layer 1 safety)",
    )
    APPROVAL_MAX_SLIPPAGE_PCT: float = Field(
        default=0.005,
        ge=0.001,
        le=0.05,
        description="Maximum price slippage allowed (0.005 = 0.5%) upon approval before aborting (Layer 2 safety)",
    )



    # --- Option Wheel Strategy (Multi-Asset Portfolio) ---
    WHEEL_ENABLED: bool = Field(
        default=True,
        description="Whether to run the Option Wheel strategy",
    )
    WHEEL_SYMBOL: str = Field(
        default="INTC",
        description="Default underlying equity symbol for single-asset option wheel",
    )
    WHEEL_SYMBOLS: Union[List[str], str] = Field(
        default=["INTC", "F", "SOFI", "HOOD", "PLTR", "XLF"],
        description="List of underlying equity symbols for multi-asset option wheel portfolio",
    )
    WHEEL_TARGET_DTE_MIN: int = Field(
        default=21,
        ge=7,
        le=90,
        description="Minimum days to expiration for wheel contracts",
    )
    WHEEL_TARGET_DTE_MAX: int = Field(
        default=45,
        ge=14,
        le=120,
        description="Maximum days to expiration for wheel contracts",
    )
    WHEEL_TARGET_DELTA: float = Field(
        default=0.25,
        ge=0.10,
        le=0.50,
        description="Target OTM delta for CSP and CC contracts",
    )
    WHEEL_PROFIT_TARGET_PCT: float = Field(
        default=0.50,
        ge=0.20,
        le=0.90,
        description="Profit percentage target to Buy-to-Close early (e.g. 0.50 = 50% max profit)",
    )
    WHEEL_CONTRACTS: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Number of option contracts traded per cycle (1 contract = 100 shares)",
    )
    WHEEL_ORDER_TTL_HOURS: float = Field(
        default=24.0,
        ge=1.0,
        le=72.0,
        description="Maximum hours an unfilled option order can remain pending before being cancelled and re-evaluated",
    )
    WHEEL_TIME_IN_FORCE: str = Field(
        default="DAY",
        description="Time-in-force for option orders ('DAY' or 'GTC')",
    )

    @field_validator("TARGET_SYMBOLS", mode="after")
    @classmethod
    def parse_target_symbols(cls, v: Any) -> List[str]:
        if isinstance(v, str):
            symbols = [s.strip().upper() for s in v.split(",") if s.strip()]
            return symbols if symbols else ["BTC/USD", "ETH/USD", "SOL/USD", "LINK/USD", "AVAX/USD", "DOGE/USD"]
        elif isinstance(v, (list, tuple)):
            return [str(s).strip().upper() for s in v if str(s).strip()]
        return ["BTC/USD", "ETH/USD", "SOL/USD", "LINK/USD", "AVAX/USD", "DOGE/USD"]

    @field_validator("WHEEL_SYMBOLS", mode="after")
    @classmethod
    def parse_wheel_symbols(cls, v: Any) -> List[str]:
        if isinstance(v, str):
            symbols = [s.strip().upper() for s in v.split(",") if s.strip()]
            return symbols if symbols else ["INTC", "F", "SOFI", "HOOD", "PLTR", "XLF"]
        elif isinstance(v, (list, tuple)):
            return [str(s).strip().upper() for s in v if str(s).strip()]
        return ["INTC", "F", "SOFI", "HOOD", "PLTR", "XLF"]

    @field_validator("LIQUIDITY_SYMBOLS", mode="after")
    @classmethod
    def parse_liquidity_symbols(cls, v: Any) -> List[str]:
        if isinstance(v, str):
            symbols = [s.strip().upper() for s in v.split(",") if s.strip()]
            return symbols if symbols else ["SGOV", "FBND"]
        elif isinstance(v, (list, tuple)):
            return [str(s).strip().upper() for s in v if str(s).strip()]
        return ["SGOV", "FBND"]


    # --- Strict Paper Trading Enforcement ---
    @field_validator("ALPACA_PAPER")
    @classmethod
    def enforce_paper_trading_only(cls, v: bool) -> bool:
        if not v:
            raise ValueError(
                "CRITICAL RISK VIOLATION: ALPACA_PAPER must be set to True. "
                "This bot is strictly engineered for Paper Trading."
            )
        return v

    @model_validator(mode="after")
    def validate_weights_and_thresholds(self) -> "BotConfig":
        total_weight = (
            self.WEIGHT_TECHNICAL + self.WEIGHT_VOLUME + self.WEIGHT_SENTIMENT
        )
        if not (0.99 <= total_weight <= 1.01):
            raise ValueError(
                f"Factor weights must sum to 1.0 (got {total_weight:.4f}: "
                f"Technical={self.WEIGHT_TECHNICAL}, Volume={self.WEIGHT_VOLUME}, Sentiment={self.WEIGHT_SENTIMENT})"
            )

        if self.BUY_TRIGGER_SCORE <= self.SELL_TRIGGER_SCORE:
            raise ValueError(
                f"BUY_TRIGGER_SCORE ({self.BUY_TRIGGER_SCORE}) must be strictly greater than "
                f"SELL_TRIGGER_SCORE ({self.SELL_TRIGGER_SCORE})"
            )
        return self


@lru_cache()
def get_config() -> BotConfig:
    """Returns a cached instance of BotConfig loaded from environment."""
    return BotConfig()
