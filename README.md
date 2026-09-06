# AutoTrader: Modular Cryptocurrency & Options Paper Trading Bot (macOS Apple Silicon)

A production-ready, modular paper trading system engineered for macOS (Apple Silicon, `arm64`). Integrates Alpaca Paper Trading for spot crypto and equity options execution, a Tri-Factor Decision Engine (Technical + Volume/Regime + Ollama NLP Sentiment), an automated **Option Wheel Strategy on INTC**, a programmatic Virtual Tax Escrow Engine, and instant multi-channel push alerting via **ntfy.sh** and Apple iMessage.

---

## Architecture Overview

```
AutoTrader/
├── .env.example               # Template for Alpaca keys & local configuration
├── .gitignore                 # Standard Python & macOS ignore patterns
├── requirements.txt           # Pinned dependencies for Apple Silicon
├── config.py                  # Pydantic Settings for centralized config & thresholds
├── tax_engine.py              # Programmatic virtual tax reserve tracking & hard capital gate
├── notifier.py                # Multi-channel alerting (ntfy.sh web push & Apple iMessage)
├── factors/
│   ├── __init__.py
│   ├── technical.py           # pandas-ta RSI(14) and 50/200 SMA deterministic scoring
│   ├── volume_regime.py       # 24h volume delta & 7-day volatility baseline metrics
│   └── sentiment.py           # Ollama JSON NLP pipeline with Gemini/FinBERT/Mock interfaces
├── execution/
│   ├── __init__.py
│   └── alpaca_client.py       # alpaca-py wrapper for spot crypto paper order placement
├── options/
│   ├── __init__.py
│   ├── options_client.py      # alpaca-py options broker client (chains, quotes, orders)
│   └── wheel_engine.py        # 4-phase Option Wheel state machine (CSP -> CC lifecycle)
├── tests/
│   ├── __init__.py
│   ├── test_tax_engine.py     # Unit tests for tax reserve, gains, losses, and tradable cash
│   ├── test_factors.py        # Unit tests for technical, volume regime, and sentiment factors
│   ├── test_execution.py      # Unit tests for Alpaca client and paper order execution
│   ├── test_simulation.py     # End-to-end integration test of trade & tax lifecycle
│   ├── test_notifier.py       # Unit tests for push notifications and formatting
│   └── test_wheel.py          # Unit tests for Option Wheel state machine & 50% profit target
└── main.py                    # Entry point & scheduled daemon with clean signal shutdown
```

---

## Core Component Specifications

### 1. Broker Integration (`execution/alpaca_client.py` & `options/options_client.py`)
- **Strict Paper Trading**: Initialized strictly with `paper=True`. Live trading is rejected at the configuration and client levels.
- **Spot Crypto**: Defaults to `BTC/USD` with fractional order sizing (supports both USD `notional` and crypto `qty`).
- **Options Trading**: Alpaca Level 3 Options Paper approval. Supports option chain filtering (`expiration_date_gte`, `expiration_date_lte`, `type=ContractType.PUT/CALL`), quote retrieval, and `PositionIntent` order routing.
- **Mock Fallback**: Automatic mock simulation when API keys are omitted or set to dry-run.

### 2. Tri-Factor Decision Engine
- **Technical Factor (`factors/technical.py`)**:
  - Computes RSI(14) and 50/200-period Simple Moving Averages using `pandas-ta`.
  - Analyzes momentum and moving average crossovers (Golden Cross / Death Cross).
  - Normalizes deterministically into a float score between `-1.0` (strongly bearish) and `+1.0` (strongly bullish).
- **Volume / Regime Factor (`factors/volume_regime.py`)**:
  - Analyzes 24-hour volume delta ($V_{\text{current}} - V_{\text{prev}}$) and return velocity.
  - Compares recent return volatility against a 7-day baseline ($\sigma_{24h} / \sigma_{7d}$).
  - Volume surges on rallies yield bullish confirmation (+); surges on sell-offs yield bearish distribution (-).
  - Returns a normalized float score between `-1.0` and `+1.0`.
- **Sentiment Factor (`factors/sentiment.py`)**:
  - Modular NLP pipeline defaulting to a local Ollama HTTP endpoint (`http://localhost:11434/api/generate`) with `qwen2.5:7b` or `llama3.1:8b`.
  - Enforces strict JSON schema output: `{"sentiment_score": float, "confidence": float, "reasoning": str}`.
  - Pluggable interfaces for Google Gemini API (`GeminiSentimentAnalyzer`), HuggingFace FinBERT (`FinBERTSentimentAnalyzer`), and offline testing (`MockSentimentAnalyzer`).
  - Graceful fallback: If Ollama is offline or unreachable, returns neutral sentiment (`0.0`) without interrupting daemon operations.

### 3. Option Wheel Strategy (`options/wheel_engine.py`)
Executes an automated Options Wheel income strategy on **INTC**:
1. **Phase 1 (Cash-Secured Put)**: If 0 shares held and no active put, scans for OTM Puts 21–45 DTE ($\approx 0.25$ Delta) and sells to open (`SELL_TO_OPEN`).
2. **Phase 2 (Monitoring Put / Profit-Taking)**: Monitors short put. If mark price drops to $\le 50\%$ of entry premium, submits early buy-to-close (`BUY_TO_CLOSE`) order to lock in 50% max profit. If assigned at expiration, takes delivery of 100 shares.
3. **Phase 3 (Covered Call)**: If $\ge 100$ shares held, sells OTM Covered Call (`SELL_TO_OPEN`) with strike $\ge$ cost basis and 21–45 DTE.
4. **Phase 4 (Monitoring Call)**: Closes call at 50% profit target or allows assignment/expiration to complete the Wheel.

### 4. Real-Time Push Notifications (`notifier.py`)
- **ntfy.sh Push Notifications**: Real-time push alerts dispatched instantly to iOS, iPadOS, macOS, and Android via `ntfy.sh/eckermike87`.
  - Supports PWA web push on iOS devices without requiring the full App Store app.
  - Dispatches audible alerts, vibrations, and badges for buy orders, sell orders, trailing stop triggers, CSP/CC openings, and 50% profit target exits.
- **Apple iMessage**: Fallback notification pipeline leveraging native macOS AppleScript targeting configured Apple IDs / phone numbers.

### 5. Virtual Tax Escrow Engine (`tax_engine.py`)
- **State Persistence**: Maintains a local, persistent JSON state file (`tax_reserve.json`) written atomically to prevent file corruption.
- **Profitable Trades & Option Premiums**: Allocates 30% of net realized crypto profits and option premiums collected to `tax_reserve`.
- **Losing Trades & Buy-Back Costs**: Deducts tax credits against the reserve, strictly floored at `$0.00`.
- **Hard Capital Gate**:
  $$\text{Tradable Cash} = \max(0.0, \text{Alpaca Cash Balance} - \text{Tax Reserve})$$
  - Order submissions and options collateral requirements are validated before broker dispatch. If requirement exceeds Tradable Cash, execution is halted.

---

## Quickstart Guide

### 1. Environment Setup
The project includes a dedicated Python 3.12 virtual environment tailored for Apple Silicon (`arm64`):
```bash
# Activate virtual environment
source .venv/bin/activate
```

### 2. Configure API Keys
Edit `/Users/mikeeckerle/AutoTrader/.env`:
```env
ALPACA_API_KEY=your_alpaca_paper_api_key
ALPACA_SECRET_KEY=your_alpaca_paper_secret_key
ALPACA_PAPER=True
TARGET_SYMBOL=BTC/USD
WHEEL_SYMBOL=INTC
NTFY_TOPIC=eckermike87
```

### 3. Running the Bot

**Single Evaluation Cycle**:
```bash
.venv/bin/python main.py --once
```

**Continuous Live Paper Trading Daemon**:
```bash
.venv/bin/python main.py --interval 60
```

To stop the daemon cleanly, send `SIGINT` (`Ctrl+C`) or `SIGTERM`. The daemon completes the active evaluation cycle, flushes all state to `tax_reserve.json`, and outputs a comprehensive financial audit summary.

---

## Running the Automated Test Suite

Run the full pytest suite (30 unit and integration tests):
```bash
.venv/bin/pytest tests/ -v
```
All tests verify:
- Strict paper trading enforcement (`paper=True`).
- Option Wheel 4-phase state transitions and 50% profit target early exit.
- Hard capital gate collateral rejection.
- Technical factor normalization ($[-1.0, 1.0]$).
- Volume regime expansion and volatility ratio metrics.
- Ollama JSON response schema validation and offline fallback.
- Tax escrow 30% allocation on gains and credit on losses (floored at $0.00).
- ntfy.sh and iMessage notification dispatch and payload sanitization.
