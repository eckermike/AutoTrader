#!/usr/bin/env python3
"""
AutoTrader Investor Dashboard Server
Serves the interactive dashboard.html and real-time JSON fund status API.
"""

import os
import re
import json
import time
import logging
from datetime import datetime, timezone
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn

from config import get_config
from options.options_client import AlpacaOptionsClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("dashboard.server")

BASE_DIR = Path(__file__).resolve().parent
DASHBOARD_HTML = BASE_DIR / "dashboard.html"
TAX_RESERVE_FILE = BASE_DIR / "tax_reserve.json"
LIQUIDITY_STATE_FILE = BASE_DIR / "liquidity_state.json"

_alpaca_client = None

def get_alpaca_client():
    global _alpaca_client
    if _alpaca_client is None:
        try:
            cfg = get_config()
            _alpaca_client = AlpacaOptionsClient(
                api_key=cfg.ALPACA_API_KEY,
                secret_key=cfg.ALPACA_SECRET_KEY,
                paper=cfg.ALPACA_PAPER,
                base_url=cfg.ALPACA_BASE_URL,
                mock_mode=False,
            )
        except Exception as e:
            logger.warning("Could not initialize AlpacaOptionsClient: %s", e)
            return None
    return _alpaca_client

_last_alpaca_fetch = 0.0
_cached_alpaca_telemetry = None

def fetch_live_alpaca_telemetry():
    global _last_alpaca_fetch, _cached_alpaca_telemetry
    now = time.time()
    # 2.5 second cache TTL to avoid hitting Alpaca rate limits on rapid UI polls
    if _cached_alpaca_telemetry is not None and (now - _last_alpaca_fetch) < 2.5:
        return _cached_alpaca_telemetry

    client = get_alpaca_client()
    if client is None or not getattr(client, "trading_client", None):
        return None

    try:
        acc = client.trading_client.get_account()
        positions = client.trading_client.get_all_positions()
        try:
            open_orders = client.trading_client.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=100))
        except Exception:
            open_orders = []

        telemetry = {
            "account": {
                "cash": float(acc.cash),
                "portfolio_value": float(acc.portfolio_value),
                "buying_power": float(acc.buying_power),
                "options_buying_power": float(getattr(acc, "options_buying_power", acc.buying_power)),
                "long_market_value": float(acc.long_market_value),
                "short_market_value": float(acc.short_market_value),
            },
            "positions": positions,
            "open_orders": open_orders,
            "timestamp": now,
        }
        _cached_alpaca_telemetry = telemetry
        _last_alpaca_fetch = now
        return telemetry
    except Exception as e:
        logger.debug("Failed to fetch live Alpaca telemetry: %s", e)
        return None
WHEEL_STATE_FILE = BASE_DIR / "wheel_state.json"
SPREAD_STATE_FILE = BASE_DIR / "spreads_state.json"
HEDGE_STATE_FILE = BASE_DIR / "tail_hedge_state.json"
DIP_STATE_FILE = BASE_DIR / "dip_buyer_state.json"
MACRO_STATE_FILE = BASE_DIR / "macro_rotation_state.json"
PAIRS_STATE_FILE = BASE_DIR / "pairs_trading_state.json"
BRIEFING_FILE = BASE_DIR / "daily_briefing.json"

AUTOTRADER_LOG_FILE = BASE_DIR / "autotrader.log"


def parse_latest_daemon_state():
    matrix = {}
    portfolio = {}
    if not AUTOTRADER_LOG_FILE.exists():
        return list(matrix.values()), portfolio

    try:
        with open(AUTOTRADER_LOG_FILE, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()[-300:]

        matrix_pattern = re.compile(
            r"\[CYCLE #(\d+)\] Symbol: ([\w/]+) \| Price: \$([0-9,.]+) \| Technical: ([+-]?[0-9.]+) \| "
            r"Volume: ([+-]?[0-9.]+) \| Sentiment: ([+-]?[0-9.]+) \| Composite: ([+-]?[0-9.]+) => Signal: (\w+)"
        )
        portfolio_pattern = re.compile(
            r"Portfolio: Cash=\$([0-9,.]+) \| Tax Reserve=\$([0-9,.]+) \| Tradable Cash=\$([0-9,.]+) \| Crypto Positions=\[(.*?)\]"
        )

        for line in lines:
            m = matrix_pattern.search(line)
            if m:
                c_num, sym, price, tech, vol, sent, comp, sig = m.groups()
                matrix[sym] = {
                    "cycle": int(c_num),
                    "symbol": sym,
                    "price": price,
                    "technical": float(tech),
                    "volume": float(vol),
                    "sentiment": float(sent),
                    "composite": float(comp),
                    "signal": sig,
                }
            mp = portfolio_pattern.search(line)
            if mp:
                c, tr, tc, pos = mp.groups()
                portfolio = {
                    "cash": float(c.replace(",", "")),
                    "tax_reserve": float(tr.replace(",", "")),
                    "tradable_cash": float(tc.replace(",", "")),
                    "positions_str": pos,
                }
    except Exception as e:
        logger.warning("Could not parse autotrader.log: %s", e)

    return list(matrix.values()), portfolio


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(BASE_DIR), **kwargs)

    def do_GET(self):
        # Normalize path
        req_path = self.path.split("?")[0]

        if req_path in ("/", "/index.html", "/dashboard", "/dashboard.html"):
            self.serve_dashboard()
        elif req_path == "/api/status":
            self.serve_status_api()
        else:
            super().do_GET()

    def serve_dashboard(self):
        if not DASHBOARD_HTML.exists():
            self.send_error(404, "Dashboard HTML file not found")
            return

        try:
            with open(DASHBOARD_HTML, "rb") as f:
                content = f.read()

            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            logger.error("Error serving dashboard: %s", e)
            self.send_error(500, "Internal Server Error")

def build_fund_status_snapshot() -> dict:
    """Compiles live fund metrics from disk states into a unified status dictionary."""
    crypto_matrix, portfolio = parse_latest_daemon_state()
    data = {
        "status": "online",
        "fund_name": "ECKAP Quant Fund",
        "executive_briefing": {
            "summary": "ECKAP Quant Fund systems operational. 8 institutional strategies active with strict risk gates, Treasury ballast, tail hedges, and automated tax escrow.",
            "generated_at": None,
            "date": None,
            "trades_count": 0,
        },
        "tax_engine": {},
        "liquidity": {},
        "portfolio": portfolio,
        "crypto_matrix": crypto_matrix,
    }

    if BRIEFING_FILE.exists():
        try:
            with open(BRIEFING_FILE, "r", encoding="utf-8") as f:
                data["executive_briefing"] = json.load(f)
        except Exception as e:
            logger.warning("Could not read daily_briefing.json: %s", e)

    if TAX_RESERVE_FILE.exists():
        try:
            with open(TAX_RESERVE_FILE, "r", encoding="utf-8") as f:
                tax_data = json.load(f)
                data["tax_engine"] = {
                    "tax_reserve": round(tax_data.get("tax_reserve", 0.0), 2),
                    "tax_rate": tax_data.get("tax_rate", 0.3),
                    "total_realized_profit": round(tax_data.get("total_realized_profit", 0.0), 2),
                    "total_realized_loss": round(tax_data.get("total_realized_loss", 0.0), 2),
                    "total_tax_allocated": round(tax_data.get("total_tax_allocated", 0.0), 2),
                    "total_tax_credits": round(tax_data.get("total_tax_credits", 0.0), 2),
                    "net_realized_pnl": round(
                        tax_data.get("total_realized_profit", 0.0) - tax_data.get("total_realized_loss", 0.0), 2
                    ),
                    "trade_count": tax_data.get("trade_count", 0),
                    "last_updated": tax_data.get("last_updated"),
                    "recent_trades": tax_data.get("trade_history", []),
                    "trade_history": tax_data.get("trade_history", []),
                }
        except Exception as e:
            logger.warning("Could not read tax_reserve.json: %s", e)

    if LIQUIDITY_STATE_FILE.exists():
        try:
            with open(LIQUIDITY_STATE_FILE, "r", encoding="utf-8") as f:
                liq_data = json.load(f)
                data["liquidity"] = {
                    "pending_approval": liq_data.get("pending_approval"),
                    "history_count": len(liq_data.get("history", [])),
                    "processed_dividends": len(liq_data.get("processed_dividend_ids", [])),
                }
        except Exception as e:
            logger.warning("Could not read liquidity_state.json: %s", e)

    data["liquidity_barbell"] = {
        "total_deployed_usd": 0.0,
        "sgov": {
            "held": False,
            "shares": 0.0,
            "market_value": 0.0,
            "current_price": 100.53,
            "entry_price": 100.53,
            "unrealized_pnl": 0.0,
            "status": "STANDBY",
        },
        "fbnd": {
            "held": False,
            "shares": 0.0,
            "market_value": 0.0,
            "current_price": 44.19,
            "entry_price": 44.19,
            "unrealized_pnl": 0.0,
            "status": "STANDBY",
        },
        "unallocated_cash": 0.0,
    }

    data["wheel"] = {
        "active_positions_count": 0,
        "total_collateral_locked": 0.0,
        "total_symbols_count": 0,
        "updated_at": None,
        "wheels": {},
    }
    if WHEEL_STATE_FILE.exists():
        try:
            with open(WHEEL_STATE_FILE, "r", encoding="utf-8") as f:
                data["wheel"] = json.load(f)
        except Exception as e:
            logger.warning("Could not read wheel_state.json: %s", e)

    data["spreads"] = {
        "active_spreads": [],
        "closed_spreads": [],
        "active_count": 0,
        "collateral_locked": 0.0,
        "closed_count": 0,
    }
    if SPREAD_STATE_FILE.exists():
        try:
            with open(SPREAD_STATE_FILE, "r", encoding="utf-8") as f:
                sprd_data = json.load(f)
                active_list = sprd_data.get("active_spreads", [])
                closed_list = sprd_data.get("closed_spreads", [])
                collateral = sum(item.get("collateral_locked", 500.0) for item in active_list)
                realized_pnl = sum(item.get("realized_pnl", 0.0) for item in closed_list)
                unrealized_pnl = sum(item.get("unrealized_pnl", 0.0) for item in active_list)
                data["spreads"] = {
                    "active_spreads": active_list,
                    "closed_spreads": closed_list,
                    "active_count": len(active_list),
                    "collateral_locked": round(collateral, 2),
                    "closed_count": len(closed_list),
                    "total_realized_pnl": round(realized_pnl, 2),
                    "total_unrealized_pnl": round(unrealized_pnl, 2),
                    "updated_at": sprd_data.get("updated_at"),
                }
        except Exception as e:
            logger.warning("Could not read spreads_state.json: %s", e)

    data["tail_hedge"] = {
        "enabled": True,
        "underlying": "SPY",
        "monthly_budget": 150.0,
        "monthly_spent": 0.0,
        "monthly_budget_remaining": 150.0,
        "active_hedge": None,
        "has_active_hedge": False,
        "total_realized_pnl": 0.0,
        "monetized_count": 0,
        "closed_count": 0,
        "closed_hedges": [],
    }
    if HEDGE_STATE_FILE.exists():
        try:
            with open(HEDGE_STATE_FILE, "r", encoding="utf-8") as f:
                hdg_data = json.load(f)
                active = hdg_data.get("active_hedge")
                closed = hdg_data.get("closed_hedges", [])
                monthly_spent = float(hdg_data.get("monthly_spent_usd", 0.0))
                monthly_budget = 150.0
                realized = sum((h.get("realized_pnl") or 0.0) for h in closed)
                monetized = len([h for h in closed if "MONETIZATION" in str(h.get("exit_reason", ""))])
                data["tail_hedge"] = {
                    "enabled": True,
                    "underlying": active.get("underlying", "SPY") if active else "SPY",
                    "monthly_budget": round(monthly_budget, 2),
                    "monthly_spent": round(monthly_spent, 2),
                    "monthly_budget_remaining": round(max(0.0, monthly_budget - monthly_spent), 2),
                    "active_hedge": active,
                    "has_active_hedge": bool(active),
                    "total_realized_pnl": round(realized, 2),
                    "monetized_count": monetized,
                    "closed_count": len(closed),
                    "closed_hedges": closed[-10:],
                    "updated_at": hdg_data.get("updated_at"),
                }
        except Exception as e:
            logger.warning("Could not read tail_hedge_state.json: %s", e)

    data["dip_buyer"] = {
        "enabled": True,
        "order_size_usd": 1000.0,
        "max_capital_usd": 5000.0,
        "capital_deployed_usd": 0.0,
        "active_positions_count": 0,
        "active_positions": [],
        "closed_trades_count": 0,
        "closed_trades": [],
        "total_realized_pnl": 0.0,
        "win_rate_pct": 0.0,
        "scanner_matrix": {},
    }
    if DIP_STATE_FILE.exists():
        try:
            with open(DIP_STATE_FILE, "r", encoding="utf-8") as f:
                dip_data = json.load(f)
                active_positions = list(dip_data.get("active_positions", {}).values())
                closed_trades = dip_data.get("closed_trades", [])
                deployed = sum(p.get("cost_basis", 0.0) for p in active_positions)
                realized = sum(t.get("realized_pnl", 0.0) for t in closed_trades)
                wins = len([t for t in closed_trades if t.get("realized_pnl", 0.0) > 0])
                win_rate = (wins / len(closed_trades) * 100.0) if closed_trades else 0.0
                data["dip_buyer"] = {
                    "enabled": True,
                    "order_size_usd": 1000.0,
                    "max_capital_usd": 5000.0,
                    "capital_deployed_usd": round(deployed, 2),
                    "active_positions_count": len(active_positions),
                    "active_positions": active_positions,
                    "closed_trades_count": len(closed_trades),
                    "closed_trades": closed_trades[-10:],
                    "total_realized_pnl": round(realized, 2),
                    "win_rate_pct": round(win_rate, 1),
                    "scanner_matrix": dip_data.get("last_scan", {}),
                }
        except Exception as e:
            logger.warning("Could not read dip_buyer_state.json: %s", e)

    data["macro_rotation"] = {
        "enabled": True,
        "regime": "STANDBY",
        "max_capital_usd": 5000.0,
        "capital_deployed_usd": 0.0,
        "active_positions_count": 0,
        "active_positions": [],
        "closed_trades_count": 0,
        "closed_trades": [],
        "total_realized_pnl": 0.0,
        "total_tax_escrow": 0.0,
        "leaderboard": [],
        "last_rebalance_time": None,
    }
    if MACRO_STATE_FILE.exists():
        try:
            with open(MACRO_STATE_FILE, "r", encoding="utf-8") as f:
                macro_data = json.load(f)
                active_pos = list(macro_data.get("active_positions", {}).values())
                closed_tr = macro_data.get("closed_trades", [])
                deployed = sum(p.get("cost_basis", 0.0) for p in active_pos)
                data["macro_rotation"] = {
                    "enabled": True,
                    "regime": macro_data.get("current_regime", "STANDBY"),
                    "max_capital_usd": 5000.0,
                    "capital_deployed_usd": round(deployed, 2),
                    "active_positions_count": len(active_pos),
                    "active_positions": active_pos,
                    "closed_trades_count": len(closed_tr),
                    "closed_trades": closed_tr[-10:],
                    "total_realized_pnl": round(macro_data.get("total_realized_pnl", 0.0), 2),
                    "total_tax_escrow": round(macro_data.get("total_tax_escrow", 0.0), 2),
                    "leaderboard": list(macro_data.get("last_leaderboard", {}).values()),
                    "last_rebalance_time": macro_data.get("last_rebalance_time"),
                }
        except Exception as e:
            logger.warning("Could not read macro_rotation_state.json: %s", e)

    data["pairs_trading"] = {
        "enabled": True,
        "max_capital_usd": 5000.0,
        "allocated_capital_usd": 0.0,
        "available_capital_usd": 5000.0,
        "active_positions_count": 0,
        "active_positions": [],
        "closed_trades_count": 0,
        "closed_trades": [],
        "scanned_metrics": {},
        "total_realized_pnl": 0.0,
        "total_tax_escrow": 0.0,
        "win_rate_pct": 0.0,
    }
    if PAIRS_STATE_FILE.exists():
        try:
            with open(PAIRS_STATE_FILE, "r", encoding="utf-8") as f:
                pairs_data = json.load(f)
                active_pos = list(pairs_data.get("active_positions", {}).values())
                closed_tr = pairs_data.get("closed_trades", [])
                deployed = sum(p.get("total_notional", 0.0) for p in active_pos)
                wins = sum(1 for t in closed_tr if t.get("net_realized_pnl", 0.0) > 0)
                win_rate = (wins / len(closed_tr) * 100.0) if closed_tr else 0.0
                data["pairs_trading"] = {
                    "enabled": True,
                    "max_capital_usd": 5000.0,
                    "allocated_capital_usd": round(deployed, 2),
                    "available_capital_usd": round(max(0.0, 5000.0 - deployed), 2),
                    "active_positions_count": len(active_pos),
                    "active_positions": active_pos,
                    "closed_trades_count": len(closed_tr),
                    "closed_trades": closed_tr[-10:],
                    "scanned_metrics": pairs_data.get("last_scanned_metrics", {}),
                    "total_realized_pnl": round(pairs_data.get("total_realized_pnl", 0.0), 2),
                    "total_tax_escrow": round(pairs_data.get("total_tax_escrow", 0.0), 2),
                    "win_rate_pct": round(win_rate, 1),
                }
        except Exception as e:
            logger.warning("Could not read pairs_trading_state.json: %s", e)

    alpaca_live = fetch_live_alpaca_telemetry()
    if alpaca_live:
        acc = alpaca_live["account"]
        positions = alpaca_live["positions"]
        open_orders = alpaca_live["open_orders"]

        # 1. Live Portfolio Telemetry
        tax_res = data.get("tax_engine", {}).get("tax_reserve", 0.0)
        tradable_cash = max(0.0, acc["cash"] - tax_res)
        data["portfolio"] = {
            "cash": round(acc["cash"], 2),
            "tax_reserve": round(tax_res, 2),
            "tradable_cash": round(tradable_cash, 2),
            "portfolio_value": round(acc["portfolio_value"], 2),
            "buying_power": round(acc["buying_power"], 2),
            "options_buying_power": round(acc["options_buying_power"], 2),
            "long_market_value": round(acc["long_market_value"], 2),
            "short_market_value": round(acc["short_market_value"], 2),
            "positions_str": f"{len(positions)} Active Positions on Alpaca",
        }

        # 2. Live Liquidity Barbell (SGOV + FBND)
        sgov_pos = next((p for p in positions if "equity" in str(getattr(p, "asset_class", "")).lower() and p.symbol == "SGOV"), None)
        fbnd_pos = next((p for p in positions if "equity" in str(getattr(p, "asset_class", "")).lower() and p.symbol == "FBND"), None)

        sgov_val = float(sgov_pos.market_value) if sgov_pos else 0.0
        fbnd_val = float(fbnd_pos.market_value) if fbnd_pos else 0.0
        total_barbell = sgov_val + fbnd_val

        data["liquidity_barbell"] = {
            "total_deployed_usd": round(total_barbell, 2),
            "sgov": {
                "held": sgov_pos is not None,
                "shares": round(float(sgov_pos.qty), 4) if sgov_pos else 0.0,
                "market_value": round(sgov_val, 2),
                "current_price": round(float(sgov_pos.current_price), 2) if sgov_pos else 0.0,
                "entry_price": round(float(sgov_pos.avg_entry_price), 2) if sgov_pos else 0.0,
                "unrealized_pnl": round(float(sgov_pos.unrealized_pl), 2) if sgov_pos else 0.0,
                "status": "ACTIVE HOLDING" if sgov_pos else "STANDBY",
            },
            "fbnd": {
                "held": fbnd_pos is not None,
                "shares": round(float(fbnd_pos.qty), 4) if fbnd_pos else 0.0,
                "market_value": round(fbnd_val, 2),
                "current_price": round(float(fbnd_pos.current_price), 2) if fbnd_pos else 0.0,
                "entry_price": round(float(fbnd_pos.avg_entry_price), 2) if fbnd_pos else 0.0,
                "unrealized_pnl": round(float(fbnd_pos.unrealized_pl), 2) if fbnd_pos else 0.0,
                "status": "ACTIVE HOLDING" if fbnd_pos else "STANDBY",
            },
            "unallocated_cash": round(tradable_cash, 2),
        }

        # 3. Live Option Wheel Telemetry (10 tickers)
        wheel_symbols = ["INTC", "F", "SOFI", "HOOD", "PLTR", "XLF", "RIVN", "PFE", "NU", "CLF"]
        company_names = {
            "INTC": "Intel Corporation", "F": "Ford Motor Company", "SOFI": "SoFi Technologies",
            "HOOD": "Robinhood Markets", "PLTR": "Palantir Technologies", "XLF": "Financial Select Sector SPDR",
            "RIVN": "Rivian Automotive", "PFE": "Pfizer Inc.", "NU": "Nu Holdings Ltd.", "CLF": "Cleveland-Cliffs Inc."
        }

        wheels_data = {}
        total_wheel_collateral = 0.0
        active_wheel_count = 0
        today = datetime.now(timezone.utc).date()

        for sym in wheel_symbols:
            opt = next((p for p in positions if "option" in str(getattr(p, "asset_class", "")).lower() and p.symbol.startswith(sym)), None)
            stk = next((p for p in positions if "equity" in str(getattr(p, "asset_class", "")).lower() and p.symbol == sym), None)
            open_ord = next((o for o in open_orders if getattr(o, "symbol", "").startswith(sym)), None)

            if opt:
                active_wheel_count += 1
                symbol = opt.symbol
                occ_part = symbol[len(sym):]
                if len(occ_part) >= 15:
                    exp_str = f"20{occ_part[:2]}-{occ_part[2:4]}-{occ_part[4:6]}"
                    c_type = "put" if occ_part[6].upper() == "P" else "call"
                    strike = float(occ_part[7:]) / 1000.0
                else:
                    exp_str = ""
                    c_type = "put"
                    strike = float(opt.avg_entry_price)

                try:
                    exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                    dte = max(0, (exp_date - today).days)
                except Exception:
                    dte = 0

                entry_prem = float(opt.avg_entry_price)
                cur_price = float(opt.current_price)
                unrealized_pl = float(opt.unrealized_pl)
                qty = abs(int(float(opt.qty)))
                collateral = strike * 100 * qty
                total_wheel_collateral += collateral

                target_profit_dollars = (entry_prem * 0.5) * 100 * qty
                current_profit_dollars = (entry_prem - cur_price) * 100 * qty
                progress_pct = max(0.0, min(100.0, (current_profit_dollars / target_profit_dollars * 100.0))) if target_profit_dollars > 0 else 0.0

                pnl_pct = (unrealized_pl / (entry_prem * 100 * qty) * 100.0) if (entry_prem > 0 and qty > 0) else 0.0

                phase_label = "Cash-Secured Put" if c_type == "put" else "Covered Call"
                status_label = "ACTIVE PUT" if c_type == "put" else "ACTIVE CALL"

                wheels_data[sym] = {
                    "state": "MONITORING_PUT" if c_type == "put" else "MONITORING_CALL",
                    "underlying": sym,
                    "stock_price": cur_price,
                    "shares_held": int(float(stk.qty)) if stk else 0,
                    "collateral_locked": collateral,
                    "details": {
                        "company_name": company_names.get(sym, sym),
                        "status_label": status_label,
                        "phase": phase_label,
                        "contract_symbol": symbol,
                        "strike_price": strike,
                        "expiration_date": exp_str,
                        "days_to_expiration": dte,
                        "contracts": qty,
                        "entry_premium": entry_prem,
                        "current_option_price": cur_price,
                        "unrealized_pnl": unrealized_pl,
                        "unrealized_pnl_pct": round(pnl_pct, 1),
                        "progress_to_target": round(progress_pct, 1),
                        "target_buyback_price": round(entry_prem * 0.5, 2),
                        "collateral_locked": collateral,
                        "shares_held": int(float(stk.qty)) if stk else 0,
                        "stock_price": cur_price,
                        "note": f"Strike ${strike:.2f} ({dte} DTE)",
                    }
                }
            elif open_ord:
                ord_sym = getattr(open_ord, "symbol", sym)
                wheels_data[sym] = {
                    "state": "MONITORING_PUT",
                    "underlying": sym,
                    "collateral_locked": 0.0,
                    "details": {
                        "company_name": company_names.get(sym, sym),
                        "status_label": "PENDING ORDER",
                        "phase": "Cash-Secured Put",
                        "contract_symbol": ord_sym,
                        "collateral_locked": 0.0,
                        "unrealized_pnl": 0.0,
                        "unrealized_pnl_pct": 0.0,
                        "progress_to_target": 0.0,
                        "note": f"Order {str(getattr(open_ord, 'id', ''))[:8]} pending in book",
                    }
                }
            else:
                wheels_data[sym] = {
                    "state": "STANDBY",
                    "underlying": sym,
                    "collateral_locked": 0.0,
                    "details": {
                        "company_name": company_names.get(sym, sym),
                        "status_label": "EXPANSION READY" if sym in ["RIVN", "PFE", "NU", "CLF"] else "STANDBY",
                        "phase": "Cash-Secured Put",
                        "contract_symbol": f"{sym} (Standby)",
                        "collateral_locked": 0.0,
                        "unrealized_pnl": 0.0,
                        "unrealized_pnl_pct": 0.0,
                        "progress_to_target": 0.0,
                        "note": "Tier 2 Expansion Asset" if sym in ["RIVN", "PFE", "NU", "CLF"] else "Monitoring underlying for entry signal.",
                    }
                }

        data["wheel"] = {
            "active_positions_count": active_wheel_count,
            "total_collateral_locked": round(total_wheel_collateral, 2),
            "total_symbols_count": len(wheel_symbols),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "wheels": wheels_data,
        }

    return data


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(BASE_DIR), **kwargs)

    def send_json_response(self, data: dict, status: int = 200):
        payload = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(payload)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_GET(self):
        req_path = self.path.split("?")[0]

        if req_path in ("/", "/index.html", "/dashboard", "/dashboard.html"):
            self.serve_dashboard()
        elif req_path == "/api/status":
            self.serve_status_api()
        else:
            super().do_GET()

    def do_POST(self):
        req_path = self.path.split("?")[0]
        if req_path == "/api/chat":
            self.handle_chat_api()
        else:
            self.send_error(404, "Endpoint not found")

    def handle_chat_api(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"
            payload = json.loads(raw_body)
            user_msg = payload.get("message", "").strip()
            if not user_msg:
                self.send_json_response({"error": "Empty query"}, status=400)
                return

            snapshot = build_fund_status_snapshot()
            from intelligence.llm_advisor import get_llm_advisor
            reply = get_llm_advisor().answer_fund_query(user_msg, snapshot)
            self.send_json_response({
                "status": "ok",
                "reply": reply,
                "timestamp": datetime.now().isoformat(),
            })
        except Exception as e:
            logger.error("Error processing /api/chat: %s", e)
            self.send_json_response({"status": "error", "reply": f"Error processing query: {e}"}, status=500)

    def serve_dashboard(self):
        if not DASHBOARD_HTML.exists():
            self.send_error(404, "Dashboard HTML file not found")
            return

        try:
            with open(DASHBOARD_HTML, "rb") as f:
                content = f.read()

            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            logger.error("Error serving dashboard: %s", e)
            self.send_error(500, "Internal Server Error")

    def serve_status_api(self):
        """Returns live fund metrics from disk states."""
        data = build_fund_status_snapshot()
        self.send_json_response(data)

    def log_message(self, format, *args):
        logger.info("%s - %s", self.address_string(), format % args)


def run_server(host="0.0.0.0", port=8080):
    server_address = (host, port)
    httpd = ThreadingHTTPServer(server_address, DashboardHandler)
    logger.info("Serving AutoTrader Investor Dashboard at http://%s:%d", host, port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Stopping Dashboard Server...")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    port = int(os.environ.get("DASHBOARD_PORT", 8080))
    run_server(port=port)
