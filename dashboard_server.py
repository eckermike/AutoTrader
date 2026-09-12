#!/usr/bin/env python3
"""
AutoTrader Investor Dashboard Server
Serves the interactive dashboard.html and real-time JSON fund status API.
"""

import os
import re
import json
import logging
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("dashboard.server")

BASE_DIR = Path(__file__).resolve().parent
DASHBOARD_HTML = BASE_DIR / "dashboard.html"
TAX_RESERVE_FILE = BASE_DIR / "tax_reserve.json"
LIQUIDITY_STATE_FILE = BASE_DIR / "liquidity_state.json"
SPREAD_STATE_FILE = BASE_DIR / "spreads_state.json"
HEDGE_STATE_FILE = BASE_DIR / "tail_hedge_state.json"


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

    def serve_status_api(self):
        """Returns live fund metrics from disk states (tax escrow, liquid reserves, trade counts, crypto matrix)."""
        crypto_matrix, portfolio = parse_latest_daemon_state()
        data = {
            "status": "online",
            "fund_name": "AutoTrader Capital Partners",
            "tax_engine": {},
            "liquidity": {},
            "portfolio": portfolio,
            "crypto_matrix": crypto_matrix,
        }

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

        payload = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(payload)

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
