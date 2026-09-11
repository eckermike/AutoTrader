#!/usr/bin/env python3
"""
AutoTrader Investor Dashboard Server
Serves the interactive dashboard.html and real-time JSON fund status API.
"""

import os
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


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


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
        """Returns live fund metrics from disk states (tax escrow, liquid reserves, trade counts)."""
        data = {
            "status": "online",
            "fund_name": "AutoTrader Capital Partners",
            "tax_engine": {},
            "liquidity": {},
        }

        if TAX_RESERVE_FILE.exists():
            try:
                with open(TAX_RESERVE_FILE, "r", encoding="utf-8") as f:
                    tax_data = json.load(f)
                    data["tax_engine"] = {
                        "tax_reserve": round(tax_data.get("tax_reserve", 0.0), 2),
                        "total_realized_profit": round(tax_data.get("total_realized_profit", 0.0), 2),
                        "total_realized_loss": round(tax_data.get("total_realized_loss", 0.0), 2),
                        "net_realized_pnl": round(
                            tax_data.get("total_realized_profit", 0.0) - tax_data.get("total_realized_loss", 0.0), 2
                        ),
                        "trade_count": tax_data.get("trade_count", 0),
                        "last_updated": tax_data.get("last_updated"),
                        "recent_trades": tax_data.get("trade_history", [])[-10:],
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
