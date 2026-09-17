"""
Tests for Dashboard Server status snapshot, CORS options, and /api/chat endpoint.
Uses in-memory stream testing to remain 100% compliant with sandbox policies.
"""

import io
import json
from unittest.mock import MagicMock
from dashboard_server import DashboardHandler, build_fund_status_snapshot, compute_income_analytics, BASE_DIR


def make_mock_handler(command: str = "GET", path: str = "/api/status", body: bytes = b""):
    """Creates a mock DashboardHandler without requiring real network sockets."""
    handler = DashboardHandler.__new__(DashboardHandler)
    handler.server = MagicMock()
    handler.client_address = ("127.0.0.1", 12345)
    handler.command = command
    handler.path = path
    handler.requestline = f"{command} {path} HTTP/1.1"
    handler.request_version = "HTTP/1.1"
    handler.headers = {
        "Content-Length": str(len(body)),
        "Content-Type": "application/json",
    }
    handler.rfile = io.BytesIO(body)
    handler.wfile = io.BytesIO()
    handler.close_connection = True
    handler.directory = str(BASE_DIR)
    return handler


def parse_response(handler: DashboardHandler):
    """Parses raw wfile bytes into status line, headers, and JSON body."""
    raw = handler.wfile.getvalue().decode("utf-8")
    parts = raw.split("\r\n\r\n", 1)
    header_block = parts[0]
    body_block = parts[1] if len(parts) > 1 else ""
    return header_block, body_block


def test_build_fund_status_snapshot_contains_executive_briefing():
    snapshot = build_fund_status_snapshot()
    assert "executive_briefing" in snapshot
    assert "summary" in snapshot["executive_briefing"]
    assert "tax_engine" in snapshot
    assert "portfolio" in snapshot


def test_build_fund_status_snapshot_contains_liquidity_barbell():
    snapshot = build_fund_status_snapshot()
    assert "liquidity_barbell" in snapshot
    bar = snapshot["liquidity_barbell"]
    assert "sgov" in bar
    assert "fbnd" in bar
    assert "total_deployed_usd" in bar
    assert "status" in bar["sgov"]
    assert "status" in bar["fbnd"]


def test_api_status_handler():
    handler = make_mock_handler(command="GET", path="/api/status")
    handler.serve_status_api()

    headers, body = parse_response(handler)
    assert "200 OK" in headers
    assert "application/json" in headers
    data = json.loads(body)
    assert data["status"] == "online"
    assert "executive_briefing" in data
    assert "fund_name" in data


def test_api_options_cors():
    handler = make_mock_handler(command="OPTIONS", path="/api/chat")
    handler.do_OPTIONS()

    headers, _ = parse_response(handler)
    assert "200 OK" in headers
    assert "Access-Control-Allow-Origin: *" in headers
    assert "Access-Control-Allow-Methods: GET, POST, OPTIONS" in headers


def test_api_chat_valid_query():
    body_data = json.dumps({"message": "What is our tax reserve?"}).encode("utf-8")
    handler = make_mock_handler(command="POST", path="/api/chat", body=body_data)
    handler.handle_chat_api()

    headers, body = parse_response(handler)
    assert "200 OK" in headers
    data = json.loads(body)
    assert data["status"] == "ok"
    assert "reply" in data
    assert len(data["reply"]) > 0


def test_api_chat_empty_query_rejected():
    body_data = json.dumps({"message": ""}).encode("utf-8")
    handler = make_mock_handler(command="POST", path="/api/chat", body=body_data)
    handler.handle_chat_api()

    headers, body = parse_response(handler)
    assert "400 Bad Request" in headers
    data = json.loads(body)
    assert "error" in data


def test_build_fund_status_snapshot_contains_income_analytics():
    snapshot = build_fund_status_snapshot()
    assert "income_analytics" in snapshot
    analytics = snapshot["income_analytics"]
    assert "dates" in analytics
    assert len(analytics["dates"]) > 0
    assert "by_strategy" in analytics
    assert "wheel" in analytics["by_strategy"]
    assert "bonds" in analytics["by_strategy"]
    assert "spreads" in analytics["by_strategy"]
    assert "crypto" in analytics["by_strategy"]
    assert "macro_dip" in analytics["by_strategy"]
    assert "overall_cumulative" in analytics
    assert "summary" in analytics
    assert analytics["summary"]["total_income"] > 0
    assert analytics["summary"]["monthly_run_rate"] > 0


def test_api_income_endpoint():
    handler = make_mock_handler(command="GET", path="/api/income")
    handler.do_GET()

    headers, body = parse_response(handler)
    assert "200 OK" in headers
    assert "application/json" in headers
    data = json.loads(body)
    assert "dates" in data
    assert "summary" in data
    assert "strategy_totals" in data["summary"]

