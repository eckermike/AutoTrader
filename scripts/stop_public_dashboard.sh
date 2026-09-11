#!/bin/bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SERVER_PID_FILE="$DIR/dashboard_server.pid"
TUNNEL_PID_FILE="$DIR/cloudflared.pid"

echo "==> Stopping AutoTrader Public Dashboard..."

if [ -f "$SERVER_PID_FILE" ]; then
    PID=$(cat "$SERVER_PID_FILE")
    echo "Stopping Dashboard Server (PID $PID)..."
    kill "$PID" 2>/dev/null || true
    rm -f "$SERVER_PID_FILE"
fi

if [ -f "$TUNNEL_PID_FILE" ]; then
    PID=$(cat "$TUNNEL_PID_FILE")
    echo "Stopping Cloudflare Tunnel (PID $PID)..."
    kill "$PID" 2>/dev/null || true
    rm -f "$TUNNEL_PID_FILE"
fi

rm -f "$DIR/public_dashboard_url.txt"
echo "==> Public Dashboard stopped."
