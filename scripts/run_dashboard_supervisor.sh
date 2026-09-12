#!/bin/bash
# ==============================================================================
# AutoTrader Dashboard & Tunnel Foreground Supervisor
# Designed for macOS launchd (com.autotrader.dashboard.plist)
# Runs dashboard_server.py and cloudflared tunnel, monitoring health.
# ==============================================================================

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DIR"

LOG_SERVER="$DIR/dashboard_server.log"
LOG_TUNNEL="$DIR/cloudflared.log"
URL_FILE="$DIR/public_dashboard_url.txt"
NTFY_TOPIC=$(grep "^NTFY_TOPIC=" "$DIR/.env" 2>/dev/null | cut -d '=' -f2 | tr -d ' "' || echo "eckermike87")

SERVER_PID=""
TUNNEL_PID=""

cleanup() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [SUPERVISOR] Shutting down dashboard and tunnel..."
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
    fi
    if [ -n "$TUNNEL_PID" ] && kill -0 "$TUNNEL_PID" 2>/dev/null; then
        kill "$TUNNEL_PID" 2>/dev/null || true
    fi
    wait 2>/dev/null || true
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [SUPERVISOR] Stopped cleanly."
    exit 0
}

trap cleanup SIGINT SIGTERM SIGHUP

start_server() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [SUPERVISOR] Starting dashboard_server.py on port 8080..."
    "$DIR/.venv/bin/python" "$DIR/dashboard_server.py" >> "$LOG_SERVER" 2>&1 &
    SERVER_PID=$!
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [SUPERVISOR] dashboard_server.py started with PID $SERVER_PID"
}

start_tunnel() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [SUPERVISOR] Starting Cloudflare tunnel..."
    rm -f "$URL_FILE"
    if [ -f "$DIR/bin/cloudflared" ]; then
        "$DIR/bin/cloudflared" tunnel --url http://127.0.0.1:8080 > "$LOG_TUNNEL" 2>&1 &
        TUNNEL_PID=$!
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] [SUPERVISOR] cloudflared started with PID $TUNNEL_PID"
        
        # Check for URL
        (
            for i in {1..20}; do
                sleep 1
                URL=$(grep -o 'https://[a-zA-Z0-9-]*\.trycloudflare\.com' "$LOG_TUNNEL" 2>/dev/null | head -n 1 || true)
                if [ -n "$URL" ]; then
                    echo "$URL" > "$URL_FILE"
                    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [SUPERVISOR] Tunnel URL live: $URL"
                    curl -s -d "🌐 AutoTrader Dashboard restarted:\n$URL" \
                         -H "Title: Investor Dashboard Online" \
                         -H "Tags: globe,chart_with_upwards_trend" \
                         -H "Click: $URL" \
                         "https://ntfy.sh/$NTFY_TOPIC" > /dev/null 2>&1 || true
                    break
                fi
            done
        ) &
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] [SUPERVISOR] Warning: $DIR/bin/cloudflared not found. Serving locally only."
    fi
}

# Initial start
start_server
sleep 1
start_tunnel

echo "[$(date '+%Y-%m-%d %H:%M:%S')] [SUPERVISOR] Monitoring services..."

# Supervise loop
while true; do
    sleep 5

    # Check server
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] [SUPERVISOR] dashboard_server.py died! Restarting..."
        start_server
    fi

    # Check tunnel
    if [ -f "$DIR/bin/cloudflared" ] && ! kill -0 "$TUNNEL_PID" 2>/dev/null; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] [SUPERVISOR] cloudflared tunnel died! Restarting..."
        start_tunnel
    fi
done
