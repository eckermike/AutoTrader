#!/bin/bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DIR"

SERVER_PID_FILE="$DIR/dashboard_server.pid"
TUNNEL_PID_FILE="$DIR/cloudflared.pid"
LOG_SERVER="$DIR/dashboard_server.log"
LOG_TUNNEL="$DIR/cloudflared.log"
URL_FILE="$DIR/public_dashboard_url.txt"
NTFY_TOPIC=$(grep "^NTFY_TOPIC=" "$DIR/.env" | cut -d '=' -f2 | tr -d ' "' || echo "eckermike87")

echo "==> Starting AutoTrader Dashboard Server on port 8080..."
# Stop old server if running
if [ -f "$SERVER_PID_FILE" ]; then
    kill -9 "$(cat "$SERVER_PID_FILE")" 2>/dev/null || true
    rm -f "$SERVER_PID_FILE"
fi

# Start Dashboard HTTP server
"$DIR/.venv/bin/python" "$DIR/dashboard_server.py" > "$LOG_SERVER" 2>&1 &
echo $! > "$SERVER_PID_FILE"
echo "==> Dashboard Server started (PID $(cat "$SERVER_PID_FILE"))"

# Stop old tunnel if running
if [ -f "$TUNNEL_PID_FILE" ]; then
    kill -9 "$(cat "$TUNNEL_PID_FILE")" 2>/dev/null || true
    rm -f "$TUNNEL_PID_FILE"
fi

echo "==> Establishing Cloudflare Secure Public Tunnel..."
rm -f "$LOG_TUNNEL" "$URL_FILE"

"$DIR/bin/cloudflared" tunnel --url http://127.0.0.1:8080 > "$LOG_TUNNEL" 2>&1 &
echo $! > "$TUNNEL_PID_FILE"

# Wait for tunnel URL to appear in log
echo "==> Waiting for public HTTPS URL..."
PUBLIC_URL=""
for i in {1..20}; do
    sleep 1
    PUBLIC_URL=$(grep -o 'https://[a-zA-Z0-9-]*\.trycloudflare\.com' "$LOG_TUNNEL" | head -n 1 || true)
    if [ -n "$PUBLIC_URL" ]; then
        break
    fi
done

if [ -n "$PUBLIC_URL" ]; then
    echo "$PUBLIC_URL" > "$URL_FILE"
    echo ""
    echo "=================================================================="
    echo "🎉 AutoTrader Live Investor Dashboard is now available externally!"
    echo "🔗 URL: $PUBLIC_URL"
    echo "=================================================================="
    echo ""
    
    # Send push notification to user's phone
    curl -s -d "🌐 AutoTrader Live Investor Dashboard is online:\n$PUBLIC_URL" \
         -H "Title: Investor Dashboard Live" \
         -H "Tags: globe,chart_with_upwards_trend" \
         -H "Click: $PUBLIC_URL" \
         "https://ntfy.sh/$NTFY_TOPIC" > /dev/null 2>&1 || true
else
    echo "⚠️ Could not extract Cloudflare tunnel URL yet. Check $LOG_TUNNEL."
fi
