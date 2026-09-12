#!/bin/bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAUNCH_DIR="$HOME/Library/LaunchAgents"

mkdir -p "$LAUNCH_DIR"

install_agent() {
    local label="$1"
    local plist_src="$DIR/$label.plist"
    local plist_dst="$LAUNCH_DIR/$label.plist"

    echo "==> Installing $label LaunchAgent..."
    if launchctl list | grep -q "$label"; then
        echo "    Unloading existing $label..."
        launchctl unload "$plist_dst" 2>/dev/null || true
    fi

    cp "$plist_src" "$plist_dst"
    chmod 644 "$plist_dst"
    echo "    Loading $label into launchd..."
    launchctl load -w "$plist_dst"
    echo "    $label successfully loaded!"
}

# Install Bot Agent
install_agent "com.autotrader.bot"

# Install Dashboard Agent
install_agent "com.autotrader.dashboard"

echo ""
echo "=================================================================="
echo "🎉 AutoTrader LaunchAgents successfully installed!"
echo "• Bot Daemon:      com.autotrader.bot (main.py)"
echo "• Dashboard Agent: com.autotrader.dashboard (dashboard_server.py + tunnel)"
echo "Both services will automatically launch on every login and auto-restart if terminated."
echo "Live Logs:"
echo "• Bot:       tail -f $DIR/autotrader.log"
echo "• Dashboard: tail -f $DIR/dashboard_supervisor.log"
echo "=================================================================="
