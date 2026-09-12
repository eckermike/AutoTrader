#!/bin/bash
set -e

LAUNCH_DIR="$HOME/Library/LaunchAgents"

uninstall_agent() {
    local label="$1"
    local plist_dst="$LAUNCH_DIR/$label.plist"

    echo "==> Stopping and removing $label LaunchAgent..."
    if launchctl list | grep -q "$label"; then
        launchctl unload "$plist_dst" 2>/dev/null || true
    fi
    rm -f "$plist_dst"
    echo "    $label uninstalled."
}

uninstall_agent "com.autotrader.bot"
uninstall_agent "com.autotrader.dashboard"

echo "==> All AutoTrader LaunchAgents uninstalled."
