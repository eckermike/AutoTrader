#!/bin/bash
set -e

PLIST_DST="$HOME/Library/LaunchAgents/com.autotrader.bot.plist"

echo "==> Stopping and removing AutoTrader LaunchAgent..."
if launchctl list | grep -q "com.autotrader.bot"; then
    launchctl unload "$PLIST_DST" 2>/dev/null || true
fi

rm -f "$PLIST_DST"
echo "==> AutoTrader LaunchAgent uninstalled."
