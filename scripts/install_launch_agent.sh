#!/bin/bash
set -e

PLIST_SRC="/Users/mikeeckerle/AutoTrader/com.autotrader.bot.plist"
PLIST_DST="$HOME/Library/LaunchAgents/com.autotrader.bot.plist"

echo "==> Installing AutoTrader LaunchAgent..."
mkdir -p "$HOME/Library/LaunchAgents"

# Unload previous instance if loaded
if launchctl list | grep -q "com.autotrader.bot"; then
    echo "==> Unloading existing agent..."
    launchctl unload "$PLIST_DST" 2>/dev/null || true
fi

# Copy plist to LaunchAgents
cp "$PLIST_SRC" "$PLIST_DST"
chmod 644 "$PLIST_DST"

# Load new agent
echo "==> Loading AutoTrader LaunchAgent into launchd..."
launchctl load -w "$PLIST_DST"

echo "==> AutoTrader LaunchAgent successfully installed and started!"
echo "==> Bot will now automatically launch on every login and auto-restart if terminated."
echo "==> Live logs: tail -f /Users/mikeeckerle/AutoTrader/autotrader.log"
