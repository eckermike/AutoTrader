"""
Notification Engine for Cryptocurrency Trading Bot.
Provides instant trade alerting via:
- ntfy.sh Web Push / iOS PWA notifications (instant push alerts with sound and vibration)
- Native Apple iMessage
- macOS Desktop Notification Banners with sound
"""

import logging
import subprocess
from typing import Optional
import requests

logger = logging.getLogger("notifier")


class TradeNotifier:
    """
    Dispatches real-time trade alerts via ntfy.sh (iOS PWA push), macOS iMessage,
    and Desktop Notification banners.
    """

    def __init__(
        self,
        recipient: Optional[str] = "eckermike87@icloud.com",
        ntfy_topic: Optional[str] = "eckermike87",
        enabled: bool = True,
        macos_banner: bool = True,
    ):
        self.recipient = recipient
        self.ntfy_topic = ntfy_topic
        self.enabled = enabled
        self.macos_banner = macos_banner

    def send_ntfy(
        self,
        message: str,
        title: str = "AutoTrader Alert",
        priority: str = "default",
        tags: str = "robot",
        actions: Optional[str] = None,
    ) -> bool:
        """
        Sends an instant Web Push notification to ntfy.sh/topic.
        Delivers push banners, sound, and vibrations to iPhone / Apple Watch.
        """
        if not self.enabled or not self.ntfy_topic:
            return False

        url = f"https://ntfy.sh/{self.ntfy_topic}"
        # HTTP headers must be latin-1/ASCII safe; non-ASCII emojis go in message body and Tags
        clean_title = title.encode("ascii", "ignore").decode("ascii").strip() or "AutoTrader Alert"
        headers = {
            "Title": clean_title,
            "Priority": priority,
            "Tags": tags,
        }
        if actions:
            headers["Actions"] = actions


        try:
            resp = requests.post(
                url,
                data=message.encode("utf-8"),
                headers=headers,
                timeout=5.0,
            )
            if resp.status_code == 200:
                logger.info("ntfy push alert sent to %s", url)
                return True
            else:
                logger.warning("ntfy returned HTTP %d: %s", resp.status_code, resp.text[:100])
                return False
        except Exception as e:
            logger.warning("ntfy push dispatch failed: %s", e)
            return False

    def send_imessage(self, message: str) -> bool:
        """
        Sends an iMessage using native macOS AppleScript.
        """
        if not self.enabled or not self.recipient:
            return False

        escaped_message = message.replace("\\", "\\\\").replace('"', '\\"')

        applescript = f'''
        tell application "Messages"
            try
                set targetService to 1st service whose service type = iMessage
                set targetBuddy to buddy "{self.recipient}" of targetService
                send "{escaped_message}" to targetBuddy
                return "SUCCESS"
            on error errMsg
                return "ERROR: " & errMsg
            end try
        end tell
        '''

        try:
            res = subprocess.run(
                ["osascript", "-e", applescript],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if res.returncode == 0 and "SUCCESS" in res.stdout:
                logger.info("iMessage trade alert sent to %s", self.recipient)
                return True
            else:
                logger.warning("Failed to send iMessage: %s", res.stdout.strip())
                return False
        except Exception as e:
            logger.warning("iMessage dispatch encountered an error: %s", e)
            return False

    def send_macos_banner(self, title: str, subtitle: str, body: str) -> bool:
        """Displays a local macOS system notification banner with sound."""
        if not self.macos_banner:
            return False

        esc_title = title.replace('"', '\\"')
        esc_sub = subtitle.replace('"', '\\"')
        esc_body = body.replace('"', '\\"')

        script = f'display notification "{esc_body}" with title "{esc_title}" subtitle "{esc_sub}" sound name "Glass"'
        try:
            subprocess.run(["osascript", "-e", script], capture_output=True, timeout=5)
            return True
        except Exception as e:
            logger.debug("Desktop banner notification failed: %s", e)
            return False

    def notify(self, message: str, title: str = "AutoTrader Alert") -> None:
        """Dispatches a generic notification to all configured notification channels."""
        if not self.enabled:
            return
        logger.info("Trade Alert [%s]:\n%s", title, message)
        self.send_ntfy(message=message, title=title)
        self.send_imessage(message)
        if self.macos_banner:
            lines = message.strip().split("\n")
            sub = lines[0] if lines else ""
            body = "\n".join(lines[1:]) if len(lines) > 1 else sub
            self.send_macos_banner(title=title, subtitle=sub, body=body)

    def notify_buy(
        self,
        symbol: str,
        price: float,
        notional: float,
        qty: float,
        composite_score: float,
        tech_score: float,
        vol_score: float,
        sent_score: float,
        tradable_cash: float,
        tax_reserve: float,
    ) -> None:
        """Formats and sends a BUY execution alert across all channels."""
        message = (
            f"🟢 BUY ORDER EXECUTED\n"
            f"Pair: {symbol}\n"
            f"Price: ${price:,.2f}\n"
            f"Size: ${notional:,.2f} ({qty:.6f} units)\n"
            f"Signal: BUY (Composite: {composite_score:+.3f})\n"
            f"📊 Factors:\n"
            f"  • Tech: {tech_score:+.3f} | Vol: {vol_score:+.3f} | Sent: {sent_score:+.3f}\n"
            f"💼 Tradable Cash: ${tradable_cash:,.2f}\n"
            f"🏛 Tax Escrow: ${tax_reserve:,.2f}"
        )

        # 1. iOS PWA / Apple Watch Push via ntfy.sh
        self.send_ntfy(
            message=message,
            title=f"🟢 BUY: {symbol} (${notional:,.0f} @ ${price:,.2f})",
            priority="high",
            tags="white_check_mark,moneybag,chart_with_upwards_trend",
        )

        # 2. Local macOS Banner
        self.send_macos_banner(
            title=f"AutoTrader BUY: {symbol}",
            subtitle=f"${notional:,.2f} at ${price:,.2f}",
            body=f"Composite Score: {composite_score:+.3f}",
        )

        # 3. iMessage
        self.send_imessage(message)

    def notify_sell(
        self,
        symbol: str,
        exit_price: float,
        qty: float,
        reason: str,
        gross_pnl: float,
        tax_allocated: float,
        tax_credit: float,
        reserve_after: float,
    ) -> None:
        """Formats and sends a SELL execution alert with tax escrow impact."""
        pnl_sign = "+" if gross_pnl >= 0 else "-"
        pnl_emoji = "📈" if gross_pnl >= 0 else "📉"
        
        tax_detail = (
            f"  • Tax Allocated (30%): +${tax_allocated:,.2f}"
            if gross_pnl >= 0
            else f"  • Tax Credit Applied: -${tax_credit:,.2f}"
        )

        message = (
            f"🔴 SELL ORDER EXECUTED\n"
            f"Pair: {symbol}\n"
            f"Exit Price: ${exit_price:,.2f}\n"
            f"Qty: {qty:.6f} units\n"
            f"Reason: {reason}\n"
            f"{pnl_emoji} Gross Realized PnL: {pnl_sign}${abs(gross_pnl):,.2f}\n"
            f"🏛 Tax Escrow:\n"
            f"{tax_detail}\n"
            f"  • Reserve Balance: ${reserve_after:,.2f}"
        )

        # 1. iOS PWA / Apple Watch Push via ntfy.sh
        priority = "urgent" if gross_pnl < 0 else "high"
        self.send_ntfy(
            message=message,
            title=f"🔴 SELL: {symbol} ({pnl_sign}${abs(gross_pnl):,.2f})",
            priority=priority,
            tags="rotating_light,chart_with_downwards_trend" if gross_pnl < 0 else "money_with_wings,chart_with_upwards_trend",
        )

        # 2. Local macOS Banner
        self.send_macos_banner(
            title=f"AutoTrader SELL: {symbol}",
            subtitle=f"{pnl_emoji} PnL: {pnl_sign}${abs(gross_pnl):,.2f}",
            body=f"Exit: ${exit_price:,.2f} | Reason: {reason}",
        )

        # 3. iMessage
        self.send_imessage(message)

    def notify_wheel_csp_open(
        self,
        underlying: str,
        contract_symbol: str,
        strike: float,
        expiration: str,
        dte: int,
        premium: float,
        contracts: int,
        collateral: float,
        tax_allocated: float,
        tradable_cash: float,
    ) -> None:
        """Alerts when a Cash-Secured Put is sold to open."""
        total_credit = premium * contracts * 100
        message = (
            f"🎡 [WHEEL: CASH-SECURED PUT OPENED]\n"
            f"Underlying: {underlying}\n"
            f"Contract: {contract_symbol}\n"
            f"Strike: ${strike:.2f} | Exp: {expiration} ({dte} DTE)\n"
            f"Contracts: {contracts} ({contracts * 100} shares)\n"
            f"💰 Premium Collected: +${total_credit:,.2f} (${premium:.2f}/share)\n"
            f"🔒 Cash Collateral Locked: ${collateral:,.2f}\n"
            f"🏛 Tax Escrow (30%): +${tax_allocated:,.2f}\n"
            f"💼 Remaining Tradable Cash: ${tradable_cash:,.2f}"
        )

        self.send_ntfy(
            message=message,
            title=f"WHEEL CSP: {underlying} ${strike:.0f}P (+${total_credit:,.0f})",
            priority="high",
            tags="package,moneybag,shield",
        )
        self.send_macos_banner(
            title=f"Wheel CSP Opened: {underlying}",
            subtitle=f"Sold ${strike:.0f} Put for +${total_credit:,.2f}",
            body=f"Collateral: ${collateral:,.2f} | {dte} DTE",
        )
        self.send_imessage(message)

    def notify_wheel_cc_open(
        self,
        underlying: str,
        contract_symbol: str,
        strike: float,
        expiration: str,
        dte: int,
        premium: float,
        contracts: int,
        tax_allocated: float,
    ) -> None:
        """Alerts when a Covered Call is sold to open."""
        total_credit = premium * contracts * 100
        message = (
            f"🎡 [WHEEL: COVERED CALL OPENED]\n"
            f"Underlying: {underlying}\n"
            f"Contract: {contract_symbol}\n"
            f"Strike: ${strike:.2f} | Exp: {expiration} ({dte} DTE)\n"
            f"Contracts: {contracts} ({contracts * 100} shares)\n"
            f"💰 Call Premium Collected: +${total_credit:,.2f} (${premium:.2f}/share)\n"
            f"🏛 Tax Escrow (30%): +${tax_allocated:,.2f}"
        )

        self.send_ntfy(
            message=message,
            title=f"WHEEL CC: {underlying} ${strike:.0f}C (+${total_credit:,.0f})",
            priority="high",
            tags="chart_with_upwards_trend,moneybag",
        )
        self.send_macos_banner(
            title=f"Wheel CC Opened: {underlying}",
            subtitle=f"Sold ${strike:.0f} Call for +${total_credit:,.2f}",
            body=f"{dte} DTE | Exp: {expiration}",
        )
        self.send_imessage(message)

    def notify_wheel_profit_close(
        self,
        underlying: str,
        contract_symbol: str,
        contracts: int,
        open_premium: float,
        close_cost: float,
        net_profit: float,
        pct_profit: float,
    ) -> None:
        """Alerts when an option is bought to close at 50% profit target."""
        message = (
            f"🎯 [WHEEL: 50% PROFIT TARGET HIT]\n"
            f"Contract: {contract_symbol}\n"
            f"Action: BUY TO CLOSE (Early Exit)\n"
            f"Contracts: {contracts}\n"
            f"Initial Credit: ${open_premium:,.2f}\n"
            f"Buyback Cost:   ${close_cost:,.2f}\n"
            f"✨ Net Profit Locked: +${net_profit:,.2f} ({pct_profit:.0%})\n"
            f"Collateral released for next cycle!"
        )

        self.send_ntfy(
            message=message,
            title=f"WHEEL PROFIT: {underlying} (+${net_profit:,.2f})",
            priority="high",
            tags="dart,tada,sparkles",
        )
        self.send_macos_banner(
            title=f"Wheel Profit Locked: {underlying}",
            subtitle=f"+${net_profit:,.2f} ({pct_profit:.0%})",
            body=f"Closed {contract_symbol} early",
        )
        self.send_imessage(message)

    def notify_wheel_assignment(
        self,
        underlying: str,
        strike: float,
        qty: int,
        assignment_type: str,
    ) -> None:
        """Alerts on option assignment (shares acquired or shares called away)."""
        is_put = "PUT" in assignment_type.upper()
        header = "📥 [WHEEL: SHARES ASSIGNED]" if is_put else "📤 [WHEEL: SHARES CALLED AWAY]"
        desc = (
            f"Assigned {qty} shares of {underlying} at ${strike:.2f}. Transitioning to Covered Calls!"
            if is_put
            else f"Called away {qty} shares of {underlying} at ${strike:.2f}. Capital gains secured! Transitioning to Cash-Secured Puts!"
        )

        message = (
            f"{header}\n"
            f"Underlying: {underlying}\n"
            f"Strike Price: ${strike:.2f}\n"
            f"Shares: {qty}\n"
            f"{desc}"
        )

        self.send_ntfy(
            message=message,
            title=f"WHEEL: {underlying} {'Assigned' if is_put else 'Called Away'}",
            priority="high",
            tags="arrows_counterclockwise,repeat",
        )
        self.send_macos_banner(
            title=f"Wheel Assignment: {underlying}",
            subtitle=f"{qty} shares @ ${strike:.2f}",
            body=desc,
        )
        self.send_imessage(message)

    def notify_spread_open(
        self,
        underlying: str,
        short_strike: float,
        long_strike: float,
        expiration: str,
        dte: int,
        net_credit: float,
        collateral_locked: float,
        max_profit: float,
        target_exit_profit: float,
    ) -> None:
        """Alerts when a Defined-Risk Bull Put Credit Spread is opened."""
        message = (
            f"🎯 [SPREAD: BULL PUT OPENED]\n"
            f"Underlying: {underlying}\n"
            f"Strikes: ${short_strike:.2f}P (Short) / ${long_strike:.2f}P (Long)\n"
            f"Expiration: {expiration} ({dte} DTE)\n"
            f"💰 Net Credit Collected: +${max_profit:,.2f} (+${net_credit:.2f}/share)\n"
            f"🔒 Collateral Locked: ${collateral_locked:,.2f}\n"
            f"🎯 50% Profit Exit Target: +${target_exit_profit:,.2f}\n"
            f"🛑 Stop-Loss Multiplier: 2.5x credit"
        )
        self.send_ntfy(
            message=message,
            title=f"SPREAD OPEN: {underlying} ${short_strike:.0f}P/${long_strike:.0f}P (+${max_profit:,.0f})",
            priority="high",
            tags="shield,dart,moneybag",
        )
        self.send_macos_banner(
            title=f"Bull Put Spread Opened: {underlying}",
            subtitle=f"${short_strike:.0f}P/${long_strike:.0f}P (+${max_profit:,.2f})",
            body=f"Collateral: ${collateral_locked:,.2f} | 50% Target: +${target_exit_profit:,.2f}",
        )
        self.send_imessage(message)

    def notify_spread_close(
        self,
        underlying: str,
        short_strike: float,
        long_strike: float,
        reason: str,
        realized_pnl: float,
        tax_allocated: float,
        tax_reserve_after: float,
    ) -> None:
        """Alerts when a Defined-Risk Spread is closed (50% profit target, stop loss, or expiration defense)."""
        sign = "+" if realized_pnl >= 0 else "-"
        abs_pnl = abs(realized_pnl)
        is_win = realized_pnl >= 0
        emoji = "🎯" if is_win else "🛑"
        priority = "high" if is_win else "urgent"
        tags = "dart,tada,money_with_wings" if is_win else "rotating_light,shield"

        tax_line = (
            f"🏛 Tax Escrow (30% Allocated): +${tax_allocated:,.2f}"
            if is_win
            else "🏛 Tax Credit Applied to Reserve"
        )

        message = (
            f"{emoji} [SPREAD: POSITION CLOSED]\n"
            f"Underlying: {underlying} (${short_strike:.2f}P/${long_strike:.2f}P)\n"
            f"Exit Reason: {reason}\n"
            f"💵 Realized PnL: {sign}${abs_pnl:,.2f}\n"
            f"{tax_line}\n"
            f"🏦 Tax Reserve Balance: ${tax_reserve_after:,.2f}\n"
            f"🔓 Collateral Unlocked & Returned to Tradable Cash!"
        )
        self.send_ntfy(
            message=message,
            title=f"SPREAD CLOSED: {underlying} ({sign}${abs_pnl:,.2f})",
            priority=priority,
            tags=tags,
        )
        self.send_macos_banner(
            title=f"Spread Closed: {underlying}",
            subtitle=f"Reason: {reason} | PnL: {sign}${abs_pnl:,.2f}",
            body=f"Tax Reserve: ${tax_reserve_after:,.2f}",
        )
        self.send_imessage(message)

    def notify_daily_recap(
        self,
        date_str: str,
        trades_count: int,
        crypto_diagnostics: list[str],
        wheel_diagnostics: list[str],
        cash: float,
        tradable_cash: float,
        tax_reserve: float,
        spread_diagnostics: Optional[list[str]] = None,
    ) -> None:
        """
        Sends an automated end-of-day daily briefing.
        If trades_count == 0, provides a clear diagnostic breakdown explaining
        why no trades were triggered across all active trading strategies.
        """
        status_line = (
            f"⚡ Today's Trades: {trades_count} executed"
            if trades_count > 0
            else "💤 Today's Trades: 0 New Orders"
        )

        crypto_section = (
            "\n".join(f"• {item}" for item in crypto_diagnostics)
            if crypto_diagnostics
            else "• All quiet"
        )
        wheel_section = (
            "\n".join(f"• {item}" for item in wheel_diagnostics)
            if wheel_diagnostics
            else "• No active wheel orders"
        )

        spread_section = ""
        if spread_diagnostics is not None:
            spread_lines = (
                "\n".join(f"• {item}" for item in spread_diagnostics)
                if spread_diagnostics
                else "• No active spread orders"
            )
            spread_section = f"\n🎯 Defined-Risk Option Spreads:\n{spread_lines}\n"

        reason_header = (
            "\n🔍 WHY NO TRADES WERE TRIGGERED TODAY:\n"
            if trades_count == 0
            else "\n📋 ACTIVITY & STATUS BREAKDOWN:\n"
        )

        message = (
            f"📊 [AUTOTRADER DAILY BRIEFING — {date_str}]\n"
            f"{status_line}\n"
            f"{reason_header}"
            f"🪙 Crypto Momentum Scanner:\n"
            f"{crypto_section}\n\n"
            f"🎡 Multi-Asset Option Wheel:\n"
            f"{wheel_section}\n"
            f"{spread_section}\n"
            f"💰 Portfolio Financials:\n"
            f"• Total Cash:    ${cash:,.2f}\n"
            f"• Tradable Cash: ${tradable_cash:,.2f}\n"
            f"• Tax Escrow:    ${tax_reserve:,.2f}"
        )

        self.send_ntfy(
            message=message,
            title=f"AutoTrader Daily Briefing ({date_str})",
            priority="default",
            tags="bar_chart,memo,clipboard",
        )
        self.send_macos_banner(
            title=f"AutoTrader Daily Briefing — {date_str}",
            subtitle=f"{trades_count} trades today | Cash: ${cash:,.2f}",
            body="Tap to view full daily breakdown.",
        )
        self.send_imessage(message)

    def notify_approval_request(
        self,
        proposal_title: str,
        proposal_message: str,
        action_topic: str = "eckermike87-actions",
        approve_body: str = "APPROVE_5050_BONDS",
        reject_body: str = "REJECT_5050_BONDS",
        approve_label: str = "Approve 50/50 Buy ($40k)",
        reject_label: str = "Reject",
    ) -> bool:
        """
        Sends an interactive notification with Action buttons to iPhone / Apple Watch.
        Tapping a button posts a background HTTP request to the ntfy action topic.
        """
        actions_header = (
            f"http, {approve_label}, https://ntfy.sh/{action_topic}, method=POST, body={approve_body}, clear=true; "
            f"http, {reject_label}, https://ntfy.sh/{action_topic}, method=POST, body={reject_body}, clear=true"
        )
        self.send_ntfy(
            message=proposal_message,
            title=proposal_title,
            priority="urgent",
            tags="bank,moneybag,scales",
            actions=actions_header,
        )
        self.send_macos_banner(
            title=proposal_title,
            subtitle="Action required via ntfy on iPhone",
            body="Check iPhone lock screen for Approve / Reject buttons.",
        )
        self.send_imessage(
            f"⚠️ [{proposal_title}]\n\n{proposal_message}\n\n👉 Tap [Approve] or [Reject] in your ntfy alert."
        )
        return True

    def notify_approval_resolution(
        self,
        title: str,
        message: str,
        approved: bool = True,
    ) -> None:
        """
        Notifies user of the execution or cancellation of an approved trade.
        """
        tags = "white_check_mark,bank" if approved else "x,warning"
        self.send_ntfy(
            message=message,
            title=title,
            priority="default",
            tags=tags,
        )
        self.send_macos_banner(
            title=title,
            subtitle="AutoTrader Capital Manager",
            body=message[:100],
        )
        self.send_imessage(f"📢 [{title}]\n{message}")


