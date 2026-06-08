import csv
import os
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import load_dotenv

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    GetOrdersRequest,
    LimitOrderRequest,
    MarketOrderRequest,
    StopLossRequest,
    TakeProfitRequest,
    TrailingStopOrderRequest,
)
from alpaca.trading.enums import (
    OrderClass,
    OrderSide,
    OrderStatus,
    QueryOrderStatus,
    TimeInForce,
)

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

# Optional Alpaca news client.
# If your alpaca-py version does not support this exact import,
# the bot will keep running and treat news as unavailable.
try:
    from alpaca.data.historical.news import NewsClient
    from alpaca.data.requests import NewsRequest

    NEWS_IMPORT_AVAILABLE = True
except Exception:
    NewsClient = None
    NewsRequest = None
    NEWS_IMPORT_AVAILABLE = False


# ============================================================
# 1. ENVIRONMENT / ACCOUNT
# ============================================================

load_dotenv()

API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

if not API_KEY or not SECRET_KEY:
    raise ValueError("Missing Alpaca API keys. Check your .env file.")

# Hard safety lock.
# This bot is intentionally paper-only.
PAPER = True
LIVE_TRADING_ALLOWED = False

if not PAPER and not LIVE_TRADING_ALLOWED:
    raise RuntimeError("Live trading is disabled. This bot is configured for paper trading only.")


# ============================================================
# 2. BOT CONFIG
# ============================================================

EASTERN = ZoneInfo("America/New_York")

# Start small. Do not scan a huge watchlist yet.
SYMBOLS = ["SPY", "QQQ", "TSLA", "AMD", "PLTR", "SOFI", "MARA", "META", "SNDK", "MSTR", "MU", "IONQ", "QBTS", "SMCI", "RIVN", "PANW", "SNOW", "RPGL", "SIDU", "MNTS", "YSS", "BBAI", "RDW", "WMT", "NVTS", "RGTI", "SOUN", "USAR", "CRWV", "VOO"]

# IEX avoids recent SIP subscription errors on many free Alpaca accounts.
DATA_FEED = DataFeed.IEX

# Candle settings.
TIMEFRAME = TimeFrame.Minute
BARS_LOOKBACK_MINUTES = 240

# Trade sizing.
# Each entry targets this percent of current account equity.
# Example: 0.05 = 5% of a $100,000 paper account = about $5,000 per trade.
POSITION_SIZE_PCT = 0.05
MIN_ENTRY_QTY = 1
ALLOW_OVERSIZED_MIN_SHARE = False
MAX_OPEN_POSITIONS = 5

# Daily risk limits.
MAX_TRADES_PER_DAY = 200
MAX_DAILY_LOSS_DOLLARS = 500.00

# No-trade windows.
NO_TRADE_FIRST_MINUTES = 10
NO_TRADE_LAST_MINUTES = 60

# End-of-day flatten.
# Since this is a day-trading bot, close remaining positions shortly before market close.
CLOSE_POSITIONS_BEFORE_MARKET_CLOSE = True
FLATTEN_MINUTES_BEFORE_CLOSE = 10

# Late-day entry protection.
# No new entries unless there is at least this much time left before the close.
# 60 minutes blocks new trades after 3:00 PM ET while still managing existing positions.
MIN_MINUTES_BEFORE_CLOSE_TO_ENTER = 60

# Strategy indicators.
FAST_EMA = 9
SLOW_EMA = 21
RSI_PERIOD = 14
VOLUME_AVG_PERIOD = 20

# Normal filters.
USE_VWAP_FILTER = True
USE_RSI_FILTER = True
USE_VOLUME_CONFIRMATION = True

# VWAP tolerance.
# 0.010 = allow entries up to 1.00% below VWAP.
# Slightly tighter than v10 to avoid weak below-VWAP chop.
VWAP_TOLERANCE_PCT = 0.010

# Normal RSI rules.
MIN_BUY_RSI = 50
MAX_BUY_RSI = 70

# Normal volume confirmation.
# Current candle volume must be at least avg volume * multiplier.
# IEX volume is partial, so this is intentionally loose for paper testing.
VOLUME_MULTIPLIER = 0.75

# Entry quality filters.
# These are designed to avoid weak/choppy EMA crosses that quickly reverse.
REQUIRE_CLOSE_ABOVE_FAST_EMA = True
REQUIRE_EMA_SLOPE_UP = True
EMA_SLOPE_LOOKBACK_BARS = 3

# Stop-loss protection / cooldown rules.
MAX_CONSECUTIVE_STOP_LOSSES = 2
PAUSE_MINUTES_AFTER_MAX_STOP_LOSSES = 30
COOLDOWN_MINUTES_AFTER_STOP = 30

# News-aware behavior.
# "BLOCK"    = skip trades if recent news exists
# "IGNORE"   = ignore news completely
# "CATALYST" = allow news trades, but require stronger confirmation
USE_NEWS_AWARE_MODE = True
NEWS_MODE = "CATALYST"
NEWS_LOOKBACK_MINUTES = 30
NEWS_FAIL_CLOSED = False

# Stronger confirmation required when recent news exists.
NEWS_VOLUME_MULTIPLIER = 1.00
NEWS_MIN_RSI = 50
NEWS_MAX_RSI = 80

# Slippage control.
# For a buy limit order, the bot will not chase more than this percent
# above the most recent close.
MAX_ENTRY_SLIPPAGE_PCT = 0.002  # 0.20%

# Spread filter.
# Blocks entries when the bid/ask spread is too wide, which usually means
# illiquidity, stale quotes, or unreliable IEX quote behavior.
MAX_SPREAD_PCT = 0.0075  # 0.75%

# Managed exit settings.
# Buy entries remain LIMIT orders.
# Exits are managed by the bot with MARKET sell orders when TP/SL conditions trigger.
# Important: managed exits only work while this bot is running.
# EMA sell exits are disabled so tiny 1-minute EMA chop does not exit early.
USE_BRACKET_ORDERS = False
USE_MANAGED_MARKET_EXITS = True
USE_EMA_EXIT = False

# 1:2 risk/reward target.
# Risk = STOP_LOSS_PCT. Reward = STOP_LOSS_PCT * RISK_REWARD_RATIO.
# Example: 0.40% stop-loss => 0.80% take-profit.
RISK_REWARD_RATIO = 2.0
STOP_LOSS_PCT = 0.004         # -0.40% risk
TAKE_PROFIT_PCT = STOP_LOSS_PCT * RISK_REWARD_RATIO  # +0.80% reward
STOP_LIMIT_EXTRA_PCT = 0.001  # retained for reference; not used when managed market exits are enabled

# Managed profit protection / trailing stop logic.
# This is not active immediately at entry. It only activates after the trade
# moves in our favor by PROFIT_PROTECTION_TRIGGER_PCT, then protects gains
# with a breakeven buffer and a loose trailing stop based on highest bar close.
USE_PROFIT_PROTECTION = True
PROFIT_PROTECTION_TRIGGER_PCT = 0.004    # +0.40% = +1R with the current stop
BREAKEVEN_BUFFER_PCT = 0.0005            # once triggered, protect roughly +0.05%
TRAILING_STOP_AFTER_TRIGGER_PCT = 0.004  # trail 0.40% below highest bar close

# Time-stop logic.
# If a trade sits for too long and is not working, exit so capital is freed
# and the trade does not slowly drift into the hard stop.
USE_TIME_STOP = True
MAX_HOLD_MINUTES = 30
MIN_PROFIT_TO_KEEP_AFTER_TIME_PCT = 0.001  # require at least +0.10% after max hold

# Optional trailing stop mode.
# Leave disabled while using managed TP/SL market exits.
USE_TRAILING_STOP = False
TRAIL_PERCENT = 0.30  # means 0.30%, not 30%

if USE_BRACKET_ORDERS and USE_TRAILING_STOP:
    raise ValueError("Choose either bracket orders or trailing stop mode, not both.")

if USE_MANAGED_MARKET_EXITS and USE_TRAILING_STOP:
    raise ValueError("Choose either managed market exits or trailing stop mode, not both.")

# Order management.
ORDER_FILL_TIMEOUT_SECONDS = 60
ORDER_POLL_SECONDS = 3

# Main scanner interval.
# 30 seconds = checks twice per minute while still using 1-minute candles.
SCAN_INTERVAL_SECONDS = 30

# Prevent duplicate entry attempts on the same symbol/candle when scanning every 30 seconds.
ONE_ENTRY_ATTEMPT_PER_CANDLE = True

# Versioned journals.
# Keep logs versioned so old bot runs do not get mixed with new bot behavior.
BOT_VERSION = "v14"
TRADE_JOURNAL_PATH = Path(f"trades_{BOT_VERSION}.csv")

# Signal journal.
# This logs every symbol evaluation, even when no trade happens.
SIGNAL_JOURNAL_PATH = Path(f"signals_{BOT_VERSION}.csv")

# Position journal.
# This logs open-position status every scan so open trades are easier to review.
POSITION_JOURNAL_PATH = Path(f"positions_{BOT_VERSION}.csv")

# Safety stubs. Keep these disabled for now.
OPTIONS_TRADING_ENABLED = False
AI_DECISION_ENABLED = False


# ============================================================
# 3. CLIENTS
# ============================================================

trading_client = TradingClient(API_KEY, SECRET_KEY, paper=PAPER)
data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

if NEWS_IMPORT_AVAILABLE:
    news_client = NewsClient(API_KEY, SECRET_KEY)
else:
    news_client = None


# ============================================================
# 4. STATE
# ============================================================

@dataclass
class BotState:
    current_day: date
    starting_equity: float
    trades_today: int = 0
    attempted_entry_candles: dict[str, str] = field(default_factory=dict)
    consecutive_stop_losses: int = 0
    paused_until: datetime | None = None
    symbol_cooldowns: dict[str, datetime] = field(default_factory=dict)
    position_entry_times: dict[str, datetime] = field(default_factory=dict)
    highest_close_by_symbol: dict[str, float] = field(default_factory=dict)


# ============================================================
# 5. UTILITY FUNCTIONS
# ============================================================

def now_et() -> datetime:
    return datetime.now(EASTERN)


def round_price(price: float) -> float:
    return round(float(price), 2)


def ensure_trade_journal_exists() -> None:
    if TRADE_JOURNAL_PATH.exists():
        return

    with TRADE_JOURNAL_PATH.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "timestamp",
            "symbol",
            "signal",
            "action",
            "qty",
            "limit_price",
            "take_profit_price",
            "stop_loss_price",
            "order_id",
            "order_status",
            "filled_qty",
            "filled_avg_price",
            "equity",
            "daily_pnl",
            "reason",
            "error",
        ])


def log_trade(
    symbol: str,
    signal: str,
    action: str,
    qty=None,
    limit_price=None,
    take_profit_price=None,
    stop_loss_price=None,
    order_id=None,
    order_status=None,
    filled_qty=None,
    filled_avg_price=None,
    equity=None,
    daily_pnl=None,
    reason="",
    error="",
) -> None:
    ensure_trade_journal_exists()

    with TRADE_JOURNAL_PATH.open("a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            now_et().isoformat(),
            symbol,
            signal,
            action,
            qty,
            limit_price,
            take_profit_price,
            stop_loss_price,
            order_id,
            order_status,
            filled_qty,
            filled_avg_price,
            equity,
            daily_pnl,
            reason,
            error,
        ])


def ensure_signal_journal_exists() -> None:
    if SIGNAL_JOURNAL_PATH.exists():
        return

    with SIGNAL_JOURNAL_PATH.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "timestamp",
            "symbol",
            "signal",
            "passed",
            "last_close",
            "reason",
            "ema_fast",
            "ema_slow",
            "vwap",
            "rsi",
            "volume",
            "avg_volume",
        ])


def log_signal(
    symbol: str,
    signal: str,
    passed: bool,
    last_close=None,
    reason="",
    bars: pd.DataFrame | None = None,
) -> None:
    ensure_signal_journal_exists()

    ema_fast = None
    ema_slow = None
    vwap = None
    rsi = None
    volume = None
    avg_volume = None

    try:
        if bars is not None and not bars.empty:
            indicator_bars = add_indicators(bars)
            current = indicator_bars.iloc[-1]

            ema_fast = float(current["ema_fast"])
            ema_slow = float(current["ema_slow"])
            vwap = float(current["vwap"])
            rsi = float(current["rsi"])
            volume = float(current["volume"])
            avg_volume = float(current["avg_volume"])
    except Exception as e:
        reason = f"{reason}; signal indicator logging error: {e}"

    with SIGNAL_JOURNAL_PATH.open("a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            now_et().isoformat(),
            symbol,
            signal,
            passed,
            last_close,
            reason,
            ema_fast,
            ema_slow,
            vwap,
            rsi,
            volume,
            avg_volume,
        ])




def ensure_position_journal_exists() -> None:
    if POSITION_JOURNAL_PATH.exists():
        return

    with POSITION_JOURNAL_PATH.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "timestamp",
            "symbol",
            "qty",
            "avg_entry_price",
            "latest_bar_close",
            "take_profit_price",
            "stop_loss_price",
            "unrealized_estimated_pnl",
            "distance_to_tp_pct",
            "distance_to_sl_pct",
            "reason",
        ])


def log_position_snapshot(
    symbol: str,
    qty: float,
    avg_entry_price: float,
    latest_bar_close=None,
    take_profit_price=None,
    stop_loss_price=None,
    unrealized_estimated_pnl=None,
    distance_to_tp_pct=None,
    distance_to_sl_pct=None,
    reason="",
) -> None:
    ensure_position_journal_exists()

    with POSITION_JOURNAL_PATH.open("a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            now_et().isoformat(),
            symbol,
            qty,
            avg_entry_price,
            latest_bar_close,
            take_profit_price,
            stop_loss_price,
            unrealized_estimated_pnl,
            distance_to_tp_pct,
            distance_to_sl_pct,
            reason,
        ])


def log_open_position_snapshots() -> None:
    """
    Logs open-position status once per main loop.

    This makes it easier to review trades that were still open when the bot
    stopped, instead of only seeing submitted/filled orders in trades.csv.
    """
    positions = get_positions()

    for position in positions:
        symbol = position.symbol

        if symbol not in SYMBOLS:
            continue

        try:
            qty = float(position.qty)
            avg_entry_price = float(position.avg_entry_price)

            if qty <= 0:
                continue

            take_profit_price = round_price(avg_entry_price * (1 + TAKE_PROFIT_PCT))
            stop_loss_price = round_price(avg_entry_price * (1 - STOP_LOSS_PCT))
            latest_bar_close, price_reason = get_latest_bar_close_for_exit(symbol)

            if latest_bar_close is None:
                log_position_snapshot(
                    symbol=symbol,
                    qty=qty,
                    avg_entry_price=avg_entry_price,
                    take_profit_price=take_profit_price,
                    stop_loss_price=stop_loss_price,
                    reason=price_reason,
                )
                continue

            unrealized_estimated_pnl = (latest_bar_close - avg_entry_price) * qty
            distance_to_tp_pct = (take_profit_price - latest_bar_close) / latest_bar_close
            distance_to_sl_pct = (latest_bar_close - stop_loss_price) / latest_bar_close

            log_position_snapshot(
                symbol=symbol,
                qty=qty,
                avg_entry_price=avg_entry_price,
                latest_bar_close=latest_bar_close,
                take_profit_price=take_profit_price,
                stop_loss_price=stop_loss_price,
                unrealized_estimated_pnl=unrealized_estimated_pnl,
                distance_to_tp_pct=distance_to_tp_pct,
                distance_to_sl_pct=distance_to_sl_pct,
                reason=price_reason,
            )

        except Exception as e:
            log_position_snapshot(
                symbol=symbol,
                qty=getattr(position, "qty", None),
                avg_entry_price=getattr(position, "avg_entry_price", None),
                reason=f"Position snapshot error: {e}",
            )

def get_account():
    return trading_client.get_account()


def get_equity() -> float:
    return float(get_account().equity)


def get_positions():
    return trading_client.get_all_positions()


def open_position_symbols() -> set[str]:
    return {position.symbol for position in get_positions()}


def has_position(symbol: str) -> bool:
    return symbol in open_position_symbols()


def count_open_positions() -> int:
    return len(get_positions())


def get_position_qty(symbol: str) -> float:
    for position in get_positions():
        if position.symbol == symbol:
            return float(position.qty)
    return 0.0


def has_open_order_for_symbol(symbol: str) -> bool:
    request = GetOrdersRequest(
        status=QueryOrderStatus.OPEN,
        symbols=[symbol],
        limit=50,
    )
    orders = trading_client.get_orders(filter=request)
    return len(orders) > 0


def reset_daily_state_if_needed(state: BotState) -> BotState:
    today = now_et().date()

    if today != state.current_day:
        new_equity = get_equity()

        print(f"{now_et()} | New trading day detected. Resetting daily counters.")
        print(f"{now_et()} | New starting equity: {new_equity}")

        log_trade(
            symbol="ALL",
            signal="RESET",
            action="DAILY_RESET",
            equity=new_equity,
            daily_pnl=0,
            reason="Automatic daily reset",
        )

        return BotState(
            current_day=today,
            starting_equity=new_equity,
            trades_today=0,
        )

    return state


# ============================================================
# 6. MARKET HOURS / RISK
# ============================================================

def is_market_time() -> bool:
    current = now_et()

    market_open = current.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = current.replace(hour=16, minute=0, second=0, microsecond=0)

    return market_open <= current <= market_close


def is_no_trade_window() -> tuple[bool, str]:
    current = now_et()

    market_open = current.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = current.replace(hour=16, minute=0, second=0, microsecond=0)

    first_trade_time = market_open + timedelta(minutes=NO_TRADE_FIRST_MINUTES)
    latest_entry_time = market_close - timedelta(minutes=MIN_MINUTES_BEFORE_CLOSE_TO_ENTER)

    if current < first_trade_time:
        return True, f"Inside first {NO_TRADE_FIRST_MINUTES}-minute no-trade window"

    if current >= latest_entry_time:
        return True, (
            f"Inside late-day no-entry window: fewer than "
            f"{MIN_MINUTES_BEFORE_CLOSE_TO_ENTER} minutes before close"
        )

    return False, ""


def risk_check(state: BotState, current_equity: float) -> tuple[bool, str, float]:
    daily_pnl = current_equity - state.starting_equity

    if state.paused_until is not None and now_et() < state.paused_until:
        return False, f"Paused after consecutive stop-losses until {state.paused_until}", daily_pnl

    if daily_pnl <= -MAX_DAILY_LOSS_DOLLARS:
        return False, f"Max daily loss hit: {daily_pnl:.2f}", daily_pnl

    if state.trades_today >= MAX_TRADES_PER_DAY:
        return False, "Max trades per day reached", daily_pnl

    if count_open_positions() >= MAX_OPEN_POSITIONS:
        return False, "Max open positions reached", daily_pnl

    blocked, reason = is_no_trade_window()
    if blocked:
        return False, reason, daily_pnl

    return True, "Risk check passed", daily_pnl


# ============================================================
# 7. DATA / INDICATORS
# ============================================================

def get_recent_bars(symbol: str, minutes_back: int = BARS_LOOKBACK_MINUTES) -> pd.DataFrame:
    end = now_et()
    start = end - timedelta(minutes=minutes_back)

    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TIMEFRAME,
        start=start,
        end=end,
        feed=DATA_FEED,
    )

    bars = data_client.get_stock_bars(request).df

    if bars.empty:
        return pd.DataFrame()

    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.loc[symbol]

    return bars


def get_latest_quote(symbol: str):
    request = StockLatestQuoteRequest(
        symbol_or_symbols=symbol,
        feed=DATA_FEED,
    )
    quotes = data_client.get_stock_latest_quote(request)
    return quotes[symbol]


def add_indicators(bars: pd.DataFrame) -> pd.DataFrame:
    bars = bars.copy()

    bars["ema_fast"] = bars["close"].ewm(span=FAST_EMA, adjust=False).mean()
    bars["ema_slow"] = bars["close"].ewm(span=SLOW_EMA, adjust=False).mean()

    typical_price = (bars["high"] + bars["low"] + bars["close"]) / 3
    cumulative_pv = (typical_price * bars["volume"]).cumsum()
    cumulative_volume = bars["volume"].cumsum()
    bars["vwap"] = cumulative_pv / cumulative_volume

    delta = bars["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(RSI_PERIOD).mean()
    avg_loss = loss.rolling(RSI_PERIOD).mean()

    rs = avg_gain / avg_loss
    bars["rsi"] = 100 - (100 / (1 + rs))

    bars["avg_volume"] = bars["volume"].rolling(VOLUME_AVG_PERIOD).mean()

    return bars


def calculate_base_signal(bars: pd.DataFrame) -> tuple[str, str]:
    if bars.empty or len(bars) < max(SLOW_EMA, RSI_PERIOD, VOLUME_AVG_PERIOD) + 5:
        return "HOLD", "Not enough bars"

    bars = add_indicators(bars)

    previous = bars.iloc[-2]
    current = bars.iloc[-1]

    crossed_above = (
        previous["ema_fast"] <= previous["ema_slow"]
        and current["ema_fast"] > current["ema_slow"]
    )

    crossed_below = (
        previous["ema_fast"] >= previous["ema_slow"]
        and current["ema_fast"] < current["ema_slow"]
    )

    if crossed_above:
        return "BUY", "EMA cross above"

    if crossed_below:
        return "SELL", "EMA cross below"

    return "HOLD", "No EMA crossover"


# ============================================================
# 8. NEWS-AWARE FILTER
# ============================================================

def get_recent_news_status(symbol: str) -> tuple[bool, str]:
    """
    Returns:
    - has_news: True/False
    - reason: explanation

    NEWS_MODE behavior:
    - BLOCK: news blocks the trade
    - IGNORE: news is ignored
    - CATALYST: news requires stronger RSI/volume confirmation
    """
    if not USE_NEWS_AWARE_MODE or NEWS_MODE == "IGNORE":
        return False, "News ignored"

    if not NEWS_IMPORT_AVAILABLE or news_client is None:
        if NEWS_FAIL_CLOSED:
            raise RuntimeError("News client unavailable and NEWS_FAIL_CLOSED=True")
        return False, "News client unavailable; treating as no-news"

    try:
        end = now_et()
        start = end - timedelta(minutes=NEWS_LOOKBACK_MINUTES)

        request = NewsRequest(
            symbols=symbol,
            start=start,
            end=end,
            limit=10,
        )

        news = news_client.get_news(request)

        articles = []

        # alpaca-py may return different object shapes depending on version.
        # Some versions return a NewsSet object, which does not support len(news).
        if hasattr(news, "data"):
            articles = news.data
        elif isinstance(news, dict):
            articles = news.get("news", []) or news.get("data", [])
        else:
            try:
                articles = list(news)
            except TypeError:
                articles = []

        if len(articles) > 0:
            return True, f"Recent news found for {symbol}"

        return False, "No recent news found"

    except Exception as e:
        if NEWS_FAIL_CLOSED:
            raise RuntimeError(f"News filter error: {e}")

        return False, f"News error; treating as no-news because NEWS_FAIL_CLOSED=False: {e}"


def filters_pass(symbol: str, bars: pd.DataFrame, signal: str) -> tuple[bool, str]:
    if signal != "BUY":
        return True, "Filters only applied to BUY entries"

    bars = add_indicators(bars)
    current = bars.iloc[-1]

    close = float(current["close"])
    vwap = float(current["vwap"])
    rsi = float(current["rsi"])
    volume = float(current["volume"])
    avg_volume = float(current["avg_volume"])

    has_news, news_reason = get_recent_news_status(symbol)

    if has_news and NEWS_MODE == "BLOCK":
        return False, f"News block active: {news_reason}"

    vwap_floor = vwap * (1 - VWAP_TOLERANCE_PCT)

    if USE_VWAP_FILTER and close < vwap_floor:
        return False, (
            f"VWAP filter failed: close {close:.2f} < VWAP floor {vwap_floor:.2f} "
            f"actual VWAP {vwap:.2f}, tolerance {VWAP_TOLERANCE_PCT:.2%}"
        )

    if REQUIRE_CLOSE_ABOVE_FAST_EMA and close <= float(current["ema_fast"]):
        return False, (
            f"Close-above-fast-EMA filter failed: close {close:.2f} <= "
            f"fast EMA {float(current['ema_fast']):.2f}"
        )

    if REQUIRE_EMA_SLOPE_UP:
        if len(bars) <= EMA_SLOPE_LOOKBACK_BARS:
            return False, "EMA slope filter failed: not enough bars for slope check"
        prior_fast_ema = float(bars.iloc[-1 - EMA_SLOPE_LOOKBACK_BARS]["ema_fast"])
        current_fast_ema = float(current["ema_fast"])
        if current_fast_ema <= prior_fast_ema:
            return False, (
                f"EMA slope filter failed: fast EMA {current_fast_ema:.2f} "
                f"<= fast EMA {EMA_SLOPE_LOOKBACK_BARS} bars ago {prior_fast_ema:.2f}"
            )

    if has_news and NEWS_MODE == "CATALYST":
        min_rsi = NEWS_MIN_RSI
        max_rsi = NEWS_MAX_RSI
        volume_multiplier = NEWS_VOLUME_MULTIPLIER
        mode_reason = "News catalyst mode"
    else:
        min_rsi = MIN_BUY_RSI
        max_rsi = MAX_BUY_RSI
        volume_multiplier = VOLUME_MULTIPLIER
        mode_reason = "Normal mode"

    if USE_RSI_FILTER and not (min_rsi <= rsi <= max_rsi):
        return False, (
            f"{mode_reason} RSI filter failed: RSI {rsi:.2f} not between "
            f"{min_rsi} and {max_rsi}. {news_reason}"
        )

    if USE_VOLUME_CONFIRMATION:
        required_volume = avg_volume * volume_multiplier

        if volume < required_volume:
            return False, (
                f"{mode_reason} volume filter failed: {volume:.0f} < {required_volume:.0f}. "
                f"{news_reason}"
            )

    return True, (
        f"{mode_reason} filters passed. {news_reason}. "
        f"Close {close:.2f}, VWAP {vwap:.2f}, RSI {rsi:.2f}, "
        f"volume {volume:.0f}, avg volume {avg_volume:.0f}"
    )


# ============================================================
# 9. AI DECISION STUB
# ============================================================

def ai_decision_veto(symbol: str, signal: str, reason: str, bars: pd.DataFrame) -> tuple[bool, str]:
    """
    Safety design:
    AI is not allowed to create trades.
    At most, AI can veto a trade that the hard-coded strategy already found.

    Right now, this is disabled.
    """
    if not AI_DECISION_ENABLED:
        return True, "AI decision disabled"

    # Placeholder only.
    # Do not connect an LLM to order execution without strict guardrails.
    return True, "AI did not veto"


# ============================================================
# 10. OPTIONS STUB
# ============================================================

def options_trading_stub() -> None:
    """
    Options trading intentionally disabled.

    Options require contract symbols, options permissions, options-specific
    market data, and different risk logic.
    """
    if OPTIONS_TRADING_ENABLED:
        raise NotImplementedError(
            "Options trading is intentionally not implemented in this starter bot."
        )


# ============================================================
# 11. ORDER PRICING
# ============================================================

def calculate_entry_limit_price(symbol: str, last_close: float) -> tuple[float | None, str]:
    """
    Entry pricing, slippage control, and spread control.

    - Pull latest bid/ask.
    - Block entries when the bid/ask spread is too wide.
    - For a buy, prefer the ask if it is within the allowed slippage cap.
    - If ask is missing/bad, use last close with a small cap.
    """
    max_allowed_price = last_close * (1 + MAX_ENTRY_SLIPPAGE_PCT)

    try:
        quote = get_latest_quote(symbol)
        bid_price = float(quote.bid_price)
        ask_price = float(quote.ask_price)

        if ask_price <= 0:
            fallback = round_price(max_allowed_price)
            return fallback, f"Bad ask; using capped fallback limit {fallback}"

        if bid_price > 0 and ask_price > bid_price:
            midpoint = (bid_price + ask_price) / 2
            spread_pct = (ask_price - bid_price) / midpoint if midpoint > 0 else 999

            if spread_pct > MAX_SPREAD_PCT:
                return None, (
                    f"Spread filter blocked entry: bid {bid_price:.2f}, ask {ask_price:.2f}, "
                    f"spread {spread_pct:.2%} > max {MAX_SPREAD_PCT:.2%}"
                )
        elif bid_price <= 0:
            # Do not hard-block on bad bid because some IEX quotes can be incomplete,
            # but keep the reason visible in the accepted/blocked entry log.
            spread_note = "Bad/missing bid; spread filter skipped"
        else:
            spread_note = "Bid/ask spread not positive; spread filter skipped"

        if ask_price > max_allowed_price:
            return None, (
                f"Slippage control blocked entry: ask {ask_price:.2f} "
                f"> max allowed {max_allowed_price:.2f}"
            )

        if bid_price > 0 and ask_price > bid_price:
            midpoint = (bid_price + ask_price) / 2
            spread_pct = (ask_price - bid_price) / midpoint if midpoint > 0 else 0
            spread_note = f"spread {spread_pct:.2%} within max {MAX_SPREAD_PCT:.2%}"

        return round_price(ask_price), (
            f"Using ask price {ask_price:.2f} within slippage cap; {spread_note}"
        )

    except Exception as e:
        fallback = round_price(max_allowed_price)
        return fallback, f"Quote error; using fallback capped limit {fallback}: {e}"


def calculate_position_qty(current_equity: float, entry_limit_price: float) -> tuple[int | None, str]:
    """
    Percent-of-equity position sizing.

    Targets POSITION_SIZE_PCT of current account equity using whole shares.
    Example: $100,000 equity, 1% target, $5 stock => about 200 shares.
    """
    if entry_limit_price <= 0:
        return None, f"Invalid entry limit price for sizing: {entry_limit_price}"

    target_notional = current_equity * POSITION_SIZE_PCT
    raw_qty = int(target_notional // entry_limit_price)

    if raw_qty >= MIN_ENTRY_QTY:
        actual_notional = raw_qty * entry_limit_price
        return raw_qty, (
            f"Position sizing: equity {current_equity:.2f}, "
            f"target {POSITION_SIZE_PCT:.2%} = {target_notional:.2f}, "
            f"entry {entry_limit_price:.2f}, qty {raw_qty}, "
            f"actual notional {actual_notional:.2f}"
        )

    if ALLOW_OVERSIZED_MIN_SHARE:
        actual_notional = MIN_ENTRY_QTY * entry_limit_price
        return MIN_ENTRY_QTY, (
            f"Position sizing oversized minimum-share allowed: equity {current_equity:.2f}, "
            f"target {POSITION_SIZE_PCT:.2%} = {target_notional:.2f}, "
            f"entry {entry_limit_price:.2f}, qty {MIN_ENTRY_QTY}, "
            f"actual notional {actual_notional:.2f}"
        )

    return None, (
        f"Position sizing blocked: equity {current_equity:.2f}, "
        f"target {POSITION_SIZE_PCT:.2%} = {target_notional:.2f}, "
        f"entry {entry_limit_price:.2f}, calculated qty {raw_qty}. "
        f"Enable ALLOW_OVERSIZED_MIN_SHARE to buy minimum share."
    )


def get_current_exit_reference_price(symbol: str, fallback_last_close: float | None = None) -> tuple[float | None, str]:
    """
    Uses the latest quote bid as the exit trigger reference.

    This avoids relying only on the most recent completed candle when deciding
    whether take-profit or stop-loss has been reached. For a long position,
    bid is the realistic side to watch for selling.
    """
    try:
        quote = get_latest_quote(symbol)
        bid_price = float(quote.bid_price)

        if bid_price > 0:
            return bid_price, f"Using latest bid {bid_price:.2f}"

        if fallback_last_close is not None:
            return float(fallback_last_close), f"Bad bid; using fallback last close {fallback_last_close:.2f}"

        return None, "Bad bid and no fallback price available"

    except Exception as e:
        if fallback_last_close is not None:
            return float(fallback_last_close), f"Quote error; using fallback last close {fallback_last_close:.2f}: {e}"

        return None, f"Quote error and no fallback price available: {e}"


def get_latest_bar_close_for_exit(symbol: str) -> tuple[float | None, str]:
    """
    Uses the latest returned 1-minute bar close as the managed exit trigger.

    This is intentionally more conservative than triggering exits from the latest
    IEX bid, because IEX quotes can occasionally be thin/stale and far away from
    the surrounding bar prices. If the bar close confirms TP/SL, the bot still
    exits with a MARKET sell order.
    """
    try:
        bars = get_recent_bars(symbol, minutes_back=10)

        if bars.empty:
            return None, "No recent bars returned for managed exit check"

        latest_close = float(bars["close"].iloc[-1])
        latest_bar_time = bars.index[-1]
        return latest_close, f"Using latest 1-minute bar close {latest_close:.2f} at {latest_bar_time}"

    except Exception as e:
        return None, f"Bar-close exit check error: {e}"

def calculate_exit_prices(entry_limit_price: float) -> tuple[float, float, float]:
    take_profit_price = round_price(entry_limit_price * (1 + TAKE_PROFIT_PCT))
    stop_loss_price = round_price(entry_limit_price * (1 - STOP_LOSS_PCT))
    stop_loss_limit_price = round_price(stop_loss_price * (1 - STOP_LIMIT_EXTRA_PCT))

    return take_profit_price, stop_loss_price, stop_loss_limit_price


# ============================================================
# 12. ORDER SUBMISSION
# ============================================================

def submit_bracket_limit_buy(symbol: str, qty: int, entry_limit_price: float):
    take_profit_price, stop_loss_price, stop_loss_limit_price = calculate_exit_prices(entry_limit_price)

    order_request = LimitOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY,
        limit_price=entry_limit_price,
        order_class=OrderClass.BRACKET,
        take_profit=TakeProfitRequest(
            limit_price=take_profit_price,
        ),
        stop_loss=StopLossRequest(
            stop_price=stop_loss_price,
            limit_price=stop_loss_limit_price,
        ),
    )

    order = trading_client.submit_order(order_data=order_request)

    return order, take_profit_price, stop_loss_price


def submit_plain_limit_buy(symbol: str, qty: int, entry_limit_price: float):
    order_request = LimitOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY,
        limit_price=entry_limit_price,
    )

    return trading_client.submit_order(order_data=order_request)


def submit_trailing_stop_sell(symbol: str, qty: float):
    order_request = TrailingStopOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.SELL,
        time_in_force=TimeInForce.DAY,
        trail_percent=TRAIL_PERCENT,
    )

    return trading_client.submit_order(order_data=order_request)


def wait_for_fill(order_id: str, timeout_seconds: int = ORDER_FILL_TIMEOUT_SECONDS):
    started = time.time()

    while time.time() - started < timeout_seconds:
        order = trading_client.get_order_by_id(order_id)

        if order.status == OrderStatus.FILLED:
            return order, True

        if order.status in [
            OrderStatus.CANCELED,
            OrderStatus.EXPIRED,
            OrderStatus.REJECTED,
        ]:
            return order, False

        time.sleep(ORDER_POLL_SECONDS)

    order = trading_client.get_order_by_id(order_id)

    try:
        trading_client.cancel_order_by_id(order_id)
        print(f"{now_et()} | Fill timeout. Cancel request submitted for order {order_id}.")
    except Exception as e:
        print(f"{now_et()} | Fill timeout, but cancel failed for order {order_id}: {e}")

    return order, False


def submit_market_sell(symbol: str, qty: float):
    """
    Market sell exit used for take-profit, stop-loss, and EMA exit signals.
    Entry orders remain limit orders.
    """
    order_request = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.SELL,
        time_in_force=TimeInForce.DAY,
    )

    return trading_client.submit_order(order_data=order_request)


def wait_for_market_sell_fill(order_id: str, timeout_seconds: int = 30):
    """
    Market sells should normally fill quickly. This confirms and returns the
    final order object so trades.csv can record actual fill details.
    """
    started = time.time()

    while time.time() - started < timeout_seconds:
        order = trading_client.get_order_by_id(order_id)

        if order.status == OrderStatus.FILLED:
            return order, True

        if order.status in [
            OrderStatus.CANCELED,
            OrderStatus.EXPIRED,
            OrderStatus.REJECTED,
        ]:
            return order, False

        time.sleep(ORDER_POLL_SECONDS)

    return trading_client.get_order_by_id(order_id), False


def managed_market_sell_and_log(
    symbol: str,
    signal: str,
    qty: float,
    take_profit_price: float | None,
    stop_loss_price: float | None,
    current_equity: float,
    daily_pnl: float,
    reason: str,
) -> tuple[bool, object]:
    order = submit_market_sell(symbol, qty)

    print(f"{now_et()} | {symbol} | {signal} MARKET sell submitted: {order.id}")

    log_trade(
        symbol=symbol,
        signal=signal,
        action="MARKET_SELL_SUBMITTED",
        qty=qty,
        take_profit_price=take_profit_price,
        stop_loss_price=stop_loss_price,
        order_id=order.id,
        order_status=order.status,
        equity=current_equity,
        daily_pnl=daily_pnl,
        reason=reason,
    )

    filled_order, filled = wait_for_market_sell_fill(order.id)

    if filled:
        print(
            f"{now_et()} | {symbol} | {signal} MARKET sell filled. "
            f"Filled qty: {filled_order.filled_qty}, avg price: {filled_order.filled_avg_price}"
        )
        log_trade(
            symbol=symbol,
            signal=signal,
            action="MARKET_SELL_FILLED",
            qty=qty,
            take_profit_price=take_profit_price,
            stop_loss_price=stop_loss_price,
            order_id=filled_order.id,
            order_status=filled_order.status,
            filled_qty=filled_order.filled_qty,
            filled_avg_price=filled_order.filled_avg_price,
            equity=current_equity,
            daily_pnl=daily_pnl,
            reason=reason,
        )
        return True, filled_order
    else:
        print(
            f"{now_et()} | {symbol} | {signal} MARKET sell not confirmed filled. "
            f"Status: {filled_order.status}"
        )
        log_trade(
            symbol=symbol,
            signal=signal,
            action="MARKET_SELL_NOT_FILLED",
            qty=qty,
            take_profit_price=take_profit_price,
            stop_loss_price=stop_loss_price,
            order_id=filled_order.id,
            order_status=filled_order.status,
            filled_qty=filled_order.filled_qty,
            filled_avg_price=filled_order.filled_avg_price,
            equity=current_equity,
            daily_pnl=daily_pnl,
            reason=reason,
        )
        return False, filled_order


def record_exit_outcome(state: BotState, symbol: str, signal: str, filled: bool) -> None:
    """
    Tracks all managed exit outcomes, not just take-profits.

    - TAKE_PROFIT and PROFIT_PROTECTION reset the consecutive stop-loss streak.
    - STOP_LOSS increments the streak, cools down that symbol, and can pause new entries.
    - TIME_STOP is treated as a neutral risk-management exit.
    - Non-filled exits do not change streaks because the position may still be open.
    """
    if not filled:
        return

    # Clear per-position tracking after any confirmed exit.
    state.position_entry_times.pop(symbol, None)
    state.highest_close_by_symbol.pop(symbol, None)

    current_time = now_et()

    if signal in {"TAKE_PROFIT", "PROFIT_PROTECTION"}:
        state.consecutive_stop_losses = 0
        return

    if signal == "TIME_STOP":
        # Neutral: the bot exited because the trade was not progressing.
        # Do not count it as a stop-loss streak, but do not reset a stop-loss streak either.
        log_trade(
            symbol=symbol,
            signal="TIME_STOP_RECORDED",
            action="TIME_STOP_TRACKED",
            reason="Time stop filled; no stop-loss streak change",
        )
        return

    if signal == "STOP_LOSS":
        state.consecutive_stop_losses += 1
        state.symbol_cooldowns[symbol] = current_time + timedelta(minutes=COOLDOWN_MINUTES_AFTER_STOP)

        print(
            f"{current_time} | {symbol} | Stop-loss recorded. "
            f"Consecutive stop-losses: {state.consecutive_stop_losses}. "
            f"Ticker cooldown until {state.symbol_cooldowns[symbol]}"
        )

        log_trade(
            symbol=symbol,
            signal="STOP_LOSS_COOLDOWN",
            action="COOLDOWN_SET",
            reason=(
                f"Stop-loss filled; ticker cooldown until {state.symbol_cooldowns[symbol]}; "
                f"consecutive stop-losses {state.consecutive_stop_losses}"
            ),
        )

        if state.consecutive_stop_losses >= MAX_CONSECUTIVE_STOP_LOSSES:
            state.paused_until = current_time + timedelta(minutes=PAUSE_MINUTES_AFTER_MAX_STOP_LOSSES)
            print(
                f"{current_time} | Global entry pause activated until {state.paused_until} "
                f"after {state.consecutive_stop_losses} consecutive stop-losses."
            )
            log_trade(
                symbol="ALL",
                signal="GLOBAL_PAUSE",
                action="PAUSE_AFTER_STOP_LOSSES",
                reason=(
                    f"Paused after {state.consecutive_stop_losses} consecutive stop-losses; "
                    f"resume at {state.paused_until}"
                ),
            )

def is_symbol_in_cooldown(state: BotState, symbol: str) -> tuple[bool, str]:
    cooldown_until = state.symbol_cooldowns.get(symbol)

    if cooldown_until is None:
        return False, "No cooldown"

    if now_et() < cooldown_until:
        return True, f"{symbol} cooldown active until {cooldown_until}"

    del state.symbol_cooldowns[symbol]
    return False, "Cooldown expired"


def manage_open_positions(state: BotState, current_equity: float, daily_pnl: float) -> None:
    """
    Managed exits for long positions.

    Active exit types:
    - TAKE_PROFIT: latest 1-minute bar close reaches TP.
    - STOP_LOSS: latest 1-minute bar close reaches hard SL.
    - PROFIT_PROTECTION: after the trade reaches +1R, protect breakeven+
      and trail below the highest 1-minute bar close.
    - TIME_STOP: after MAX_HOLD_MINUTES, exit if the trade is not at least
      MIN_PROFIT_TO_KEEP_AFTER_TIME_PCT in profit.

    All exits submit MARKET sell orders after bar-close confirmation.
    """
    positions = get_positions()
    current_time = now_et()
    active_symbols = set()

    for position in positions:
        symbol = position.symbol
        active_symbols.add(symbol)

        if symbol not in SYMBOLS:
            continue

        if has_open_order_for_symbol(symbol):
            print(f"{now_et()} | {symbol} | Open order exists. Skipping managed exit check.")
            continue

        qty = float(position.qty)

        if qty <= 0:
            continue

        avg_entry_price = float(position.avg_entry_price)
        take_profit_price = round_price(avg_entry_price * (1 + TAKE_PROFIT_PCT))
        stop_loss_price = round_price(avg_entry_price * (1 - STOP_LOSS_PCT))

        exit_price, price_reason = get_latest_bar_close_for_exit(symbol)

        if exit_price is None:
            print(f"{now_et()} | {symbol} | Could not check managed exit: {price_reason}")
            log_trade(
                symbol=symbol,
                signal="EXIT_CHECK_ERROR",
                action="MANAGED_EXIT_CHECK",
                qty=qty,
                equity=current_equity,
                daily_pnl=daily_pnl,
                reason=price_reason,
            )
            continue

        # Initialize tracking for positions that existed before this bot instance
        # or that were not recorded for some reason.
        state.position_entry_times.setdefault(symbol, current_time)
        previous_high = state.highest_close_by_symbol.get(symbol, avg_entry_price)
        highest_close = max(previous_high, float(exit_price))
        state.highest_close_by_symbol[symbol] = highest_close

        # 1. Normal take-profit exit.
        if exit_price >= take_profit_price:
            reason = (
                f"{price_reason}; avg entry {avg_entry_price:.2f}; "
                f"TP target {take_profit_price:.2f}; bar close confirmed TP hit"
            )
            filled, _ = managed_market_sell_and_log(
                symbol=symbol,
                signal="TAKE_PROFIT",
                qty=qty,
                take_profit_price=take_profit_price,
                stop_loss_price=stop_loss_price,
                current_equity=current_equity,
                daily_pnl=daily_pnl,
                reason=reason,
            )
            record_exit_outcome(state, symbol, "TAKE_PROFIT", filled)
            continue

        # 2. Hard stop-loss exit.
        if exit_price <= stop_loss_price:
            reason = (
                f"{price_reason}; avg entry {avg_entry_price:.2f}; "
                f"SL target {stop_loss_price:.2f}; bar close confirmed SL hit"
            )
            filled, _ = managed_market_sell_and_log(
                symbol=symbol,
                signal="STOP_LOSS",
                qty=qty,
                take_profit_price=take_profit_price,
                stop_loss_price=stop_loss_price,
                current_equity=current_equity,
                daily_pnl=daily_pnl,
                reason=reason,
            )
            record_exit_outcome(state, symbol, "STOP_LOSS", filled)
            continue

        # 3. Profit-protection trailing exit.
        if USE_PROFIT_PROTECTION:
            trigger_price = avg_entry_price * (1 + PROFIT_PROTECTION_TRIGGER_PCT)

            if highest_close >= trigger_price:
                breakeven_stop_price = avg_entry_price * (1 + BREAKEVEN_BUFFER_PCT)
                trailing_stop_price = highest_close * (1 - TRAILING_STOP_AFTER_TRIGGER_PCT)
                protected_stop_price = round_price(max(breakeven_stop_price, trailing_stop_price))

                if exit_price <= protected_stop_price:
                    reason = (
                        f"{price_reason}; avg entry {avg_entry_price:.2f}; "
                        f"highest close {highest_close:.2f}; profit protection trigger {trigger_price:.2f}; "
                        f"protected stop {protected_stop_price:.2f}; bar close confirmed profit-protection exit"
                    )
                    filled, _ = managed_market_sell_and_log(
                        symbol=symbol,
                        signal="PROFIT_PROTECTION",
                        qty=qty,
                        take_profit_price=take_profit_price,
                        stop_loss_price=stop_loss_price,
                        current_equity=current_equity,
                        daily_pnl=daily_pnl,
                        reason=reason,
                    )
                    record_exit_outcome(state, symbol, "PROFIT_PROTECTION", filled)
                    continue

        # 4. Time-stop exit for trades that are not working.
        if USE_TIME_STOP:
            entry_time = state.position_entry_times.get(symbol, current_time)
            hold_minutes = (current_time - entry_time).total_seconds() / 60
            current_return_pct = (exit_price - avg_entry_price) / avg_entry_price

            if hold_minutes >= MAX_HOLD_MINUTES and current_return_pct < MIN_PROFIT_TO_KEEP_AFTER_TIME_PCT:
                reason = (
                    f"{price_reason}; avg entry {avg_entry_price:.2f}; "
                    f"held {hold_minutes:.1f} minutes; return {current_return_pct:.2%} "
                    f"< required {MIN_PROFIT_TO_KEEP_AFTER_TIME_PCT:.2%}; time stop exit"
                )
                filled, _ = managed_market_sell_and_log(
                    symbol=symbol,
                    signal="TIME_STOP",
                    qty=qty,
                    take_profit_price=take_profit_price,
                    stop_loss_price=stop_loss_price,
                    current_equity=current_equity,
                    daily_pnl=daily_pnl,
                    reason=reason,
                )
                record_exit_outcome(state, symbol, "TIME_STOP", filled)
                continue

    # Clean stale tracking keys for symbols no longer held.
    for tracked_symbol in list(state.position_entry_times.keys()):
        if tracked_symbol not in active_symbols:
            state.position_entry_times.pop(tracked_symbol, None)
            state.highest_close_by_symbol.pop(tracked_symbol, None)

def is_flatten_window() -> tuple[bool, str]:
    if not CLOSE_POSITIONS_BEFORE_MARKET_CLOSE:
        return False, "End-of-day flatten disabled"

    current = now_et()
    market_close = current.replace(hour=16, minute=0, second=0, microsecond=0)
    flatten_time = market_close - timedelta(minutes=FLATTEN_MINUTES_BEFORE_CLOSE)

    if flatten_time <= current < market_close:
        return True, f"Inside EOD flatten window: {FLATTEN_MINUTES_BEFORE_CLOSE} minutes before close"

    return False, "Not in EOD flatten window"


def flatten_positions_before_close(state: BotState, current_equity: float, daily_pnl: float) -> bool:
    """
    Market-sells open positions near the end of regular market hours.

    This helps keep the bot day-trade focused and prevents accidental overnight
    holds when a position has not hit TP or SL by the end of the day.
    """
    should_flatten, reason = is_flatten_window()

    if not should_flatten:
        return False

    positions = get_positions()

    if not positions:
        print(f"{now_et()} | {reason}. No open positions to flatten.")
        return True

    print(f"{now_et()} | {reason}. Flattening open positions.")

    for position in positions:
        symbol = position.symbol

        if symbol not in SYMBOLS:
            continue

        if has_open_order_for_symbol(symbol):
            print(f"{now_et()} | {symbol} | Open order exists. Skipping EOD flatten.")
            continue

        qty = float(position.qty)

        if qty <= 0:
            continue

        avg_entry_price = float(position.avg_entry_price)
        take_profit_price = round_price(avg_entry_price * (1 + TAKE_PROFIT_PCT))
        stop_loss_price = round_price(avg_entry_price * (1 - STOP_LOSS_PCT))

        filled, _ = managed_market_sell_and_log(
            symbol=symbol,
            signal="EOD_FLATTEN",
            qty=qty,
            take_profit_price=take_profit_price,
            stop_loss_price=stop_loss_price,
            current_equity=current_equity,
            daily_pnl=daily_pnl,
            reason=reason,
        )

        # Do not treat EOD flatten as a TP or SL for streak logic.
        if filled:
            state.position_entry_times.pop(symbol, None)
            state.highest_close_by_symbol.pop(symbol, None)
            print(f"{now_et()} | {symbol} | EOD flatten filled.")

    return True


# ============================================================
# 13. SCANNER / STRATEGY
# ============================================================

def evaluate_symbol(symbol: str) -> dict:
    bars = get_recent_bars(symbol)

    if bars.empty:
        return {
            "symbol": symbol,
            "signal": "HOLD",
            "reason": "No bars returned",
            "bars": bars,
            "last_close": None,
            "last_bar_timestamp": None,
            "passed": False,
        }

    signal, signal_reason = calculate_base_signal(bars)
    last_close = float(bars["close"].iloc[-1])
    last_bar_timestamp = str(bars.index[-1])

    if signal == "BUY":
        passed, filter_reason = filters_pass(symbol, bars, signal)
    else:
        passed, filter_reason = True, "No entry filters needed"

    ai_ok, ai_reason = ai_decision_veto(symbol, signal, signal_reason, bars)

    if not ai_ok:
        passed = False
        filter_reason = f"AI veto: {ai_reason}"

    return {
        "symbol": symbol,
        "signal": signal,
        "reason": f"{signal_reason}; {filter_reason}; {ai_reason}",
        "bars": bars,
        "last_close": last_close,
        "last_bar_timestamp": last_bar_timestamp,
        "passed": passed,
    }


def scan_symbols() -> list[dict]:
    results = []

    for symbol in SYMBOLS:
        try:
            result = evaluate_symbol(symbol)
            results.append(result)

            print(
                f"{now_et()} | {symbol} | Last close: {result['last_close']} | "
                f"Signal: {result['signal']} | Passed: {result['passed']} | "
                f"Reason: {result['reason']}"
            )

            log_signal(
                symbol=symbol,
                signal=result["signal"],
                passed=result["passed"],
                last_close=result["last_close"],
                reason=result["reason"],
                bars=result["bars"],
            )

        except Exception as e:
            print(f"{now_et()} | Error evaluating {symbol}: {e}")
            log_trade(
                symbol=symbol,
                signal="ERROR",
                action="EVALUATE",
                reason="Symbol evaluation failed",
                error=str(e),
            )

    return results


# ============================================================
# 14. MAIN LOOP
# ============================================================

def main():
    options_trading_stub()
    ensure_trade_journal_exists()
    ensure_signal_journal_exists()
    ensure_position_journal_exists()

    starting_equity = get_equity()
    state = BotState(
        current_day=now_et().date(),
        starting_equity=starting_equity,
        trades_today=0,
    )

    print("Connected to Alpaca paper trading.")
    print(f"Bot version: {BOT_VERSION}")
    print(f"Starting equity: {starting_equity}")
    print(f"Position size target: {POSITION_SIZE_PCT:.2%} of current equity")
    print(f"Min entry qty: {MIN_ENTRY_QTY}")
    print(f"Allow oversized min share: {ALLOW_OVERSIZED_MIN_SHARE}")
    print(f"Watching symbols: {SYMBOLS}")
    print(f"Using data feed: {DATA_FEED}")
    print(f"News mode: {NEWS_MODE}")
    print(f"VWAP tolerance: {VWAP_TOLERANCE_PCT:.2%}")
    print(f"Require close above fast EMA: {REQUIRE_CLOSE_ABOVE_FAST_EMA}")
    print(f"Require EMA slope up: {REQUIRE_EMA_SLOPE_UP} over {EMA_SLOPE_LOOKBACK_BARS} bars")
    print(f"Max consecutive stop-losses: {MAX_CONSECUTIVE_STOP_LOSSES}")
    print(f"Pause after max stop-losses: {PAUSE_MINUTES_AFTER_MAX_STOP_LOSSES} minutes")
    print(f"Ticker cooldown after stop-loss: {COOLDOWN_MINUTES_AFTER_STOP} minutes")
    print(f"Normal volume multiplier: {VOLUME_MULTIPLIER}")
    print(f"News volume multiplier: {NEWS_VOLUME_MULTIPLIER}")
    print(f"Bracket orders: {USE_BRACKET_ORDERS}")
    print(f"Managed market exits: {USE_MANAGED_MARKET_EXITS}")
    print(f"EMA exits enabled: {USE_EMA_EXIT}")
    print(f"Risk/reward target: 1:{RISK_REWARD_RATIO}")
    print(f"Take profit: {TAKE_PROFIT_PCT:.2%}")
    print(f"Stop loss: {STOP_LOSS_PCT:.2%}")
    print(f"Profit protection enabled: {USE_PROFIT_PROTECTION}")
    print(f"Profit protection trigger: {PROFIT_PROTECTION_TRIGGER_PCT:.2%}")
    print(f"Profit protection trail: {TRAILING_STOP_AFTER_TRIGGER_PCT:.2%}")
    print(f"Time stop enabled: {USE_TIME_STOP}, max hold minutes: {MAX_HOLD_MINUTES}")
    print(f"Trailing stop mode: {USE_TRAILING_STOP}")
    print(f"Scan interval seconds: {SCAN_INTERVAL_SECONDS}")
    print(f"Max entry slippage: {MAX_ENTRY_SLIPPAGE_PCT:.2%}")
    print(f"Max spread: {MAX_SPREAD_PCT:.2%}")
    print(f"End-of-day flatten: {CLOSE_POSITIONS_BEFORE_MARKET_CLOSE}, {FLATTEN_MINUTES_BEFORE_CLOSE} minutes before close")
    print(f"Minimum minutes before close to enter: {MIN_MINUTES_BEFORE_CLOSE_TO_ENTER}")
    print(f"One entry attempt per candle: {ONE_ENTRY_ATTEMPT_PER_CANDLE}")
    print(f"Trade journal: {TRADE_JOURNAL_PATH.resolve()}")
    print(f"Signal journal: {SIGNAL_JOURNAL_PATH.resolve()}")
    print(f"Position journal: {POSITION_JOURNAL_PATH.resolve()}")

    while True:
        try:
            state = reset_daily_state_if_needed(state)

            if not is_market_time():
                print(f"{now_et()} | Market not open. Waiting.")
                time.sleep(60)
                continue

            current_equity = get_equity()
            current_daily_pnl = current_equity - state.starting_equity

            if flatten_positions_before_close(state, current_equity, current_daily_pnl):
                log_open_position_snapshots()
                time.sleep(SCAN_INTERVAL_SECONDS)
                continue

            manage_open_positions(state, current_equity, current_daily_pnl)
            log_open_position_snapshots()

            # Refresh equity after possible managed exits.
            current_equity = get_equity()
            allowed, risk_reason, daily_pnl = risk_check(state, current_equity)

            if not allowed:
                print(f"{now_et()} | Trading blocked: {risk_reason} | Daily P/L: {daily_pnl:.2f}")
                log_trade(
                    symbol="ALL",
                    signal="BLOCKED",
                    action="RISK_CHECK",
                    equity=current_equity,
                    daily_pnl=daily_pnl,
                    reason=risk_reason,
                )
                time.sleep(60)
                continue

            results = scan_symbols()

            for result in results:
                symbol = result["symbol"]
                signal = result["signal"]
                last_close = result["last_close"]

                if last_close is None:
                    continue

                if has_open_order_for_symbol(symbol):
                    print(f"{now_et()} | {symbol} | Open order already exists. Skipping.")
                    continue

                # ENTRY LOGIC
                if signal == "BUY" and result["passed"]:
                    if has_position(symbol):
                        print(f"{now_et()} | {symbol} | Already in position. Skipping BUY.")
                        continue

                    in_cooldown, cooldown_reason = is_symbol_in_cooldown(state, symbol)
                    if in_cooldown:
                        print(f"{now_et()} | {symbol} | Entry blocked: {cooldown_reason}")
                        log_trade(
                            symbol=symbol,
                            signal=signal,
                            action="ENTRY_BLOCKED_COOLDOWN",
                            equity=current_equity,
                            daily_pnl=daily_pnl,
                            reason=cooldown_reason,
                        )
                        continue

                    if count_open_positions() >= MAX_OPEN_POSITIONS:
                        print(f"{now_et()} | Max open positions reached. Skipping BUY.")
                        continue

                    candle_key = result.get("last_bar_timestamp")
                    if ONE_ENTRY_ATTEMPT_PER_CANDLE and candle_key:
                        previous_attempt = state.attempted_entry_candles.get(symbol)
                        if previous_attempt == candle_key:
                            print(
                                f"{now_et()} | {symbol} | Already attempted entry for candle {candle_key}. Skipping duplicate."
                            )
                            log_trade(
                                symbol=symbol,
                                signal=signal,
                                action="ENTRY_DUPLICATE_CANDLE_SKIPPED",
                                equity=current_equity,
                                daily_pnl=daily_pnl,
                                reason=f"Duplicate entry attempt prevented for candle {candle_key}",
                            )
                            continue
                        state.attempted_entry_candles[symbol] = candle_key

                    entry_limit_price, price_reason = calculate_entry_limit_price(symbol, last_close)

                    if entry_limit_price is None:
                        print(f"{now_et()} | {symbol} | Entry blocked: {price_reason}")
                        log_trade(
                            symbol=symbol,
                            signal=signal,
                            action="ENTRY_BLOCKED",
                            equity=current_equity,
                            daily_pnl=daily_pnl,
                            reason=price_reason,
                        )
                        continue

                    entry_qty, sizing_reason = calculate_position_qty(current_equity, entry_limit_price)

                    if entry_qty is None:
                        print(f"{now_et()} | {symbol} | Entry blocked by position sizing: {sizing_reason}")
                        log_trade(
                            symbol=symbol,
                            signal=signal,
                            action="ENTRY_BLOCKED_POSITION_SIZE",
                            limit_price=entry_limit_price,
                            equity=current_equity,
                            daily_pnl=daily_pnl,
                            reason=f"{result['reason']} | {price_reason} | {sizing_reason}",
                        )
                        continue

                    print(
                        f"{now_et()} | {symbol} | BUY setup accepted. "
                        f"Limit price: {entry_limit_price}. Qty: {entry_qty}. "
                        f"Reason: {result['reason']} | {price_reason} | {sizing_reason}"
                    )

                    # Entries are always LIMIT buys.
                    # Quantity is sized as a percent of current account equity.
                    # Take-profit and stop-loss exits are managed separately as MARKET sells.
                    order = submit_plain_limit_buy(
                        symbol=symbol,
                        qty=entry_qty,
                        entry_limit_price=entry_limit_price,
                    )
                    take_profit_price, stop_loss_price, _ = calculate_exit_prices(entry_limit_price)

                    print(f"{now_et()} | {symbol} | Entry order submitted: {order.id}")

                    filled_order, filled = wait_for_fill(order.id)

                    if filled:
                        state.trades_today += 1

                        print(
                            f"{now_et()} | {symbol} | Entry filled. "
                            f"Trades today: {state.trades_today}"
                        )

                        log_trade(
                            symbol=symbol,
                            signal=signal,
                            action="BUY_FILLED",
                            qty=entry_qty,
                            limit_price=entry_limit_price,
                            take_profit_price=take_profit_price,
                            stop_loss_price=stop_loss_price,
                            order_id=filled_order.id,
                            order_status=filled_order.status,
                            filled_qty=filled_order.filled_qty,
                            filled_avg_price=filled_order.filled_avg_price,
                            equity=current_equity,
                            daily_pnl=daily_pnl,
                            reason=result["reason"],
                        )

                        try:
                            filled_avg_price = float(filled_order.filled_avg_price)
                        except Exception:
                            filled_avg_price = entry_limit_price

                        state.position_entry_times[symbol] = now_et()
                        state.highest_close_by_symbol[symbol] = filled_avg_price

                        if USE_TRAILING_STOP:
                            position_qty = get_position_qty(symbol)
                            trailing_order = submit_trailing_stop_sell(symbol, position_qty)

                            print(
                                f"{now_et()} | {symbol} | Trailing stop submitted: "
                                f"{trailing_order.id}"
                            )

                            log_trade(
                                symbol=symbol,
                                signal="TRAILING_STOP",
                                action="TRAILING_STOP_SUBMITTED",
                                qty=position_qty,
                                order_id=trailing_order.id,
                                equity=current_equity,
                                daily_pnl=daily_pnl,
                                reason=f"Trail percent: {TRAIL_PERCENT}",
                            )

                    else:
                        print(
                            f"{now_et()} | {symbol} | Entry not filled. "
                            f"Status: {filled_order.status}. Not counting as a trade."
                        )

                        log_trade(
                            symbol=symbol,
                            signal=signal,
                            action="BUY_NOT_FILLED",
                            qty=entry_qty,
                            limit_price=entry_limit_price,
                            order_id=filled_order.id,
                            order_status=filled_order.status,
                            filled_qty=filled_order.filled_qty,
                            filled_avg_price=filled_order.filled_avg_price,
                            equity=current_equity,
                            daily_pnl=daily_pnl,
                            reason="Order not filled before timeout; not counted as trade",
                        )

                # OPTIONAL EMA EXIT LOGIC.
                # Disabled by default because 1-minute EMA cross-downs were exiting trades too early.
                # Managed TP/SL exits above are now the primary exit method.
                elif signal == "SELL" and has_position(symbol):
                    if not USE_EMA_EXIT:
                        print(
                            f"{now_et()} | {symbol} | SELL signal found, but EMA exits are disabled. "
                            f"Holding until managed TP/SL exit."
                        )
                        continue

                    if USE_BRACKET_ORDERS:
                        print(
                            f"{now_et()} | {symbol} | SELL signal found, but bracket exits are active. "
                            f"Skipping manual sell to avoid conflicting exit orders."
                        )
                        continue

                    qty = get_position_qty(symbol)
                    managed_market_sell_and_log(
                        symbol=symbol,
                        signal="EMA_EXIT",
                        qty=qty,
                        take_profit_price=None,
                        stop_loss_price=None,
                        current_equity=current_equity,
                        daily_pnl=daily_pnl,
                        reason=result["reason"],
                    )

            time.sleep(SCAN_INTERVAL_SECONDS)

        except KeyboardInterrupt:
            print("Bot stopped manually.")
            break

        except Exception as e:
            print(f"{now_et()} | Main loop error: {e}")
            log_trade(
                symbol="ALL",
                signal="ERROR",
                action="MAIN_LOOP",
                reason="Unhandled main loop error",
                error=str(e),
            )
            time.sleep(60)


if __name__ == "__main__":
    main()