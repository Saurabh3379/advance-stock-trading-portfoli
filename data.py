"""
data.py - Market data fetching and caching module for Advance Stock Trading and Portfolio Management.
Fetches stock historical candles and real-time metadata using yfinance,
caches data using Streamlit st.cache_data (5-minute TTL),
and gracefully handles network issues or unknown tickers.
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple, List

# Standard curated list of popular NSE stocks
POPULAR_NSE_STOCKS = {
    "RELIANCE.NS": "Reliance Industries Ltd",
    "TCS.NS": "Tata Consultancy Services Ltd",
    "INFY.NS": "Infosys Ltd",
    "HDFCBANK.NS": "HDFC Bank Ltd",
    "ICICIBANK.NS": "ICICI Bank Ltd",
    "ITC.NS": "ITC Ltd",
    "SBIN.NS": "State Bank of India",
    "BHARTIARTL.NS": "Bharti Airtel Ltd",
    "TATAMOTORS.NS": "Tata Motors Ltd",
    "LT.NS": "Larsen & Toubro Ltd",
    "KOTAKBANK.NS": "Kotak Mahindra Bank Ltd",
    "WIPRO.NS": "Wipro Ltd",
    "HINDUNILVR.NS": "Hindustan Unilever Ltd",
    "BAJFINANCE.NS": "Bajaj Finance Ltd",
    "MARUTI.NS": "Maruti Suzuki India Ltd"
}

PERIOD_MAP = {
    "1mo": {"period": "1mo", "interval": "1d", "label": "1 Month"},
    "6mo": {"period": "6mo", "interval": "1d", "label": "6 Months"},
    "1y": {"period": "1y", "interval": "1d", "label": "1 Year"},
    "5y": {"period": "5y", "interval": "1wk", "label": "5 Years"}
}


def normalize_symbol(symbol: str) -> str:
    """Normalizes symbol: trims whitespace, upper-cases, and appends .NS if Indian ticker without extension."""
    sym = symbol.strip().upper()
    # If standard ticker without exchange suffix and looks like an Indian ticker, suggest or append .NS
    if "." not in sym and "^" not in sym:
        sym = f"{sym}.NS"
    return sym


@st.cache_data(ttl=300, show_spinner=False)
def get_stock_data(symbol: str, period: str = "1y") -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Fetches historical OHLCV data for a given symbol and period.
    Returns (DataFrame, error_message).
    Cached for 5 minutes (ttl=300).
    """
    symbol = symbol.strip().upper()
    period_cfg = PERIOD_MAP.get(period, {"period": "1y", "interval": "1d"})

    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period_cfg["period"], interval=period_cfg["interval"], auto_adjust=True)

        if df is None or df.empty:
            return None, f"No historical data available for '{symbol}'. Please check the symbol and try again."

        # Flatten multi-index columns if present (can happen in newer yfinance versions)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Ensure index is timezone-naive datetime for consistency
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # Standardize expected columns
        expected_cols = ["Open", "High", "Low", "Close", "Volume"]
        for col in expected_cols:
            if col not in df.columns:
                return None, f"Missing required price column '{col}' for '{symbol}'."

        df = df[expected_cols].copy()
        df.dropna(subset=["Close"], inplace=True)
        return df, None

    except Exception as e:
        # Graceful error handling
        return None, f"Unable to fetch market data for {symbol}: {str(e)}"


@st.cache_data(ttl=300, show_spinner=False)
def get_stock_quote(symbol: str) -> Dict[str, Any]:
    """
    Retrieves current market quote and fundamental overview for a stock.
    Returns dictionary with price, day change, volume, high, low, etc.
    """
    symbol = symbol.strip().upper()
    default_quote = {
        "symbol": symbol,
        "name": POPULAR_NSE_STOCKS.get(symbol, symbol),
        "current_price": 0.0,
        "previous_close": 0.0,
        "change": 0.0,
        "change_percent": 0.0,
        "day_high": 0.0,
        "day_low": 0.0,
        "volume": 0,
        "fifty_two_week_high": 0.0,
        "fifty_two_week_low": 0.0,
        "currency": "INR",
        "valid": False,
        "error": None
    }

    try:
        ticker = yf.Ticker(symbol)
        
        # First attempt: fast_info (quickest in yfinance)
        fast_info = getattr(ticker, "fast_info", None)
        current_price = 0.0
        prev_close = 0.0

        if fast_info:
            try:
                current_price = float(fast_info.last_price or 0.0)
                prev_close = float(fast_info.previous_close or 0.0)
                default_quote["day_high"] = float(fast_info.day_high or 0.0)
                default_quote["day_low"] = float(fast_info.day_low or 0.0)
                default_quote["fifty_two_week_high"] = float(fast_info.year_high or 0.0)
                default_quote["fifty_two_week_low"] = float(fast_info.year_low or 0.0)
                default_quote["currency"] = str(fast_info.currency or "INR")
            except Exception:
                pass

        # Fallback to history if fast_info missing price
        if current_price <= 0:
            hist = ticker.history(period="5d", interval="1d")
            if hist is not None and not hist.empty:
                current_price = float(hist["Close"].iloc[-1])
                if len(hist) > 1:
                    prev_close = float(hist["Close"].iloc[-2])
                else:
                    prev_close = float(hist["Open"].iloc[-1])
                default_quote["day_high"] = float(hist["High"].iloc[-1])
                default_quote["day_low"] = float(hist["Low"].iloc[-1])
                default_quote["volume"] = int(hist["Volume"].iloc[-1])

        if current_price > 0:
            change = current_price - prev_close if prev_close > 0 else 0.0
            change_percent = (change / prev_close) * 100 if prev_close > 0 else 0.0

            # Try to get company name
            try:
                info = ticker.info
                if info and isinstance(info, dict):
                    default_quote["name"] = info.get("shortName") or info.get("longName") or default_quote["name"]
                    if not default_quote["fifty_two_week_high"]:
                        default_quote["fifty_two_week_high"] = float(info.get("fiftyTwoWeekHigh") or 0.0)
                    if not default_quote["fifty_two_week_low"]:
                        default_quote["fifty_two_week_low"] = float(info.get("fiftyTwoWeekLow") or 0.0)
            except Exception:
                pass

            default_quote.update({
                "current_price": round(current_price, 2),
                "previous_close": round(prev_close, 2),
                "change": round(change, 2),
                "change_percent": round(change_percent, 2),
                "valid": True
            })
            return default_quote

        default_quote["error"] = f"Could not obtain current price for {symbol}."
        return default_quote

    except Exception as e:
        default_quote["error"] = str(e)
        return default_quote


@st.cache_data(ttl=300, show_spinner=False)
def get_multiple_quotes(symbols: Tuple[str, ...]) -> Dict[str, Dict[str, Any]]:
    """
    Fetches current quotes for a list of symbols (tuple passed for cache hashability).
    Returns dict mapping symbol -> quote dict.
    """
    quotes = {}
    for sym in symbols:
        quotes[sym] = get_stock_quote(sym)
    return quotes


def get_curated_stocks_list() -> List[Dict[str, str]]:
    """Returns list of curated popular NSE stocks with labels."""
    return [{"symbol": sym, "name": name} for sym, name in POPULAR_NSE_STOCKS.items()]
