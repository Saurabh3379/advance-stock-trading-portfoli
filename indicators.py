"""
indicators.py - Technical indicator computations and Plotly chart generation.
Includes Moving Averages (20, 50), RSI(14), MACD(12, 26, 9), and Bollinger Bands (20, 2),
with educational plain-English explanations.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, Tuple


# Plain-English educational explanations
INDICATOR_EXPLANATIONS = {
    "MA": (
        "**Moving Averages (MA 20 & MA 50):**\n"
        "Moving averages smooth out short-term noise to highlight the underlying market trend. "
        "The **20 MA** reflects short-term momentum, while the **50 MA** captures medium-term trend direction. "
        "A bullish signal (**Golden Cross**) occurs when the 20 MA crosses above the 50 MA. "
        "A bearish signal (**Death Cross**) occurs when the 20 MA crosses below the 50 MA."
    ),
    "RSI": (
        "**Relative Strength Index (RSI - 14 Days):**\n"
        "RSI is a momentum oscillator measuring the speed and change of price movements between 0 and 100. "
        "- **Above 70 (Overbought):** The asset may be overheated or overvalued, suggesting a potential correction or pullback.\n"
        "- **Below 30 (Oversold):** The asset has experienced heavy selling and may be undervalued, presenting potential rebound opportunities.\n"
        "- **Around 50:** Neutral territory with balanced buying and selling pressure."
    ),
    "MACD": (
        "**Moving Average Convergence Divergence (MACD 12, 26, 9):**\n"
        "MACD reveals changes in trend strength, direction, momentum, and duration. "
        "It consists of the fast MACD line (12 EMA - 26 EMA) and the slower Signal line (9 EMA). "
        "When the MACD line crosses **above** the Signal line, it signals bullish momentum (buying wave). "
        "When it crosses **below**, it signals bearish momentum. "
        "The histogram bars visualize the expanding or contracting gap between both lines."
    ),
    "BB": (
        "**Bollinger Bands (20 SMA, 2 Standard Deviations):**\n"
        "Bollinger Bands define dynamic volatility envelopes above and below the 20-day moving average. "
        "When market volatility dries up, the bands contract into a **Squeeze**, often signaling an explosive price breakout soon. "
        "Prices touching or breaking above the upper band indicate strong bullish strength or overextended conditions, "
        "while prices touching the lower band indicate oversold support levels."
    )
}


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes all standard technical indicators on historical price dataframe.
    Calculates SMA 20, SMA 50, EMA 20, Bollinger Bands, RSI(14), and MACD.
    """
    data = df.copy()

    # 1. Moving Averages
    data["SMA_20"] = data["Close"].rolling(window=20).mean()
    data["SMA_50"] = data["Close"].rolling(window=50).mean()
    data["EMA_20"] = data["Close"].ewm(span=20, adjust=False).mean()

    # 2. Bollinger Bands (20-period SMA, 2 std deviations)
    bb_period = 20
    bb_std = 2
    bb_mid = data["Close"].rolling(window=bb_period).mean()
    bb_sigma = data["Close"].rolling(window=bb_period).std()
    data["BB_Middle"] = bb_mid
    data["BB_Upper"] = bb_mid + (bb_sigma * bb_std)
    data["BB_Lower"] = bb_mid - (bb_sigma * bb_std)
    data["BB_Width"] = (data["BB_Upper"] - data["BB_Lower"]) / data["BB_Middle"]

    # 3. Relative Strength Index (RSI - 14)
    delta = data["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    
    # Wilder's exponential smoothing
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    data["RSI_14"] = 100 - (100 / (1 + rs))
    # Replace NaNs for edge cases
    data["RSI_14"] = data["RSI_14"].fillna(50)

    # 4. MACD (12, 26, 9)
    ema_12 = data["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = data["Close"].ewm(span=26, adjust=False).mean()
    data["MACD"] = ema_12 - ema_26
    data["MACD_Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()
    data["MACD_Hist"] = data["MACD"] - data["MACD_Signal"]

    return data


def create_candlestick_chart(
    df: pd.DataFrame,
    symbol: str,
    show_sma20: bool = True,
    show_sma50: bool = True,
    show_bollinger: bool = False,
    show_volume: bool = True
) -> go.Figure:
    """
    Creates an interactive Plotly candlestick chart with volume and optional indicator overlays.
    """
    data = calculate_indicators(df)

    rows = 2 if show_volume else 1
    row_heights = [0.75, 0.25] if show_volume else [1.0]

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=row_heights
    )

    # 1. Main Candlestick
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name="Price",
            increasing_line_color="#00C805",
            decreasing_line_color="#FF3B30",
            increasing_fillcolor="#00C805",
            decreasing_fillcolor="#FF3B30"
        ),
        row=1, col=1
    )

    # Bollinger Bands
    if show_bollinger:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["BB_Upper"],
                line=dict(color="rgba(173, 216, 230, 0.6)", width=1, dash="dash"),
                name="BB Upper (20,2)"
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["BB_Lower"],
                line=dict(color="rgba(173, 216, 230, 0.6)", width=1, dash="dash"),
                fill="tonexty",
                fillcolor="rgba(173, 216, 230, 0.08)",
                name="BB Lower (20,2)"
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["BB_Middle"],
                line=dict(color="rgba(173, 216, 230, 0.4)", width=1),
                name="BB Mid"
            ),
            row=1, col=1
        )

    # SMA 20
    if show_sma20:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["SMA_20"],
                line=dict(color="#FF9500", width=1.5),
                name="SMA 20"
            ),
            row=1, col=1
        )

    # SMA 50
    if show_sma50:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["SMA_50"],
                line=dict(color="#007AFF", width=1.5),
                name="SMA 50"
            ),
            row=1, col=1
        )

    # Volume Chart
    if show_volume:
        colors = [
            "#00C805" if c >= o else "#FF3B30"
            for c, o in zip(data["Close"], data["Open"])
        ]
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data["Volume"],
                name="Volume",
                marker_color=colors,
                opacity=0.65
            ),
            row=2, col=1
        )

    # Formatting Layout for dark mode
    fig.update_layout(
        title=f"<b>{symbol}</b> Price Action & Technical Overview",
        title_font=dict(size=18, color="#FFFFFF"),
        template="plotly_dark",
        plot_bgcolor="#11141A",
        paper_bgcolor="#11141A",
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11)
        ),
        margin=dict(l=40, r=40, t=50, b=30),
        height=540,
        hovermode="x unified"
    )

    fig.update_xaxes(showgrid=True, gridcolor="#1F2430")
    fig.update_yaxes(showgrid=True, gridcolor="#1F2430")

    return fig


def create_rsi_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    """Creates a dedicated RSI(14) oscillator chart with overbought/oversold levels."""
    data = calculate_indicators(df)
    current_rsi = data["RSI_14"].iloc[-1] if not data.empty else 50.0

    fig = go.Figure()

    # Overbought zone fill (70 - 100)
    fig.add_hrect(
        y0=70, y1=100,
        fillcolor="rgba(255, 59, 48, 0.12)",
        line_width=0,
        annotation_text="Overbought (>70)",
        annotation_position="top left",
        annotation_font=dict(color="#FF3B30", size=10)
    )

    # Oversold zone fill (0 - 30)
    fig.add_hrect(
        y0=0, y1=30,
        fillcolor="rgba(0, 200, 5, 0.12)",
        line_width=0,
        annotation_text="Oversold (<30)",
        annotation_position="bottom left",
        annotation_font=dict(color="#00C805", size=10)
    )

    # Threshold lines
    fig.add_hline(y=70, line=dict(color="#FF3B30", width=1, dash="dash"))
    fig.add_hline(y=50, line=dict(color="rgba(255, 255, 255, 0.2)", width=1, dash="dot"))
    fig.add_hline(y=30, line=dict(color="#00C805", width=1, dash="dash"))

    # RSI Line
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["RSI_14"],
            mode="lines",
            line=dict(color="#AF52DE", width=2),
            name="RSI (14)"
        )
    )

    fig.update_layout(
        title=f"<b>RSI (14) Indicator</b> — Current: <span style='color: {'#FF3B30' if current_rsi >= 70 else '#00C805' if current_rsi <= 30 else '#AF52DE'}'>{current_rsi:.1f}</span>",
        title_font=dict(size=15, color="#FFFFFF"),
        template="plotly_dark",
        plot_bgcolor="#11141A",
        paper_bgcolor="#11141A",
        yaxis=dict(range=[0, 100], tickvals=[20, 30, 50, 70, 80]),
        height=260,
        margin=dict(l=40, r=40, t=40, b=25),
        showlegend=False,
        hovermode="x unified"
    )
    fig.update_xaxes(showgrid=True, gridcolor="#1F2430")
    fig.update_yaxes(showgrid=True, gridcolor="#1F2430")

    return fig


def create_macd_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    """Creates a dedicated MACD chart with MACD line, Signal line, and colored Histogram."""
    data = calculate_indicators(df)

    fig = go.Figure()

    # Histogram colors
    hist_colors = [
        "#00C805" if val >= 0 else "#FF3B30"
        for val in data["MACD_Hist"]
    ]

    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data["MACD_Hist"],
            name="Histogram",
            marker_color=hist_colors,
            opacity=0.75
        )
    )

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["MACD"],
            name="MACD (12, 26)",
            line=dict(color="#007AFF", width=1.8)
        )
    )

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["MACD_Signal"],
            name="Signal (9)",
            line=dict(color="#FF9500", width=1.5, dash="dot")
        )
    )

    fig.update_layout(
        title="<b>MACD Indicator (12, 26, 9)</b>",
        title_font=dict(size=15, color="#FFFFFF"),
        template="plotly_dark",
        plot_bgcolor="#11141A",
        paper_bgcolor="#11141A",
        height=260,
        margin=dict(l=40, r=40, t=40, b=25),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )
    fig.update_xaxes(showgrid=True, gridcolor="#1F2430")
    fig.update_yaxes(showgrid=True, gridcolor="#1F2430")

    return fig
