"""
prediction.py - Machine Learning price forecasting module using scikit-learn.
Provides 7-day future stock price predictions using Linear Regression or Random Forest,
displays historical vs forecast trajectory on an interactive Plotly chart,
evaluates model metrics (R2, RMSE, MAE), and includes mandatory educational disclaimers.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta
from typing import Dict, Any, Tuple, Optional
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

PREDICTION_DISCLAIMER = (
    "⚠️ **Educational & Research Purpose Only:** "
    "Stock markets are influenced by macroeconomic dynamics, geopolitical events, news sentiment, "
    "and liquidity flows that standard statistical ML models cannot foresee. "
    "This 7-day forecast is for educational and algorithmic demonstration only and should "
    "never be construed as investment advice, price targets, or financial recommendations."
)


def prepare_features(df: pd.DataFrame, lags: int = 5) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Constructs feature matrix using price lags and rolling statistical metrics.
    """
    data = df.copy()
    
    # Feature 1: Lagged close prices
    for i in range(1, lags + 1):
        data[f"Lag_{i}"] = data["Close"].shift(i)

    # Feature 2: Rolling moving averages
    data["Rolling_Mean_5"] = data["Close"].shift(1).rolling(window=5).mean()
    data["Rolling_Mean_10"] = data["Close"].shift(1).rolling(window=10).mean()
    data["Rolling_Std_5"] = data["Close"].shift(1).rolling(window=5).std()

    # Feature 3: Price momentum
    data["Return_1d"] = data["Close"].shift(1).pct_change()
    data["Return_5d"] = data["Close"].shift(1).pct_change(periods=5)

    # Target: Next day's close price
    data["Target"] = data["Close"]

    data.dropna(inplace=True)

    feature_cols = [c for c in data.columns if c.startswith("Lag_") or c.startswith("Rolling_") or c.startswith("Return_")]
    X = data[feature_cols]
    y = data["Target"]
    return X, y


def train_and_predict_7days(
    df: pd.DataFrame,
    model_name: str = "Linear Regression",
    n_days: int = 7
) -> Dict[str, Any]:
    """
    Trains selected scikit-learn model on historical data,
    evaluates validation metrics, and forecasts next n_days prices recursively.
    """
    if len(df) < 40:
        return {
            "success": False,
            "error": "Insufficient historical data points (minimum 40 required) for ML training."
        }

    X, y = prepare_features(df)
    if len(X) < 30:
        return {
            "success": False,
            "error": "Dataset too small after feature generation."
        }

    # Train-test split (80% train, 20% test for chronological evaluation)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Select model
    if model_name == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
    else:
        model = LinearRegression()

    model.fit(X_train, y_train)

    # Test set evaluation
    y_pred_test = model.predict(X_test)
    r2 = r2_score(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae = mean_absolute_error(y_test, y_pred_test)

    # Retrain on full dataset for forward forecasting
    model.fit(X, y)

    # Recursive multi-step forward forecasting
    current_df = df.copy()
    future_dates = []
    future_prices = []

    last_date = current_df.index[-1]
    curr_date = last_date

    for step in range(n_days):
        # Move to next trading day (skip weekends)
        curr_date += timedelta(days=1)
        while curr_date.weekday() >= 5:  # 5: Sat, 6: Sun
            curr_date += timedelta(days=1)

        X_step, _ = prepare_features(current_df)
        if X_step.empty:
            break

        latest_features = X_step.iloc[[-1]]
        pred_price = float(model.predict(latest_features)[0])
        pred_price = max(1.0, round(pred_price, 2))

        future_dates.append(curr_date)
        future_prices.append(pred_price)

        # Append predicted row to DataFrame for next recursive feature computation
        new_row = pd.DataFrame({
            "Open": [pred_price],
            "High": [pred_price * 1.005],
            "Low": [pred_price * 0.995],
            "Close": [pred_price],
            "Volume": [int(current_df["Volume"].iloc[-5:].mean())]
        }, index=[curr_date])
        current_df = pd.concat([current_df, new_row])

    current_price = float(df["Close"].iloc[-1])
    predicted_7d_price = future_prices[-1] if future_prices else current_price
    pct_change = ((predicted_7d_price - current_price) / current_price) * 100 if current_price > 0 else 0.0

    return {
        "success": True,
        "model_name": model_name,
        "r2_score": round(float(r2), 4),
        "rmse": round(float(rmse), 2),
        "mae": round(float(mae), 2),
        "current_price": round(current_price, 2),
        "predicted_price_7d": round(predicted_7d_price, 2),
        "pct_change_7d": round(pct_change, 2),
        "future_dates": future_dates,
        "future_prices": future_prices
    }


def create_prediction_chart(
    df: pd.DataFrame,
    prediction_result: Dict[str, Any],
    symbol: str,
    lookback_days: int = 60
) -> go.Figure:
    """
    Renders an interactive Plotly chart showing recent historical prices alongside the 7-day ML forecast.
    """
    hist_slice = df.tail(lookback_days)
    fig = go.Figure()

    # Historical prices
    fig.add_trace(
        go.Scatter(
            x=hist_slice.index,
            y=hist_slice["Close"],
            mode="lines+markers",
            line=dict(color="#007AFF", width=2.5),
            marker=dict(size=4),
            name="Actual Historical Close"
        )
    )

    if prediction_result.get("success"):
        # Connect last historical point to first prediction point
        last_hist_date = hist_slice.index[-1]
        last_hist_price = float(hist_slice["Close"].iloc[-1])

        forecast_x = [last_hist_date] + prediction_result["future_dates"]
        forecast_y = [last_hist_price] + prediction_result["future_prices"]

        # Forecast line
        fig.add_trace(
            go.Scatter(
                x=forecast_x,
                y=forecast_y,
                mode="lines+markers",
                line=dict(color="#FF9500", width=2.5, dash="dash"),
                marker=dict(size=6, symbol="diamond", color="#FF9500"),
                name=f"7-Day Forecast ({prediction_result['model_name']})"
            )
        )

        # Confidence envelope / visual band around forecast
        rmse = prediction_result.get("rmse", 0.0)
        upper_bound = [p + rmse for p in forecast_y]
        lower_bound = [max(1.0, p - rmse) for p in forecast_y]

        fig.add_trace(
            go.Scatter(
                x=forecast_x,
                y=upper_bound,
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip"
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast_x,
                y=lower_bound,
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(255, 149, 0, 0.12)",
                name="Prediction Interval (±1 RMSE)"
            )
        )

    fig.update_layout(
        title=f"<b>{symbol}</b> — 7-Day Price Forecast Analysis",
        title_font=dict(size=17, color="#FFFFFF"),
        template="plotly_dark",
        plot_bgcolor="#11141A",
        paper_bgcolor="#11141A",
        height=480,
        margin=dict(l=40, r=40, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )

    fig.update_xaxes(showgrid=True, gridcolor="#1F2430")
    fig.update_yaxes(showgrid=True, gridcolor="#1F2430")

    return fig
