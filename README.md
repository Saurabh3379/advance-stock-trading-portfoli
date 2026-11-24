# Advance Stock Trading and Portfolio Management 📈💼
> 🌐 **Live Web Application:** [https://advance-stock-trading-portfoli-c5axpwnvmreotunjhasjnn.streamlit.app/](https://advance-stock-trading-portfoli-c5axpwnvmreotunjhasjnn.streamlit.app/)
A comprehensive, production-grade paper trading and quantitative portfolio analytics web application built with **Streamlit**, **yfinance**, **Plotly**, **scikit-learn**, and **SQLite**.

Designed specifically for the Indian equity market (NSE stocks like `RELIANCE.NS`, `TCS.NS`, `INFY.NS`, `HDFCBANK.NS`, `ITC.NS`, `SBIN.NS`, etc.) with full support for any global ticker symbol.

---

## 🌟 Key Features

1. **🔐 Authentication & Capital Initialization**
   - User Registration & Login with PBKDF2-HMAC-SHA256 password hashing and individual cryptographic salts.
   - Each newly registered trader automatically receives **₹1,00,000** in virtual balance to practice risk-free paper trading.
   - Session state persistence and secure logout.

2. **📊 Real-Time Market Dashboard**
   - Curated dropdown of premier NSE bluechip stocks plus custom symbol search.
   - Horizon period filters: **1 Month (`1mo`)**, **6 Months (`6mo`)**, **1 Year (`1y`)**, and **5 Years (`5y`)**.
   - Metric cards: Current Market Price, Day Price Change (₹ and %), 24h Trading Volume, Day Range (Low/High), and 52-Week High/Low.
   - High-performance Candlestick chart with volume subchart.
   - Instant quick-order execution bar.

3. **📈 Technical Indicators with Plain-English Explanations**
   - **Moving Averages (SMA 20 & SMA 50):** Trend direction and Golden/Death cross detection.
   - **Bollinger Bands (20, 2):** Volatility envelopes, band squeezes, and breakout signals.
   - **Relative Strength Index (RSI 14):** Momentum oscillator with 70/30 overbought/oversold bands.
   - **MACD (12, 26, 9):** Fast line, Signal line, and colored volume momentum histogram.
   - Interactive toggle switches allowing users to layer overlays onto the chart.
   - Educational plain-English guide under each indicator explaining practical interpretation.

4. **💹 Paper Trading Order Terminal**
   - Simulated **BUY** and **SELL** order placement.
   - Market orders and Custom Limit orders.
   - Atomic SQLite transactions validating available liquid cash for purchases and existing share ownership for sales.
   - Instant P&L preview and trade validation badges.

5. **💼 Portfolio Management & Allocation**
   - **Holdings Table:** Live tracking of Quantity, Average Buy Price, Current Price, Invested Value, Current Market Value, and Unrealized P&L (₹ and %).
   - **Portfolio KPI Cards:** Total Portfolio Value, Unrealized P&L, Available Cash, and Overall Return % on starting capital.
   - **Interactive Asset Allocation:** Plotly donut chart depicting percentage distribution across stocks and cash.
   - **Trade History Ledger:** Complete audit trail of executed orders with timestamps and pricing.

6. **🤖 AI Price Prediction (Next 7 Days)**
   - Supervised Machine Learning algorithms using **scikit-learn** (`Linear Regression` and `Random Forest Regressor`).
   - Feature engineering with lagged close prices, rolling moving averages, and return momentum.
   - Recursive multi-step forward projection for the upcoming 7 trading days.
   - Chronological test evaluation metrics: R² Score, Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE).
   - Interactive forecast visualization chart with historical context and ±1 RMSE confidence bands.
   - Prominent educational disclaimer note.

7. **🛡️ Risk Analysis & Portfolio Analytics**
   - **Annualized Volatility (%):** Standard deviation of daily returns annualized over 252 trading days.
   - **Sharpe Ratio:** Risk-adjusted returns relative to the Indian 6.5% benchmark rate.
   - **Maximum Drawdown (MDD %):** Peak-to-trough decline analysis.
   - **Portfolio Diversification Score (0–100):** Herfindahl-Hirschman Index (HHI) concentration metric classifying portfolios as *Well Diversified*, *Moderately Diversified*, or *Concentrated*.

8. **⭐ Watchlist & Target Price Alerts**
   - Personal stock watchlist with real-time price updates and 24h change %.
   - Custom price alerts with `ABOVE` or `BELOW` threshold conditions.
   - Automatic banner notifications and toast alerts when active stocks cross user-defined price targets.

---

## 🏗️ Project Architecture

```
hcl project/
│
├── app.py              # Main Streamlit application and UI layout
├── database.py         # SQLite connection, tables, trade execution & audit trails
├── auth.py             # User password hashing, authentication & session state
├── data.py             # yfinance market data fetcher with st.cache_data (5-min TTL)
├── indicators.py       # SMA, RSI, MACD, Bollinger Bands & Plotly chart generators
├── prediction.py       # Scikit-learn 7-day multi-step recursive ML forecasting
├── risk.py             # Sharpe ratio, volatility, MDD & Diversification score
├── requirements.txt    # Project Python dependencies
└── README.md           # Documentation and run instructions
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10, 3.11, 3.12, 3.13, or 3.14
- Git (optional)

### 2. Installation

Clone or navigate to the project directory:
```bash
cd "c:\Users\Nabeel-IT\OneDrive\Desktop\hcl project"
```

Install all required packages:
```bash
pip install -r requirements.txt
```

*(Or if you use a specific Python executable path:)*
```powershell
& "C:\Users\Nabeel-IT\AppData\Local\Python\bin\python.exe" -m pip install -r requirements.txt
```

### 3. Running the Web App

Launch Streamlit:
```bash
streamlit run app.py
```

*(Or using the direct Python binary:)*
```powershell
& "C:\Users\Nabeel-IT\AppData\Local\Python\bin\python.exe" -m streamlit run app.py
```

The application will start and open automatically in your browser at:
```
http://localhost:8501
```

---

## 💡 How to Use

1. **Sign Up**: Create an account on the login screen. You will be credited with **₹1,00,000** in virtual balance.
2. **Explore Stocks**: Navigate to the **Stock Dashboard** to inspect live candles, volume, and quotes for Reliance, TCS, Infosys, HDFC Bank, etc.
3. **Analyze Indicators**: Open **Technical Indicators** to toggle SMA 20/50, Bollinger Bands, RSI, and MACD overlays. Read the plain-English guide to learn what each signal indicates.
4. **Place Trades**: Head to the **Trade Terminal** to execute paper BUY/SELL orders. Notice your cash balance and holdings update immediately.
5. **Manage Portfolio**: Review your **Portfolio Management** tab for real-time P&L, asset allocation donut chart, and complete transaction history.
6. **Forecast Trends**: Switch to **Price Prediction** to generate a 7-day forward trajectory using Linear Regression or Random Forest.
7. **Evaluate Risk**: Go to **Risk & Analytics** to check your portfolio's Sharpe ratio, max drawdown, and diversification score.
8. **Set Alerts**: Add stocks to your **Watchlist** and configure target price notifications.

---

## ⚠️ Disclaimer

*This application is built strictly for **educational and research purposes**. It does not constitute financial, investment, or trading advice. Paper trading uses simulated executions, and machine learning price predictions are theoretical experimental forecasts that should never be relied upon for real capital deployment.*
