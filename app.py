"""
app.py - Advance Stock Trading and Portfolio Management Platform.
Main Streamlit application with authentication, interactive trading terminal,
portfolio analytics, technical indicators, ML price prediction, risk metrics,
watchlist, and price alerts.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Import modular project components
import database as db
import auth
import data as market_data
import indicators
import prediction
import risk

# -----------------------------------------------------------------------------
# Streamlit Application Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Advance Stock Trading & Portfolio Management",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database schema if not already initialized
db.init_db()
auth.init_session_state()

# -----------------------------------------------------------------------------
# Custom Styling / CSS for Modern Dark Financial UI
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Dark Theme Core Tweaks */
    .stApp {
        background-color: #0b0e14;
        color: #e2e8f0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Top title styling */
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
    
    /* Financial Metric Card */
    .metric-card {
        background: #151a23;
        border: 1px solid #222b3a;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .metric-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #f8fafc;
    }
    .metric-delta-pos {
        color: #22c55e;
        font-size: 0.88rem;
        font-weight: 600;
    }
    .metric-delta-neg {
        color: #ef4444;
        font-size: 0.88rem;
        font-weight: 600;
    }
    
    /* Alert badge */
    .alert-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-buy { background: rgba(34, 197, 94, 0.2); color: #22c55e; border: 1px solid #22c55e; }
    .badge-sell { background: rgba(239, 68, 68, 0.2); color: #ef4444; border: 1px solid #ef4444; }
    .badge-info { background: rgba(56, 189, 248, 0.2); color: #38bdf8; border: 1px solid #38bdf8; }

    /* Custom Footer */
    .app-footer {
        text-align: center;
        padding: 24px 0 10px 0;
        color: #64748b;
        font-size: 0.85rem;
        border-top: 1px solid #1e293b;
        margin-top: 40px;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0f131a;
        border-right: 1px solid #1e293b;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Authentication Screen (when not logged in)
# -----------------------------------------------------------------------------
def render_auth_page():
    st.markdown("<div class='main-header'>📈 Advance Stock Trading & Portfolio Management</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Paper Trading, Technical Indicators, AI Forecasting & Portfolio Risk Analytics</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("### 🔐 User Login")
        with st.form("login_form"):
            login_username = st.text_input("Username", key="login_user")
            login_password = st.text_input("Password", type="password", key="login_pass")
            submit_login = st.form_submit_button("Sign In", use_container_width=True)

            if submit_login:
                success, msg, user_data = auth.login_user(login_username, login_password)
                if success:
                    auth.login_session(user_data)
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)

    with col2:
        st.markdown("### ✨ Create New Account")
        st.info("🎁 **New Account Bonus:** Every new trader receives **₹1,00,000** in virtual balance to practice paper trading risk-free!")
        with st.form("signup_form"):
            reg_username = st.text_input("Choose Username", key="reg_user")
            reg_password = st.text_input("Choose Password (min 6 chars)", type="password", key="reg_pass")
            reg_confirm = st.text_input("Confirm Password", type="password", key="reg_confirm")
            submit_signup = st.form_submit_button("Register & Claim ₹1,00,000", use_container_width=True)

            if submit_signup:
                success, msg, user_data = auth.register_user(reg_username, reg_password, reg_confirm)
                if success:
                    auth.login_session(user_data)
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)

    st.markdown("---")
    st.markdown("#### 🌟 Platform Highlights")
    fcol1, fcol2, fcol3, fcol4 = st.columns(4)
    with fcol1:
        st.markdown("**NSE Real-Time Data**\nLive candlestick charts and market quotes for top Indian bluechips.")
    with fcol2:
        st.markdown("**Technical Indicators**\nSMA 20/50, Bollinger Bands, RSI(14), and MACD with plain-English insights.")
    with fcol3:
        st.markdown("**AI Price Prediction**\nScikit-learn powered 7-day multi-step recursive price forecasting.")
    with fcol4:
        st.markdown("**Risk & Diversification**\nSharpe ratio, annualized volatility, max drawdown, and HHI portfolio scores.")

    render_footer()


# -----------------------------------------------------------------------------
# Common Components: Stock Selector Header & Alerts Checker
# -----------------------------------------------------------------------------
def get_selected_stock(key_prefix: str = "main") -> str:
    """Stock symbol selector with curated NSE stocks + custom ticker input."""
    curated = market_data.POPULAR_NSE_STOCKS
    options = list(curated.keys()) + ["Custom Symbol..."]

    col1, col2 = st.columns([2, 1])
    with col1:
        selected_option = st.selectbox(
            "Select Stock",
            options=options,
            format_func=lambda x: f"{x} - {curated.get(x, 'Custom')}" if x in curated else x,
            key=f"{key_prefix}_stock_select"
        )
    with col2:
        if selected_option == "Custom Symbol...":
            custom_sym = st.text_input("Enter Ticker (e.g. INFY.NS, AAPL)", value="TATAMOTORS.NS", key=f"{key_prefix}_custom_sym")
            return market_data.normalize_symbol(custom_sym)
        return selected_option


def check_and_render_active_alerts(user_id: int):
    """Checks active user price alerts against real-time quotes and displays banner notifications."""
    alerts = db.get_user_alerts(user_id, active_only=True)
    if not alerts:
        return

    symbols_to_check = tuple(set(a["symbol"] for a in alerts))
    quotes = market_data.get_multiple_quotes(symbols_to_check)

    triggered_now = []
    for alert in alerts:
        sym = alert["symbol"]
        quote = quotes.get(sym, {})
        curr_price = quote.get("current_price", 0.0)
        target = alert["target_price"]
        condition = alert["condition"]

        if curr_price > 0:
            if condition == "ABOVE" and curr_price >= target:
                triggered_now.append((alert, curr_price))
            elif condition == "BELOW" and curr_price <= target:
                triggered_now.append((alert, curr_price))

    if triggered_now:
        for alert, curr_price in triggered_now:
            st.toast(f"🔔 ALERT: {alert['symbol']} reached ₹{curr_price:,.2f} ({alert['condition']} ₹{alert['target_price']:,.2f})!", icon="🎯")
            st.warning(f"🎯 **PRICE ALERT TRIGGERED:** **{alert['symbol']}** is currently trading at **₹{curr_price:,.2f}**, reaching your target of **{alert['condition']} ₹{alert['target_price']:,.2f}**!")
            db.mark_alert_triggered(alert["id"])


# -----------------------------------------------------------------------------
# PAGE 1: Dashboard
# -----------------------------------------------------------------------------
def render_dashboard_page(user: dict):
    st.markdown("<div class='main-header'>📊 Market Overview & Stock Dashboard</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Real-time price quotes, technical candlestick trends, and volume dynamics</div>", unsafe_allow_html=True)

    symbol = get_selected_stock(key_prefix="dash")
    period = st.radio("Chart Horizon", ["1mo", "6mo", "1y", "5y"], horizontal=True, index=2, key="dash_period")

    # Fetch live quote & historical candles
    with st.spinner(f"Loading market data for {symbol}..."):
        quote = market_data.get_stock_quote(symbol)
        df, err = market_data.get_stock_data(symbol, period=period)

    # Metric Cards Row
    if quote.get("valid"):
        curr_price = quote["current_price"]
        change = quote["change"]
        change_pct = quote["change_percent"]
        delta_class = "metric-delta-pos" if change >= 0 else "metric-delta-neg"
        delta_sign = "+" if change >= 0 else ""

        mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
        with mcol1:
            st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>Current Price</div>
                    <div class='metric-value'>₹{curr_price:,.2f}</div>
                    <div class='{delta_class}'>{delta_sign}₹{change:,.2f} ({delta_sign}{change_pct:.2f}%)</div>
                </div>
            """, unsafe_allow_html=True)
        with mcol2:
            st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>Day Range</div>
                    <div class='metric-value' style='font-size:1.15rem;'>₹{quote['day_low']:,.1f} - ₹{quote['day_high']:,.1f}</div>
                    <div class='metric-label' style='margin-top:4px;'>Prev Close: ₹{quote['previous_close']:,.2f}</div>
                </div>
            """, unsafe_allow_html=True)
        with mcol3:
            st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>52-Week High / Low</div>
                    <div class='metric-value' style='font-size:1.15rem;'>₹{quote['fifty_two_week_low']:,.1f} - ₹{quote['fifty_two_week_high']:,.1f}</div>
                    <div class='metric-label' style='margin-top:4px;'>Currency: {quote['currency']}</div>
                </div>
            """, unsafe_allow_html=True)
        with mcol4:
            st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>24h Volume</div>
                    <div class='metric-value'>{quote['volume']:,}</div>
                    <div class='metric-label' style='margin-top:4px;'>Asset: {quote['name'][:18]}</div>
                </div>
            """, unsafe_allow_html=True)
        with mcol5:
            # Quick Watchlist Toggle
            watchlist = db.get_watchlist(user["id"])
            in_watchlist = symbol in watchlist
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.markdown("<div class='metric-label'>Watchlist Status</div>", unsafe_allow_html=True)
            if in_watchlist:
                if st.button("⭐ Remove Watchlist", key="btn_rem_wl", use_container_width=True):
                    db.remove_from_watchlist(user["id"], symbol)
                    st.rerun()
            else:
                if st.button("➕ Add to Watchlist", key="btn_add_wl", use_container_width=True):
                    db.add_to_watchlist(user["id"], symbol)
                    st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    if err or df is None:
        st.error(f"❌ {err or 'Unable to fetch historical data.'}")
        return

    # Candlestick chart
    st.markdown("#### 🕯️ Candlestick Price & Volume Action")
    fig = indicators.create_candlestick_chart(
        df,
        symbol=symbol,
        show_sma20=True,
        show_sma50=True,
        show_bollinger=False,
        show_volume=True
    )
    st.plotly_chart(fig, use_container_width=True)

    # Quick Trade Bar
    st.markdown("---")
    st.markdown("#### ⚡ Quick Order Execution")
    qcol1, qcol2, qcol3 = st.columns([1, 1, 2])
    with qcol1:
        quick_qty = st.number_input("Shares", min_value=1, max_value=10000, value=10, step=1, key="dash_quick_qty")
    with qcol2:
        quick_price = quote.get("current_price", 0.0)
        st.metric("Estimated Cost", f"₹{(quick_qty * quick_price):,.2f}")
    with qcol3:
        bcol1, bcol2 = st.columns(2)
        with bcol1:
            st.write("")
            st.write("")
            if st.button(f"🟢 Quick BUY {quick_qty} Shares", use_container_width=True, key="dash_quick_buy"):
                res, msg = db.execute_trade(user["id"], symbol, "BUY", quick_qty, quick_price)
                if res:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)
        with bcol2:
            st.write("")
            st.write("")
            if st.button(f"🔴 Quick SELL {quick_qty} Shares", use_container_width=True, key="dash_quick_sell"):
                res, msg = db.execute_trade(user["id"], symbol, "SELL", quick_qty, quick_price)
                if res:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)


# -------------------------------------------------------------
# PAGE 2: Technical Indicators
# -------------------------------------------------------------
def render_indicators_page(user: dict):
    st.markdown("<div class='main-header'>📈 Technical Indicators & Analysis</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Interactive overlay toggles with plain-English algorithmic breakdowns</div>", unsafe_allow_html=True)

    symbol = get_selected_stock(key_prefix="ind")
    period = st.radio("Lookback Period", ["1mo", "6mo", "1y", "5y"], horizontal=True, index=2, key="ind_period")

    with st.spinner("Calculating indicators..."):
        df, err = market_data.get_stock_data(symbol, period=period)

    if err or df is None:
        st.error(err or "Unable to load data.")
        return

    # Toggles Row
    st.markdown("#### 🛠️ Indicator Chart Overlays")
    tcol1, tcol2, tcol3, tcol4 = st.columns(4)
    with tcol1:
        toggle_sma20 = st.toggle("Moving Average 20 (SMA)", value=True, key="tog_sma20")
    with tcol2:
        toggle_sma50 = st.toggle("Moving Average 50 (SMA)", value=True, key="tog_sma50")
    with tcol3:
        toggle_bb = st.toggle("Bollinger Bands (20, 2)", value=True, key="tog_bb")
    with tcol4:
        toggle_vol = st.toggle("Trading Volume", value=True, key="tog_vol")

    # Render main chart with selected overlays
    fig_main = indicators.create_candlestick_chart(
        df,
        symbol=symbol,
        show_sma20=toggle_sma20,
        show_sma50=toggle_sma50,
        show_bollinger=toggle_bb,
        show_volume=toggle_vol
    )
    st.plotly_chart(fig_main, use_container_width=True)

    # Sub-Charts: RSI & MACD
    st.markdown("---")
    rcol1, rcol2 = st.columns(2)
    with rcol1:
        fig_rsi = indicators.create_rsi_chart(df, symbol)
        st.plotly_chart(fig_rsi, use_container_width=True)
    with rcol2:
        fig_macd = indicators.create_macd_chart(df, symbol)
        st.plotly_chart(fig_macd, use_container_width=True)

    # Plain English Explanations
    st.markdown("---")
    st.markdown("### 📚 Plain-English Indicator Guide")
    exp_col1, exp_col2 = st.columns(2)
    with exp_col1:
        with st.expander("📖 What are Moving Averages (20 & 50)?", expanded=True):
            st.markdown(indicators.INDICATOR_EXPLANATIONS["MA"])
        with st.expander("📖 What is the Relative Strength Index (RSI)?", expanded=True):
            st.markdown(indicators.INDICATOR_EXPLANATIONS["RSI"])

    with exp_col2:
        with st.expander("📖 What are Bollinger Bands?", expanded=True):
            st.markdown(indicators.INDICATOR_EXPLANATIONS["BB"])
        with st.expander("📖 What is MACD (Moving Average Convergence Divergence)?", expanded=True):
            st.markdown(indicators.INDICATOR_EXPLANATIONS["MACD"])


# -------------------------------------------------------------
# PAGE 3: Trade Terminal
# -------------------------------------------------------------
def render_trade_page(user: dict):
    st.markdown("<div class='main-header'>💹 Paper Trading Order Terminal</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Execute simulated Buy & Sell orders with real-time balance validation</div>", unsafe_allow_html=True)

    symbol = get_selected_stock(key_prefix="trade")
    quote = market_data.get_stock_quote(symbol)

    curr_price = quote.get("current_price", 0.0)
    current_cash = user.get("cash_balance", 0.0)
    holding = db.get_holding(user["id"], symbol)
    owned_qty = holding["quantity"] if holding else 0
    avg_buy = holding["avg_buy_price"] if holding else 0.0

    # User Status Bar
    scol1, scol2, scol3, scol4 = st.columns(4)
    with scol1:
        st.metric("Available Cash", f"₹{current_cash:,.2f}")
    with scol2:
        st.metric(f"Current Price ({symbol})", f"₹{curr_price:,.2f}")
    with scol3:
        st.metric("Shares Currently Owned", f"{owned_qty} units")
    with scol4:
        st.metric("Avg Purchase Price", f"₹{avg_buy:,.2f}" if owned_qty > 0 else "N/A")

    st.markdown("---")

    # Order Entry Box
    ocol1, ocol2 = st.columns([1, 1], gap="large")

    with ocol1:
        st.markdown("### 📝 Place Order")
        trade_action = st.radio("Order Type", ["BUY", "SELL"], horizontal=True, key="trade_action")
        qty = st.number_input("Order Quantity", min_value=1, max_value=50000, value=10, step=1, key="trade_qty")
        
        # Limit price or Market price
        order_price_type = st.radio("Price Execution", ["Market Price", "Custom / Limit Price"], horizontal=True)
        if order_price_type == "Market Price":
            exec_price = curr_price
            st.info(f"Executing at live market price: **₹{exec_price:,.2f}**")
        else:
            exec_price = st.number_input("Limit Price (₹)", min_value=0.05, value=curr_price if curr_price > 0 else 100.0, step=0.5)

        total_val = round(qty * exec_price, 2)
        st.markdown(f"#### Order Total: **₹{total_val:,.2f}**")

        # Validation feedback
        if trade_action == "BUY":
            if total_val > current_cash:
                st.error(f"⚠️ Insufficient Cash! Need ₹{total_val:,.2f}, have ₹{current_cash:,.2f}.")
                btn_disabled = True
            else:
                st.success(f"✅ Cash remaining after trade: ₹{(current_cash - total_val):,.2f}")
                btn_disabled = False
        else:
            if qty > owned_qty:
                st.error(f"⚠️ Insufficient Shares! You own {owned_qty} shares, requested to sell {qty}.")
                btn_disabled = True
            else:
                pnl = round((exec_price - avg_buy) * qty, 2)
                pnl_color = "green" if pnl >= 0 else "red"
                st.markdown(f"Estimated Realized P&L on this sell: <b style='color:{pnl_color};'>₹{pnl:,.2f}</b>", unsafe_allow_html=True)
                btn_disabled = False

        st.write("")
        submit_trade = st.button(
            f"Confirm {trade_action} Order",
            type="primary" if trade_action == "BUY" else "secondary",
            disabled=btn_disabled or exec_price <= 0,
            use_container_width=True
        )

        if submit_trade:
            success, msg = db.execute_trade(user["id"], symbol, trade_action, int(qty), float(exec_price))
            if success:
                st.balloons()
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)

    with ocol2:
        st.markdown("### 📋 Position Overview")
        if owned_qty > 0:
            market_val = round(owned_qty * curr_price, 2)
            invested_val = round(owned_qty * avg_buy, 2)
            unrealized_pnl = round(market_val - invested_val, 2)
            unrealized_pct = round((unrealized_pnl / invested_val) * 100, 2) if invested_val > 0 else 0.0
            pnl_color = "#22c55e" if unrealized_pnl >= 0 else "#ef4444"

            st.markdown(f"""
                <div class='metric-card' style='margin-bottom: 15px;'>
                    <div class='metric-label'>Position Value ({symbol})</div>
                    <div class='metric-value'>₹{market_val:,.2f}</div>
                    <div style='color: {pnl_color}; font-weight:600; font-size:1.1rem; margin-top:5px;'>
                        Unrealized P&L: ₹{unrealized_pnl:,.2f} ({unrealized_pct:+.2f}%)
                    </div>
                </div>
            """, unsafe_allow_html=True)
            st.write(f"- **Invested Capital:** ₹{invested_val:,.2f}")
            st.write(f"- **Average Entry:** ₹{avg_buy:,.2f}")
            st.write(f"- **Current Quote:** ₹{curr_price:,.2f}")
        else:
            st.info(f"You currently have no open positions in **{symbol}**.")

        # Recent trades for this stock
        st.markdown("#### Recent Trades for this Stock")
        user_trades = db.get_user_trades(user["id"], limit=50)
        stock_trades = [t for t in user_trades if t["symbol"] == symbol]
        if stock_trades:
            st.dataframe(
                pd.DataFrame(stock_trades)[["timestamp", "trade_type", "quantity", "price", "total_amount"]],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.caption("No historical executions recorded for this symbol yet.")


# -------------------------------------------------------------
# PAGE 4: Portfolio Management
# -------------------------------------------------------------
def render_portfolio_page(user: dict):
    st.markdown("<div class='main-header'>💼 Portfolio Management & Performance</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Holdings analysis, capital allocation, P&L reporting, and full trade ledger</div>", unsafe_allow_html=True)

    holdings = db.get_user_holdings(user["id"])
    cash_left = user.get("cash_balance", 0.0)

    # Fetch live quotes for all current holdings
    holding_symbols = tuple(h["symbol"] for h in holdings)
    quotes = market_data.get_multiple_quotes(holding_symbols) if holding_symbols else {}

    # Calculate portfolio values
    total_invested = 0.0
    total_market_val = 0.0
    processed_holdings = []

    for h in holdings:
        sym = h["symbol"]
        qty = h["quantity"]
        avg_price = h["avg_buy_price"]
        q = quotes.get(sym, {})
        curr_price = q.get("current_price", avg_price)
        if curr_price <= 0:
            curr_price = avg_price

        invested = round(qty * avg_price, 2)
        cur_val = round(qty * curr_price, 2)
        pnl = round(cur_val - invested, 2)
        pnl_pct = round((pnl / invested) * 100, 2) if invested > 0 else 0.0

        total_invested += invested
        total_market_val += cur_val

        processed_holdings.append({
            "Symbol": sym,
            "Quantity": qty,
            "Avg Buy (₹)": avg_price,
            "Current Price (₹)": curr_price,
            "Invested (₹)": invested,
            "Current Value (₹)": cur_val,
            "P&L (₹)": pnl,
            "P&L (%)": pnl_pct
        })

    total_portfolio_value = round(cash_left + total_market_val, 2)
    total_unrealized_pnl = round(total_market_val - total_invested, 2)
    total_pnl_pct = round((total_unrealized_pnl / total_invested) * 100, 2) if total_invested > 0 else 0.0

    # Overall return on ₹1,00,000 initial capital
    initial_capital = 100000.0
    net_overall_return = round(total_portfolio_value - initial_capital, 2)
    net_overall_pct = round((net_overall_return / initial_capital) * 100, 2)

    # Top KPI Cards
    pcol1, pcol2, pcol3, pcol4 = st.columns(4)
    with pcol1:
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Total Portfolio Value</div>
                <div class='metric-value'>₹{total_portfolio_value:,.2f}</div>
                <div class='{"metric-delta-pos" if net_overall_return >= 0 else "metric-delta-neg"}'>
                    {"+" if net_overall_return >= 0 else ""}₹{net_overall_return:,.2f} ({"+" if net_overall_pct >= 0 else ""}{net_overall_pct:.2f}% Total)
                </div>
            </div>
        """, unsafe_allow_html=True)
    with pcol2:
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Unrealized P&L</div>
                <div class='metric-value'>₹{total_unrealized_pnl:,.2f}</div>
                <div class='{"metric-delta-pos" if total_unrealized_pnl >= 0 else "metric-delta-neg"}'>
                    {"+" if total_pnl_pct >= 0 else ""}{total_pnl_pct:.2f}% on holdings
                </div>
            </div>
        """, unsafe_allow_html=True)
    with pcol3:
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Available Cash</div>
                <div class='metric-value'>₹{cash_left:,.2f}</div>
                <div class='metric-label' style='margin-top:4px;'>Free Liquid Balance</div>
            </div>
        """, unsafe_allow_html=True)
    with pcol4:
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Active Positions</div>
                <div class='metric-value'>{len(holdings)} Stocks</div>
                <div class='metric-label' style='margin-top:4px;'>Invested: ₹{total_invested:,.2f}</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Holdings Table and Asset Allocation Pie
    hcol1, hcol2 = st.columns([1.6, 1], gap="medium")

    with hcol1:
        st.markdown("#### 📋 Stock Holdings")
        if processed_holdings:
            hdf = pd.DataFrame(processed_holdings)
            st.dataframe(
                hdf.style.format({
                    "Avg Buy (₹)": "₹{:,.2f}",
                    "Current Price (₹)": "₹{:,.2f}",
                    "Invested (₹)": "₹{:,.2f}",
                    "Current Value (₹)": "₹{:,.2f}",
                    "P&L (₹)": "₹{:,.2f}",
                    "P&L (%)": "{:+.2f}%"
                }),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No active holdings found in your portfolio. Visit the **Trade Terminal** to execute your first paper trade!")

    with hcol2:
        st.markdown("#### 🥧 Asset Allocation")
        allocation_data = [{"Asset": "Available Cash", "Value": cash_left}]
        for ph in processed_holdings:
            allocation_data.append({"Asset": ph["Symbol"], "Value": ph["Current Value (₹)"]})

        alloc_df = pd.DataFrame(allocation_data)
        fig_pie = px.pie(
            alloc_df,
            names="Asset",
            values="Value",
            hole=0.45,
            color_discrete_sequence=px.colors.qualitative.Dark24
        )
        fig_pie.update_layout(
            template="plotly_dark",
            plot_bgcolor="#11141A",
            paper_bgcolor="#11141A",
            margin=dict(l=20, r=20, t=20, b=20),
            height=320,
            showlegend=True
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # Trade History Table
    st.markdown("---")
    st.markdown("#### 📜 Trade Execution History")
    trades = db.get_user_trades(user["id"], limit=100)
    if trades:
        tdf = pd.DataFrame(trades)
        st.dataframe(
            tdf[["timestamp", "symbol", "trade_type", "quantity", "price", "total_amount"]].rename(columns={
                "timestamp": "Timestamp",
                "symbol": "Symbol",
                "trade_type": "Action",
                "quantity": "Qty",
                "price": "Price (₹)",
                "total_amount": "Total (₹)"
            }),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.caption("No trade transactions executed yet.")


# -------------------------------------------------------------
# PAGE 5: Price Prediction (Scikit-Learn)
# -------------------------------------------------------------
def render_prediction_page(user: dict):
    st.markdown("<div class='main-header'>🤖 AI Price Prediction (Next 7 Days)</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Multi-step forward forecast using supervised Machine Learning</div>", unsafe_allow_html=True)

    st.warning(prediction.PREDICTION_DISCLAIMER)

    symbol = get_selected_stock(key_prefix="pred")

    pcol1, pcol2 = st.columns([1, 1])
    with pcol1:
        model_choice = st.selectbox("Machine Learning Algorithm", ["Linear Regression", "Random Forest"], key="pred_model")
    with pcol2:
        lookback = st.slider("Historical Chart Context (Trading Days)", min_value=30, max_value=180, value=60, step=15)

    with st.spinner(f"Training {model_choice} model and projecting 7-day trajectory for {symbol}..."):
        df, err = market_data.get_stock_data(symbol, period="1y")

        if err or df is None:
            st.error(err or "Unable to retrieve data for prediction.")
            return

        res = prediction.train_and_predict_7days(df, model_name=model_choice, n_days=7)

    if not res.get("success"):
        st.error(res.get("error", "Prediction failed."))
        return

    # Model Evaluation Metric Cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Current Price", f"₹{res['current_price']:,.2f}")
    with m2:
        proj_chg = res["pct_change_7d"]
        st.metric(
            "7-Day Forecasted Price",
            f"₹{res['predicted_price_7d']:,.2f}",
            delta=f"{proj_chg:+.2f}%",
            delta_color="normal"
        )
    with m3:
        st.metric("Test R² Score", f"{res['r2_score']:.4f}")
    with m4:
        st.metric("RMSE Error", f"±₹{res['rmse']:,.2f}")

    # Forecast Plotly Chart
    fig_pred = prediction.create_prediction_chart(df, res, symbol=symbol, lookback_days=lookback)
    st.plotly_chart(fig_pred, use_container_width=True)

    # 7-Day Day-by-Day Forecast Table
    st.markdown("#### 📅 Day-by-Day Projected Trajectory")
    forecast_rows = []
    base_price = res["current_price"]
    for date, price in zip(res["future_dates"], res["future_prices"]):
        diff = price - base_price
        pct = (diff / base_price) * 100 if base_price > 0 else 0
        forecast_rows.append({
            "Date": date.strftime("%A, %d %b %Y"),
            "Forecasted Close (₹)": f"₹{price:,.2f}",
            "Change from Current (₹)": f"{'+' if diff >= 0 else ''}₹{diff:,.2f}",
            "Change (%)": f"{'+' if pct >= 0 else ''}{pct:.2f}%"
        })
    st.dataframe(pd.DataFrame(forecast_rows), use_container_width=True, hide_index=True)


# -------------------------------------------------------------
# PAGE 6: Risk Analysis & Portfolio Analytics
# -------------------------------------------------------------
def render_risk_page(user: dict):
    st.markdown("<div class='main-header'>🛡️ Risk Analysis & Portfolio Metrics</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Annualized Volatility, Sharpe Ratio, Max Drawdown, and Diversification Scoring</div>", unsafe_allow_html=True)

    holdings = db.get_user_holdings(user["id"])

    # Collect data for portfolio risk computation
    holding_symbols = [h["symbol"] for h in holdings]
    quotes = market_data.get_multiple_quotes(tuple(holding_symbols)) if holding_symbols else {}
    current_prices = {sym: quotes[sym]["current_price"] for sym in holding_symbols if sym in quotes}

    hist_dfs = {}
    for sym in holding_symbols:
        df, _ = market_data.get_stock_data(sym, period="1y")
        if df is not None:
            hist_dfs[sym] = df

    port_risk = risk.calculate_portfolio_risk(holdings, current_prices, hist_dfs)

    st.markdown("### 💼 Portfolio-Level Risk Profile")
    if not port_risk.get("has_holdings"):
        st.info("ℹ️ " + port_risk.get("message", "No holdings found."))
    else:
        # Diversification Score & Metrics
        rcol1, rcol2, rcol3, rcol4 = st.columns(4)
        with rcol1:
            div_score = port_risk["diversification_score"]
            div_color = port_risk["diversification_color"]
            st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>Diversification Score</div>
                    <div class='metric-value' style='color:{div_color};'>{div_score}/100</div>
                    <div style='font-size:0.85rem; font-weight:600; color:{div_color};'>{port_risk['diversification_label']}</div>
                </div>
            """, unsafe_allow_html=True)
        with rcol2:
            st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>Annualized Volatility</div>
                    <div class='metric-value'>{port_risk['annualized_volatility']:.2f}%</div>
                    <div class='metric-label' style='margin-top:4px;'>Annualized Standard Deviation</div>
                </div>
            """, unsafe_allow_html=True)
        with rcol3:
            st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>Sharpe Ratio</div>
                    <div class='metric-value'>{port_risk['sharpe_ratio']:.2f}</div>
                    <div class='metric-label' style='margin-top:4px;'>Rf Benchmark: 6.50%</div>
                </div>
            """, unsafe_allow_html=True)
        with rcol4:
            st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>Max Drawdown (MDD)</div>
                    <div class='metric-value' style='color:#ef4444;'>{port_risk['max_drawdown']:.2f}%</div>
                    <div class='metric-label' style='margin-top:4px;'>Peak to Trough Decline</div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Single Stock Risk Analyzer
    st.markdown("### 🔍 Individual Stock Risk Analysis")
    selected_stock = get_selected_stock(key_prefix="risk")

    with st.spinner(f"Computing risk metrics for {selected_stock}..."):
        df_stock, err_stock = market_data.get_stock_data(selected_stock, period="1y")

    if df_stock is not None and not err_stock:
        s_metrics = risk.calculate_stock_risk_metrics(df_stock)
        if s_metrics.get("valid"):
            scol1, scol2, scol3, scol4 = st.columns(4)
            with scol1:
                st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-label'>Volatility Classification</div>
                        <div class='metric-value' style='color:{s_metrics["volatility_color"]}; font-size:1.25rem;'>{s_metrics["volatility_tier"]}</div>
                        <div style='color:{s_metrics["volatility_color"]}; font-weight:600;'>{s_metrics["annualized_volatility"]}% Ann. Vol</div>
                    </div>
                """, unsafe_allow_html=True)
            with scol2:
                st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-label'>Sharpe Ratio</div>
                        <div class='metric-value'>{s_metrics["sharpe_ratio"]}</div>
                        <div class='metric-label' style='margin-top:4px;'>Risk-Adjusted Return</div>
                    </div>
                """, unsafe_allow_html=True)
            with scol3:
                st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-label'>Annualized Return (CAGR)</div>
                        <div class='metric-value' style='color:{"#22c55e" if s_metrics["cagr"] >= 0 else "#ef4444"};'>{s_metrics["cagr"]}%</div>
                        <div class='metric-label' style='margin-top:4px;'>1-Year Growth Rate</div>
                    </div>
                """, unsafe_allow_html=True)
            with scol4:
                st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-label'>Maximum Drawdown</div>
                        <div class='metric-value' style='color:#ef4444;'>{s_metrics["max_drawdown"]}%</div>
                        <div class='metric-label' style='margin-top:4px;'>Worst Historic Drop</div>
                    </div>
                """, unsafe_allow_html=True)
    else:
        st.error(err_stock or "Failed to compute risk metrics.")

    # Educational Guide to Risk Metrics
    st.markdown("---")
    st.markdown("#### 📚 Understanding Risk Parameters")
    with st.expander("ℹ️ How are Sharpe Ratio, Volatility, and Drawdown defined?"):
        st.markdown("""
        - **Annualized Volatility:** The annualized standard deviation of daily percentage price returns. It quantifies how violently a stock's price swings. Volatilities below 20% are typically defensive bluechips, while those above 35% are high-beta or speculative.
        - **Sharpe Ratio:** Measures excess return earned per unit of total risk taken above the risk-free rate (6.5% repo rate). A Sharpe ratio above 1.0 is considered good; above 2.0 is exceptional.
        - **Maximum Drawdown (MDD):** The observed peak-to-trough decline before a new peak is achieved. It indicates the worst-case historical loss an investor would have endured.
        - **Diversification Score:** Uses the Herfindahl-Hirschman Index (HHI) to measure portfolio asset concentration. Distributing capital across 5+ non-correlated assets increases portfolio resilience and minimizes single-stock ruin risk.
        """)


# -------------------------------------------------------------
# PAGE 7: Watchlist & Price Alerts
# -------------------------------------------------------------
def render_watchlist_page(user: dict):
    st.markdown("<div class='main-header'>⭐ Watchlist & Target Price Alerts</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Monitor key tickers and trigger alerts when market prices cross your targets</div>", unsafe_allow_html=True)

    wcol1, wcol2 = st.columns([1.2, 1], gap="large")

    with wcol1:
        st.markdown("### 📋 Active Watchlist")
        watchlist_symbols = db.get_watchlist(user["id"])

        # Add new symbol form
        with st.form("add_watchlist_form"):
            ncol1, ncol2 = st.columns([2, 1])
            with ncol1:
                new_sym = st.selectbox(
                    "Add Stock",
                    options=list(market_data.POPULAR_NSE_STOCKS.keys()) + ["Custom..."]
                )
                if new_sym == "Custom...":
                    new_sym = st.text_input("Enter Ticker", value="WIPRO.NS")
            with ncol2:
                st.write("")
                st.write("")
                add_btn = st.form_submit_button("➕ Add to Watchlist", use_container_width=True)

            if add_btn and new_sym:
                res, msg = db.add_to_watchlist(user["id"], market_data.normalize_symbol(new_sym))
                if res:
                    st.success(msg)
                    st.rerun()
                else:
                    st.warning(msg)

        if watchlist_symbols:
            quotes = market_data.get_multiple_quotes(tuple(watchlist_symbols))
            watchlist_data = []

            for sym in watchlist_symbols:
                q = quotes.get(sym, {})
                c_price = q.get("current_price", 0.0)
                chg = q.get("change", 0.0)
                chg_pct = q.get("change_percent", 0.0)
                watchlist_data.append({
                    "Symbol": sym,
                    "Name": q.get("name", sym),
                    "Price (₹)": f"₹{c_price:,.2f}",
                    "Change (₹)": f"{'+' if chg >= 0 else ''}₹{chg:,.2f}",
                    "Change (%)": f"{'+' if chg_pct >= 0 else ''}{chg_pct:.2f}%"
                })

            st.dataframe(pd.DataFrame(watchlist_data), use_container_width=True, hide_index=True)

            # Quick removal
            rem_sym = st.selectbox("Remove from Watchlist", options=watchlist_symbols, key="rem_wl_select")
            if st.button(f"🗑️ Remove {rem_sym}", key="btn_remove_wl"):
                db.remove_from_watchlist(user["id"], rem_sym)
                st.rerun()
        else:
            st.info("Your watchlist is currently empty.")

    with wcol2:
        st.markdown("### 🔔 Set Price Alert")
        with st.form("new_alert_form"):
            alert_sym = st.selectbox(
                "Stock Symbol",
                options=watchlist_symbols if watchlist_symbols else list(market_data.POPULAR_NSE_STOCKS.keys()),
                key="alert_sym_select"
            )
            # Default target to current quote price
            current_q = market_data.get_stock_quote(alert_sym)
            c_price = current_q.get("current_price", 100.0)

            acol1, acol2 = st.columns(2)
            with acol1:
                condition = st.selectbox("Trigger Condition", ["ABOVE", "BELOW"])
            with acol2:
                target_p = st.number_input(
                    "Target Price (₹)",
                    min_value=0.1,
                    value=float(round(c_price * 1.05 if condition == "ABOVE" else c_price * 0.95, 2)),
                    step=1.0
                )

            submit_alert = st.form_submit_button("🔔 Create Alert", use_container_width=True)
            if submit_alert:
                res, msg = db.create_alert(user["id"], alert_sym, target_p, condition)
                if res:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)

        # Existing alerts list
        st.markdown("#### Your Saved Alerts")
        alerts = db.get_user_alerts(user["id"])
        if alerts:
            for a in alerts:
                status_icon = "✅ TRIGGERED" if a["is_triggered"] else "⏳ ACTIVE"
                color = "#22c55e" if a["is_triggered"] else "#38bdf8"
                acol1, acol2 = st.columns([3, 1])
                with acol1:
                    st.markdown(f"""
                        <div style='background:#151a23; padding:8px 12px; border-radius:8px; margin-bottom:6px; border:1px solid #222b3a;'>
                            <b>{a['symbol']}</b>: Trigger when <b>{a['condition']} ₹{a['target_price']:,.2f}</b>
                            <span style='color:{color}; font-weight:600; margin-left:8px; font-size:0.8rem;'>[{status_icon}]</span>
                        </div>
                    """, unsafe_allow_html=True)
                with acol2:
                    if st.button("Delete", key=f"del_alert_{a['id']}"):
                        db.delete_alert(a["id"], user["id"])
                        st.rerun()
        else:
            st.caption("No alerts configured.")


# -------------------------------------------------------------
# Common Footer
# -------------------------------------------------------------
def render_footer():
    st.markdown("""
        <div class='app-footer'>
            🛡️ <b>Advance Stock Trading and Portfolio Management Platform</b><br>
            ⚠️ <b>Disclaimer:</b> For educational purposes only, not financial advice. Past performance and machine learning forecasts do not guarantee future market returns.
        </div>
    """, unsafe_allow_html=True)


# -------------------------------------------------------------
# Main Application Flow
# -------------------------------------------------------------
def main():
    if not auth.is_authenticated():
        render_auth_page()
        return

    # User is authenticated
    current_user = auth.get_current_user()

    # Sidebar Navigation & User Info
    with st.sidebar:
        st.markdown("### 👤 Trader Profile")
        st.markdown(f"**Trader:** `{current_user['username']}`")
        st.markdown(f"**Virtual Cash:** `₹{current_user['cash_balance']:,.2f}`")

        st.markdown("---")
        st.markdown("### 🧭 Navigation")
        page = st.radio(
            "Go to",
            [
                "📊 Stock Dashboard",
                "📈 Technical Indicators",
                "💹 Trade Terminal",
                "💼 Portfolio Management",
                "🤖 Price Prediction",
                "🛡️ Risk & Analytics",
                "⭐ Watchlist & Alerts"
            ],
            key="nav_page"
        )

        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            auth.logout_session()
            st.rerun()

    # Check for active price alerts across app navigation
    check_and_render_active_alerts(current_user["id"])

    # Route to selected page
    if page == "📊 Stock Dashboard":
        render_dashboard_page(current_user)
    elif page == "📈 Technical Indicators":
        render_indicators_page(current_user)
    elif page == "💹 Trade Terminal":
        render_trade_page(current_user)
    elif page == "💼 Portfolio Management":
        render_portfolio_page(current_user)
    elif page == "🤖 Price Prediction":
        render_prediction_page(current_user)
    elif page == "🛡️ Risk & Analytics":
        render_risk_page(current_user)
    elif page == "⭐ Watchlist & Alerts":
        render_watchlist_page(current_user)

    render_footer()


if __name__ == "__main__":
    main()
