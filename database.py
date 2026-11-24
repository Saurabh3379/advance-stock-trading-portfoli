"""
database.py - Database operations for Advance Stock Trading and Portfolio Management.
Manages SQLite storage for users, trades, holdings, watchlist, and price alerts.
"""

import sqlite3
import os
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple

DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trading_platform.db")


def get_connection(db_path: str = DB_FILE) -> sqlite3.Connection:
    """Creates a connection to SQLite database with row factory for dictionary-like access."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(db_path: str = DB_FILE) -> None:
    """Initializes database schema with required tables."""
    conn = get_connection(db_path)
    with conn:
        # Users table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                cash_balance REAL NOT NULL DEFAULT 100000.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Trades / Order history table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                trade_type TEXT NOT NULL CHECK(trade_type IN ('BUY', 'SELL')),
                quantity INTEGER NOT NULL CHECK(quantity > 0),
                price REAL NOT NULL CHECK(price > 0),
                total_amount REAL NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );
        """)

        # User stock holdings table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS holdings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                quantity INTEGER NOT NULL DEFAULT 0,
                avg_buy_price REAL NOT NULL DEFAULT 0.0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, symbol),
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );
        """)

        # Watchlist table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS watchlist (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, symbol),
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );
        """)

        # Price alerts table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                target_price REAL NOT NULL,
                condition TEXT NOT NULL CHECK(condition IN ('ABOVE', 'BELOW')),
                is_triggered INTEGER NOT NULL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                triggered_at TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );
        """)
    conn.close()


# -------------------------------------------------------------
# User Functions
# -------------------------------------------------------------

def create_user(username: str, password_hash: str, salt: str, initial_balance: float = 100000.0) -> Tuple[bool, str, Optional[int]]:
    """Creates a new user record with 1,00,000 virtual balance."""
    conn = get_connection()
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO users (username, password_hash, salt, cash_balance) VALUES (?, ?, ?, ?)",
                (username.strip(), password_hash, salt, initial_balance)
            )
            user_id = cursor.lastrowid
            
            # Pre-populate watchlist with popular stocks for fresh accounts
            default_stocks = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"]
            for sym in default_stocks:
                cursor.execute(
                    "INSERT OR IGNORE INTO watchlist (user_id, symbol) VALUES (?, ?)",
                    (user_id, sym)
                )
            return True, "Account created successfully with ₹1,00,000 virtual cash!", user_id
    except sqlite3.IntegrityError:
        return False, f"Username '{username}' already exists. Please choose another.", None
    except Exception as e:
        return False, f"Error creating user: {str(e)}", None
    finally:
        conn.close()


def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    """Fetches user record by username."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE LOWER(username) = LOWER(?)", (username.strip(),))
        row = cursor.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    """Fetches user record by user_id."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        row = cursor.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def update_user_balance(user_id: int, new_balance: float) -> bool:
    """Updates user's cash balance."""
    conn = get_connection()
    try:
        with conn:
            conn.execute("UPDATE users SET cash_balance = ? WHERE id = ?", (new_balance, user_id))
        return True
    except Exception:
        return False
    finally:
        conn.close()


# -------------------------------------------------------------
# Trading / Portfolio Functions
# -------------------------------------------------------------

def execute_trade(user_id: int, symbol: str, trade_type: str, quantity: int, price: float) -> Tuple[bool, str]:
    """
    Executes a paper buy or sell trade inside an atomic transaction.
    Validates virtual cash balance for BUY and owned holdings for SELL.
    """
    if quantity <= 0:
        return False, "Quantity must be greater than 0."
    if price <= 0:
        return False, "Price must be greater than 0."

    symbol = symbol.strip().upper()
    total_amount = round(quantity * price, 2)
    conn = get_connection()

    try:
        with conn:
            cursor = conn.cursor()

            # Lock and fetch user cash
            cursor.execute("SELECT cash_balance FROM users WHERE id = ?", (user_id,))
            user_row = cursor.fetchone()
            if not user_row:
                return False, "User not found."
            current_balance = user_row["cash_balance"]

            # Fetch existing holding
            cursor.execute("SELECT quantity, avg_buy_price FROM holdings WHERE user_id = ? AND symbol = ?", (user_id, symbol))
            holding_row = cursor.fetchone()
            existing_qty = holding_row["quantity"] if holding_row else 0
            existing_avg = holding_row["avg_buy_price"] if holding_row else 0.0

            if trade_type.upper() == "BUY":
                if current_balance < total_amount:
                    return False, f"Insufficient balance. Required: ₹{total_amount:,.2f}, Available: ₹{current_balance:,.2f}"

                new_balance = round(current_balance - total_amount, 2)
                new_qty = existing_qty + quantity
                # Calculate weighted average buy price
                new_avg = round(((existing_qty * existing_avg) + total_amount) / new_qty, 2)

                # Update user cash
                cursor.execute("UPDATE users SET cash_balance = ? WHERE id = ?", (new_balance, user_id))

                # Update or insert holding
                if holding_row:
                    cursor.execute(
                        "UPDATE holdings SET quantity = ?, avg_buy_price = ?, updated_at = CURRENT_TIMESTAMP WHERE user_id = ? AND symbol = ?",
                        (new_qty, new_avg, user_id, symbol)
                    )
                else:
                    cursor.execute(
                        "INSERT INTO holdings (user_id, symbol, quantity, avg_buy_price) VALUES (?, ?, ?, ?)",
                        (user_id, symbol, new_qty, new_avg)
                    )

                # Record trade history
                cursor.execute(
                    "INSERT INTO trades (user_id, symbol, trade_type, quantity, price, total_amount) VALUES (?, ?, 'BUY', ?, ?, ?)",
                    (user_id, symbol, quantity, price, total_amount)
                )

                return True, f"Successfully bought {quantity} shares of {symbol} at ₹{price:,.2f} (Total: ₹{total_amount:,.2f})"

            elif trade_type.upper() == "SELL":
                if existing_qty < quantity:
                    return False, f"Insufficient shares to sell. Owned: {existing_qty}, Requested: {quantity}"

                new_balance = round(current_balance + total_amount, 2)
                new_qty = existing_qty - quantity

                # Update user cash
                cursor.execute("UPDATE users SET cash_balance = ? WHERE id = ?", (new_balance, user_id))

                # Update holding
                if new_qty == 0:
                    cursor.execute("DELETE FROM holdings WHERE user_id = ? AND symbol = ?", (user_id, symbol))
                else:
                    cursor.execute(
                        "UPDATE holdings SET quantity = ?, updated_at = CURRENT_TIMESTAMP WHERE user_id = ? AND symbol = ?",
                        (new_qty, user_id, symbol)
                    )

                # Record trade history
                cursor.execute(
                    "INSERT INTO trades (user_id, symbol, trade_type, quantity, price, total_amount) VALUES (?, ?, 'SELL', ?, ?, ?)",
                    (user_id, symbol, quantity, price, total_amount)
                )

                return True, f"Successfully sold {quantity} shares of {symbol} at ₹{price:,.2f} (Total: ₹{total_amount:,.2f})"

            else:
                return False, f"Invalid trade type '{trade_type}'."

    except Exception as e:
        return False, f"Transaction failed: {str(e)}"
    finally:
        conn.close()


def get_user_holdings(user_id: int) -> List[Dict[str, Any]]:
    """Returns active holdings for the user."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT symbol, quantity, avg_buy_price, updated_at FROM holdings WHERE user_id = ? AND quantity > 0 ORDER BY symbol ASC",
            (user_id,)
        )
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def get_holding(user_id: int, symbol: str) -> Optional[Dict[str, Any]]:
    """Returns a specific holding for a user and symbol."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT symbol, quantity, avg_buy_price FROM holdings WHERE user_id = ? AND symbol = ?",
            (user_id, symbol.strip().upper())
        )
        row = cursor.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_user_trades(user_id: int, limit: int = 100) -> List[Dict[str, Any]]:
    """Fetches user trade execution history sorted by most recent first."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, symbol, trade_type, quantity, price, total_amount, timestamp FROM trades WHERE user_id = ? ORDER BY id DESC LIMIT ?",
            (user_id, limit)
        )
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


# -------------------------------------------------------------
# Watchlist Functions
# -------------------------------------------------------------

def get_watchlist(user_id: int) -> List[str]:
    """Returns the list of stock symbols saved in user's watchlist."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT symbol FROM watchlist WHERE user_id = ? ORDER BY added_at DESC", (user_id,))
        return [row["symbol"] for row in cursor.fetchall()]
    finally:
        conn.close()


def add_to_watchlist(user_id: int, symbol: str) -> Tuple[bool, str]:
    """Adds a stock symbol to user's watchlist."""
    symbol = symbol.strip().upper()
    conn = get_connection()
    try:
        with conn:
            conn.execute("INSERT INTO watchlist (user_id, symbol) VALUES (?, ?)", (user_id, symbol))
        return True, f"{symbol} added to Watchlist."
    except sqlite3.IntegrityError:
        return False, f"{symbol} is already in your Watchlist."
    except Exception as e:
        return False, f"Failed to add to watchlist: {str(e)}"
    finally:
        conn.close()


def remove_from_watchlist(user_id: int, symbol: str) -> Tuple[bool, str]:
    """Removes a stock symbol from user's watchlist."""
    symbol = symbol.strip().upper()
    conn = get_connection()
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM watchlist WHERE user_id = ? AND symbol = ?", (user_id, symbol))
            if cursor.rowcount > 0:
                return True, f"{symbol} removed from Watchlist."
            return False, f"{symbol} was not found in Watchlist."
    except Exception as e:
        return False, f"Failed to remove from watchlist: {str(e)}"
    finally:
        conn.close()


# -------------------------------------------------------------
# Price Alert Functions
# -------------------------------------------------------------

def create_alert(user_id: int, symbol: str, target_price: float, condition: str) -> Tuple[bool, str]:
    """Sets a target price alert for a stock."""
    symbol = symbol.strip().upper()
    condition = condition.strip().upper()
    if condition not in ('ABOVE', 'BELOW'):
        return False, "Condition must be 'ABOVE' or 'BELOW'."
    if target_price <= 0:
        return False, "Target price must be positive."

    conn = get_connection()
    try:
        with conn:
            conn.execute(
                "INSERT INTO alerts (user_id, symbol, target_price, condition) VALUES (?, ?, ?, ?)",
                (user_id, symbol, target_price, condition)
            )
        return True, f"Alert created: Notify when {symbol} is {condition} ₹{target_price:,.2f}"
    except Exception as e:
        return False, f"Failed to set alert: {str(e)}"
    finally:
        conn.close()


def get_user_alerts(user_id: int, active_only: bool = False) -> List[Dict[str, Any]]:
    """Fetches user price alerts."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        query = "SELECT * FROM alerts WHERE user_id = ?"
        params: List[Any] = [user_id]
        if active_only:
            query += " AND is_triggered = 0"
        query += " ORDER BY id DESC"
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def mark_alert_triggered(alert_id: int) -> bool:
    """Marks an alert as triggered with current timestamp."""
    conn = get_connection()
    try:
        with conn:
            conn.execute(
                "UPDATE alerts SET is_triggered = 1, triggered_at = CURRENT_TIMESTAMP WHERE id = ?",
                (alert_id,)
            )
        return True
    except Exception:
        return False
    finally:
        conn.close()


def delete_alert(alert_id: int, user_id: int) -> Tuple[bool, str]:
    """Deletes an alert."""
    conn = get_connection()
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM alerts WHERE id = ? AND user_id = ?", (alert_id, user_id))
            if cursor.rowcount > 0:
                return True, "Alert removed."
            return False, "Alert not found."
    except Exception as e:
        return False, f"Error deleting alert: {str(e)}"
    finally:
        conn.close()
