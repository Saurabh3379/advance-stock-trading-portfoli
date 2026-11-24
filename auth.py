"""
auth.py - Authentication module for Advance Stock Trading and Portfolio Management.
Provides secure password hashing (PBKDF2-HMAC-SHA256) with unique per-user salts,
user registration with Rs 1,00,000 virtual balance, login verification,
and Streamlit session state management.
"""

import hashlib
import os
import secrets
from typing import Tuple, Optional, Dict, Any
import streamlit as st
import database as db


def generate_salt(length: int = 16) -> str:
    """Generates a secure random hexadecimal salt string."""
    return secrets.token_hex(length)


def hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
    """
    Hashes a plain text password using PBKDF2-HMAC-SHA256 with 100,000 iterations.
    Returns (hex_hash, hex_salt).
    """
    if salt is None:
        salt = generate_salt()
    
    key = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt.encode('utf-8'),
        100000
    )
    return key.hex(), salt


def verify_password(password: str, salt: str, expected_hash: str) -> bool:
    """Verifies a password against the stored salt and hash using constant-time comparison."""
    calculated_hash, _ = hash_password(password, salt)
    return secrets.compare_digest(calculated_hash, expected_hash)


def register_user(username: str, password: str, confirm_password: str) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Registers a new user after validation.
    Initializes user with Rs 1,00,000 virtual cash balance.
    """
    username = username.strip()
    if not username:
        return False, "Username cannot be empty.", None
    if len(username) < 3:
        return False, "Username must be at least 3 characters long.", None
    if not password:
        return False, "Password cannot be empty.", None
    if len(password) < 6:
        return False, "Password must be at least 6 characters long.", None
    if password != confirm_password:
        return False, "Passwords do not match.", None

    pwd_hash, salt = hash_password(password)
    success, message, user_id = db.create_user(username, pwd_hash, salt, initial_balance=100000.0)

    if success and user_id:
        user = db.get_user_by_id(user_id)
        return True, message, user
    return False, message, None


def login_user(username: str, password: str) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Authenticates a user against database records.
    """
    username = username.strip()
    if not username or not password:
        return False, "Please enter both username and password.", None

    user = db.get_user_by_username(username)
    if not user:
        return False, "Invalid username or password.", None

    if verify_password(password, user["salt"], user["password_hash"]):
        # Refresh latest user data
        latest_user = db.get_user_by_id(user["id"])
        return True, "Login successful!", latest_user
    else:
        return False, "Invalid username or password.", None


# -------------------------------------------------------------
# Streamlit Session State Management
# -------------------------------------------------------------

def init_session_state() -> None:
    """Initializes authentication variables in Streamlit session state."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "user" not in st.session_state:
        st.session_state.user = None


def login_session(user_dict: Dict[str, Any]) -> None:
    """Sets session state for authenticated user."""
    st.session_state.authenticated = True
    st.session_state.user = user_dict


def logout_session() -> None:
    """Clears user session state to log out."""
    st.session_state.authenticated = False
    st.session_state.user = None


def is_authenticated() -> bool:
    """Checks whether a user is currently logged in."""
    init_session_state()
    return bool(st.session_state.authenticated and st.session_state.user)


def get_current_user() -> Optional[Dict[str, Any]]:
    """
    Returns fresh user record from DB for current session, ensuring
    cash balance is always up to date.
    """
    init_session_state()
    user = st.session_state.user
    if user and "id" in user:
        fresh_user = db.get_user_by_id(user["id"])
        if fresh_user:
            st.session_state.user = fresh_user
            return fresh_user
    return user
