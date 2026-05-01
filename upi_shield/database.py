"""
UPI Shield - Database Layer (SQLite)
Handles all database operations for transaction storage and velocity checks.
"""

import sqlite3 
import json
from datetime import datetime, timedelta

DB_PATH = "upi_shield.db"


def init_db():
    """Create tables if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            upi_id      TEXT    NOT NULL,
            amount      REAL    NOT NULL,
            timestamp   TEXT    NOT NULL,
            risk_score  REAL    NOT NULL,
            verdict     TEXT    NOT NULL,
            features    TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_transaction(upi_id: str, amount: float, timestamp: str,
                     risk_score: float, verdict: str, features: dict):
    """Insert a new transaction record."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """INSERT INTO transactions
           (upi_id, amount, timestamp, risk_score, verdict, features)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (upi_id, amount, timestamp, risk_score, verdict, json.dumps(features))
    )
    conn.commit()
    conn.close()


def get_velocity(upi_id: str, minutes: int = 60) -> int:
    """
    Count how many transactions this UPI ID made in the last N minutes.
    High velocity → suspicious.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    cutoff = (datetime.utcnow() - timedelta(minutes=minutes)).isoformat()
    c.execute(
        "SELECT COUNT(*) FROM transactions WHERE upi_id = ? AND timestamp > ?",
        (upi_id, cutoff)
    )
    count = c.fetchone()[0]
    conn.close()
    return count


def get_recent_transactions(limit: int = 20) -> list:
    """Fetch the most recent transactions (for history table)."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """SELECT id, upi_id, amount, timestamp, risk_score, verdict
           FROM transactions
           ORDER BY id DESC
           LIMIT ?""",
        (limit,)
    )
    rows = c.fetchall()
    conn.close()
    return rows


def get_stats() -> dict:
    """Aggregate statistics for the dashboard."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("SELECT COUNT(*) FROM transactions")
    total = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM transactions WHERE verdict = 'FRAUD'")
    fraud_count = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM transactions WHERE verdict = 'SAFE'")
    safe_count = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM transactions WHERE verdict = 'SUSPICIOUS'")
    suspicious_count = c.fetchone()[0]

    c.execute("SELECT AVG(risk_score) FROM transactions")
    avg_risk = c.fetchone()[0] or 0

    conn.close()
    return {
        "total": total,
        "fraud": fraud_count,
        "safe": safe_count,
        "suspicious": suspicious_count,
        "avg_risk": round(avg_risk, 1)
    }
