"""
Shared fraud-analysis logic for UPI Shield.

Both the FastAPI backend and the Streamlit app import from here so they
stay in sync on feature extraction, scoring, verdicts, and persistence.
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from functools import lru_cache
from typing import Any

import joblib
import numpy as np

from database import save_transaction, get_velocity

MODEL_PATH = "fraud_model.pkl"

LEGIT_HANDLES = [
    "okicici", "okhdfcbank", "okaxis", "oksbi", "ybl", "paytm",
    "apl", "upi", "ibl", "axl", "cnrb", "okbizaxis", "waicici",
    "wahdfcbank", "pthdfc", "ptyes", "icicipay", "hdfcbank",
    "sbi", "axisbank", "kotak", "rbl", "indus", "aubank", "fbl",
]

SUSPICIOUS_KEYWORDS = [
    "refund", "helpdesk", "support", "kyc", "verify",
    "reward", "prize", "lucky", "agent", "cashback",
    "offer", "winner", "claim", "bonus", "free",
]


@lru_cache(maxsize=1)
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            "fraud_model.pkl not found!\n"
            "Please run:  python train_model.py  first."
        )
    return joblib.load(MODEL_PATH)


def extract_upi_features(upi_id: str):
    upi_id = upi_id.strip().lower()

    if "@" not in upi_id or upi_id.count("@") != 1:
        return 1.0, 1

    username, handle = upi_id.split("@")
    handle_risk = 0 if handle in LEGIT_HANDLES else 1

    risk = 0.0
    if len(username) < 3:
        risk += 0.3
    if len(username) > 25:
        risk += 0.15
    if username.isdigit() and len(username) >= 10:
        risk += 0.40

    digit_ratio = sum(c.isdigit() for c in username) / max(len(username), 1)
    risk += digit_ratio * 0.25

    if any(kw in username for kw in SUSPICIOUS_KEYWORDS):
        risk += 0.45
    if re.search(r"[^a-z0-9._\-]", username):
        risk += 0.30

    vowels = sum(c in "aeiou" for c in username)
    if len(username) > 8 and vowels / max(len(username), 1) < 0.1:
        risk += 0.20

    return min(risk, 1.0), handle_risk


def get_verdict(score: float) -> str:
    if score < 30:
        return "SAFE"
    if score < 65:
        return "SUSPICIOUS"
    return "FRAUD"


def analyze_transaction(
    upi_id: str,
    amount: float,
    payee_name: str = "",
    model: Any | None = None,
    persist: bool = True,
    now: datetime | None = None,
):
    if not upi_id or "@" not in upi_id:
        raise ValueError("Invalid UPI ID format")
    if amount <= 0:
        raise ValueError("Amount must be greater than 0")

    now = now or datetime.utcnow()
    hour = now.hour
    velocity = get_velocity(upi_id, minutes=60)

    username_risk, handle_risk = extract_upi_features(upi_id)
    model = model or load_model()

    features = np.array([[username_risk, handle_risk, amount, hour, velocity]])
    prob = float(model.predict_proba(features)[0][1])
    risk_score = round(prob * 100, 1)
    verdict = get_verdict(risk_score)

    explanation = {
        "upi_id_risk": round((username_risk + handle_risk) / 2 * 100, 1),
        "amount_flag": amount > 50000,
        "odd_hour": hour < 6 or hour > 22,
        "high_velocity": velocity > 5,
        "handle_known": handle_risk == 0,
        "velocity_count": velocity,
        "payee_name": payee_name.strip(),
    }

    tips: list[str] = []
    if handle_risk == 1:
        tips.append("Unknown bank handle - not registered with NPCI.")
    if username_risk > 0.4:
        tips.append("Suspicious UPI ID pattern detected.")
    if amount > 50000:
        tips.append("Very large amount - verify the receiver independently.")
    if hour < 6:
        tips.append("Transaction at odd hours (night) - common fraud pattern.")
    if velocity > 5:
        tips.append(f"This UPI ID made {velocity} transactions in the last hour.")
    if any(kw in upi_id.lower() for kw in SUSPICIOUS_KEYWORDS):
        tips.append("UPI ID contains fraud keywords (refund/support/kyc/etc).")
    if not tips:
        tips.append("No obvious red flags detected.")

    if persist:
        save_transaction(
            upi_id=upi_id,
            amount=amount,
            timestamp=now.isoformat(),
            risk_score=risk_score,
            verdict=verdict,
            features={
                "username_risk": username_risk,
                "handle_risk": handle_risk,
                "hour": hour,
                "velocity": velocity,
            },
        )

    return {
        "risk_score": risk_score,
        "verdict": verdict,
        "explanation": explanation,
        "tips": tips,
        "features": {
            "username_risk": username_risk,
            "handle_risk": handle_risk,
            "hour": hour,
            "velocity": velocity,
        },
    }
