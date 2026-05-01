"""
UPI Shield - Dataset Generator
Run this to generate upi_transactions.csv
Usage: python generate_dataset.py
"""

import numpy as np
import pandas as pd
import re
import os

np.random.seed(42)

# ── Constants ─────────────────────────────────────────────────────
LEGIT_HANDLES = [
    'okicici', 'okhdfcbank', 'okaxis', 'oksbi', 'ybl', 'paytm',
    'apl', 'upi', 'ibl', 'axl', 'cnrb', 'okbizaxis', 'waicici',
    'wahdfcbank', 'pthdfc', 'ptyes', 'icicipay', 'hdfcbank',
    'sbi', 'axisbank', 'kotak', 'rbl', 'indus', 'aubank', 'fbl'
]

SUSPICIOUS_KEYWORDS = [
    'refund', 'helpdesk', 'support', 'kyc', 'verify',
    'reward', 'prize', 'lucky', 'agent', 'cashback',
    'offer', 'winner', 'claim', 'bonus', 'free'
]

LEGIT_NAMES = [
    'ram', 'priya', 'suresh', 'meena', 'ravi', 'anita', 'kumar',
    'sita', 'vikram', 'divya', 'arun', 'pooja', 'kiran', 'neha',
    'raj', 'sunita', 'mohan', 'lakshmi', 'ganesh', 'kavya',
    'rahul', 'swati', 'ajay', 'nisha', 'deepak', 'rekha',
    'nikhil', 'savita', 'amit', 'jyoti', 'rohit', 'geeta'
]


# ── Feature Extraction ────────────────────────────────────────────
def extract_upi_features(upi_id: str):
    upi_id = upi_id.strip().lower()
    if '@' not in upi_id or upi_id.count('@') != 1:
        return 1.0, 1
    username, handle = upi_id.split('@')
    handle_risk = 0 if handle in LEGIT_HANDLES else 1

    risk = 0.0
    if len(username) < 3:
        risk += 0.30
    if len(username) > 25:
        risk += 0.15
    if username.isdigit() and len(username) >= 10:
        risk += 0.40
    digit_ratio = sum(c.isdigit() for c in username) / max(len(username), 1)
    risk += digit_ratio * 0.25
    if any(kw in username for kw in SUSPICIOUS_KEYWORDS):
        risk += 0.45
    if re.search(r'[^a-z0-9._\-]', username):
        risk += 0.30
    vowels = sum(c in 'aeiou' for c in username)
    if len(username) > 8 and vowels / max(len(username), 1) < 0.1:
        risk += 0.20

    return min(risk, 1.0), handle_risk


# ── Legit Transaction Generator ───────────────────────────────────
def generate_legit(n=1000):
    records = []
    for _ in range(n):
        name    = np.random.choice(LEGIT_NAMES)
        suffix  = str(np.random.randint(1, 9999))
        handle  = np.random.choice(LEGIT_HANDLES[:15])
        upi_id  = f"{name}{suffix}@{handle}"

        u_risk, h_risk = extract_upi_features(upi_id)
        amount   = float(np.clip(np.random.lognormal(6.5, 1.2), 10, 25000))
        hour     = int(np.random.choice(range(7, 23)))
        velocity = int(np.random.randint(0, 4))

        records.append({
            'upi_id':        upi_id,
            'username':      upi_id.split('@')[0],
            'handle':        handle,
            'amount':        round(amount, 2),
            'hour':          hour,
            'velocity':      velocity,
            'username_risk': round(u_risk, 4),
            'handle_risk':   h_risk,
            'fraud_type':    'none',
            'label':         0
        })
    return records


# ── Fraud Transaction Generators ──────────────────────────────────
def _fake_handle_fraud():
    chars    = list('abcdefghijklmnopqrstuvwxyz')
    handle   = ''.join(np.random.choice(chars, np.random.randint(4, 10)))
    username = ''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz0123456789'), np.random.randint(8, 16)))
    upi_id   = f"{username}@{handle}"
    amount   = float(np.random.uniform(500, 60000))
    hour     = int(np.random.randint(0, 24))
    velocity = int(np.random.randint(0, 18))
    return upi_id, handle, amount, hour, velocity, 'fake_handle'


def _large_amount_night_fraud():
    handle   = np.random.choice(LEGIT_HANDLES[:5])
    username = 'agent' + str(np.random.randint(1000, 9999))
    upi_id   = f"{username}@{handle}"
    amount   = float(np.random.uniform(80000, 500000))
    hour     = int(np.random.randint(0, 5))
    velocity = int(np.random.randint(1, 10))
    return upi_id, handle, amount, hour, velocity, 'large_amount_night'


def _high_velocity_fraud():
    handle   = np.random.choice(LEGIT_HANDLES[:8])
    username = 'pay' + str(np.random.randint(100, 9999))
    upi_id   = f"{username}@{handle}"
    amount   = float(np.random.uniform(100, 5000))
    hour     = int(np.random.randint(0, 24))
    velocity = int(np.random.randint(20, 70))
    return upi_id, handle, amount, hour, velocity, 'high_velocity'


def _suspicious_keyword_fraud():
    kw       = np.random.choice(SUSPICIOUS_KEYWORDS)
    handle   = np.random.choice(LEGIT_HANDLES[:6])
    username = f"{kw}{np.random.randint(1, 999)}"
    upi_id   = f"{username}@{handle}"
    amount   = float(np.random.uniform(1000, 100000))
    hour     = int(np.random.randint(8, 22))
    velocity = int(np.random.randint(0, 25))
    return upi_id, handle, amount, hour, velocity, 'suspicious_keyword'


def _phone_as_upi_fraud():
    phone    = str(np.random.randint(7000000000, 9999999999))
    handle   = np.random.choice(LEGIT_HANDLES[:4])
    upi_id   = f"{phone}@{handle}"
    amount   = float(np.random.uniform(5000, 200000))
    hour     = int(np.random.randint(0, 24))
    velocity = int(np.random.randint(5, 35))
    return upi_id, handle, amount, hour, velocity, 'phone_as_upi'


def _gibberish_username_fraud():
    chars    = list('bcdfghjklmnpqrstvwxyz0123456789')
    username = ''.join(np.random.choice(chars, np.random.randint(10, 18)))
    handle   = np.random.choice(LEGIT_HANDLES[:10])
    upi_id   = f"{username}@{handle}"
    amount   = float(np.random.uniform(200, 80000))
    hour     = int(np.random.randint(0, 24))
    velocity = int(np.random.randint(0, 20))
    return upi_id, handle, amount, hour, velocity, 'gibberish_username'


FRAUD_GENERATORS = [
    _fake_handle_fraud,
    _large_amount_night_fraud,
    _high_velocity_fraud,
    _suspicious_keyword_fraud,
    _phone_as_upi_fraud,
    _gibberish_username_fraud,
]


def generate_fraud(n=600):
    records = []
    for i in range(n):
        gen = FRAUD_GENERATORS[i % len(FRAUD_GENERATORS)]
        upi_id, handle, amount, hour, velocity, ftype = gen()
        u_risk, h_risk = extract_upi_features(upi_id)
        records.append({
            'upi_id':        upi_id,
            'username':      upi_id.split('@')[0],
            'handle':        handle,
            'amount':        round(amount, 2),
            'hour':          hour,
            'velocity':      velocity,
            'username_risk': round(u_risk, 4),
            'handle_risk':   h_risk,
            'fraud_type':    ftype,
            'label':         1
        })
    return records


# ── Main ──────────────────────────────────────────────────────────
def main():
    OUTPUT = 'upi_transactions.csv'

    print("=" * 55)
    print("  UPI Shield — Dataset Generator")
    print("=" * 55)

    print("\n[1/3] Generating legitimate transactions (1000)...")
    legit = generate_legit(1000)

    print("[2/3] Generating fraud transactions (600)...")
    fraud = generate_fraud(600)

    print("[3/3] Combining, shuffling, saving CSV...")
    all_records = legit + fraud
    df = pd.DataFrame(all_records)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.index.name = 'row_id'

    df.to_csv(OUTPUT)

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print(f"  File saved : {OUTPUT}")
    print(f"  Total rows : {len(df)}")
    print(f"  Legit      : {(df['label'] == 0).sum()}")
    print(f"  Fraud      : {(df['label'] == 1).sum()}")
    print(f"\n  Fraud breakdown by type:")
    for ftype, count in df[df['label'] == 1]['fraud_type'].value_counts().items():
        print(f"    {ftype:<25} {count}")

    print(f"\n  Feature preview (first 5 rows):")
    print(df[['upi_id', 'amount', 'hour', 'velocity', 'username_risk', 'handle_risk', 'label']].head().to_string())
    print(f"{'─'*55}")
    print("\n  Next step: python train_model.py")
    print("=" * 55)


if __name__ == '__main__':
    main()
