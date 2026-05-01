import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle, random, re

random.seed(42)
np.random.seed(42)

FRAUD_UPI_PATTERNS = [
    "refund", "cashback", "prize", "lottery", "verify", "kyc", "update",
    "win", "reward", "offer", "free", "claim", "urgent", "support", "help"
]

LEGIT_MERCHANTS = [
    "amazon", "flipkart", "swiggy", "zomato", "ola", "uber", "paytm",
    "phonepe", "gpay", "bpcl", "irctc", "hdfc", "icici", "sbi", "axis"
]

def upi_id_risk_score(upi_id: str) -> float:
    upi_id = upi_id.lower()
    score = 0.0
    for pattern in FRAUD_UPI_PATTERNS:
        if pattern in upi_id:
            score += 0.3
    for merchant in LEGIT_MERCHANTS:
        if merchant in upi_id:
            score -= 0.2
    # random numbers in upi id
    num_count = len(re.findall(r'\d', upi_id))
    if num_count > 8:
        score += 0.2
    # very short or very long
    local = upi_id.split("@")[0] if "@" in upi_id else upi_id
    if len(local) < 4:
        score += 0.15
    if len(local) > 20:
        score += 0.1
    return min(max(score, 0.0), 1.0)

def generate_dataset(n=5000):
    records = []
    for _ in range(n):
        is_fraud = random.random() < 0.3

        if is_fraud:
            pattern = random.choice(FRAUD_UPI_PATTERNS)
            suffix = random.choice(["@ybl","@ibl","@upi","@paytm","@apl"])
            nums = ''.join([str(random.randint(0,9)) for _ in range(random.randint(4,10))])
            upi_id = f"{pattern}{nums}{suffix}"
            amount = random.choice([
                random.uniform(1,10),
                random.uniform(9000, 50000),
                round(random.uniform(100,500), 2)
            ])
            hour = random.choice(list(range(0,6)) + list(range(22,24)))
            tx_count_1h = random.randint(5, 30)
            tx_count_24h = random.randint(20, 100)
        else:
            merchant = random.choice(LEGIT_MERCHANTS)
            suffix = random.choice(["@okaxis","@okhdfcbank","@oksbi","@okicici","@ybl"])
            upi_id = f"{merchant}@{suffix.strip('@')}"
            amount = random.uniform(50, 5000)
            hour = random.randint(8, 21)
            tx_count_1h = random.randint(0, 3)
            tx_count_24h = random.randint(1, 15)

        upi_risk = upi_id_risk_score(upi_id)
        amount_risk = 1.0 if amount < 5 or amount > 20000 else (0.5 if amount > 10000 else 0.0)
        hour_risk = 1.0 if hour in range(0,5) else (0.3 if hour in [5,22,23] else 0.0)
        velocity_risk = min(tx_count_1h / 10.0, 1.0)

        records.append({
            "upi_risk": upi_risk,
            "amount": amount,
            "amount_risk": amount_risk,
            "hour": hour,
            "hour_risk": hour_risk,
            "tx_count_1h": tx_count_1h,
            "tx_count_24h": tx_count_24h,
            "velocity_risk": velocity_risk,
            "is_fraud": int(is_fraud)
        })

    return pd.DataFrame(records)

df = generate_dataset(5000)
features = ["upi_risk", "amount", "amount_risk", "hour", "hour_risk",
            "tx_count_1h", "tx_count_24h", "velocity_risk"]
X = df[features]
y = df["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
model.fit(X_train, y_train)

acc = model.score(X_test, y_test)
print(f"Model Accuracy: {acc:.4f}")

with open("/home/claude/upi_shield/model/fraud_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved.")
