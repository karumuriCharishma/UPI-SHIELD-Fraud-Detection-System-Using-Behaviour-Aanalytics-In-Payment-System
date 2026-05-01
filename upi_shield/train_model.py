"""
UPI Shield - Model Trainer
Loads upi_transactions.csv and trains a RandomForest classifier.

Run order:
  1. python generate_dataset.py   → creates upi_transactions.csv
  2. python train_model.py        → trains and saves fraud_model.pkl
  3. uvicorn main:app --reload    → starts the API server
"""

import sys
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, accuracy_score,
    confusion_matrix, roc_auc_score
)

# ── Config ────────────────────────────────────────────────────────
CSV_PATH   = 'upi_transactions.csv'
MODEL_PATH = 'fraud_model.pkl'
FEATURES   = ['username_risk', 'handle_risk', 'amount', 'hour', 'velocity']
TARGET     = 'label'


# ── Load CSV ──────────────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"\n  X CSV not found: {path}")
        print("    Run  python generate_dataset.py  first.\n")
        sys.exit(1)

    df = pd.read_csv(path)

    required = FEATURES + [TARGET]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        print(f"\n  X CSV is missing columns: {missing}")
        print(f"    Expected columns: {required}\n")
        sys.exit(1)

    return df


# ── Train ─────────────────────────────────────────────────────────
def train():
    print("=" * 55)
    print("  UPI Shield - Model Trainer")
    print("=" * 55)

    # 1. Load
    print(f"\n[1/5] Loading dataset from '{CSV_PATH}'...")
    df = load_data(CSV_PATH)
    n_total = len(df)
    n_fraud = int(df[TARGET].sum())
    n_legit = n_total - n_fraud
    print(f"      Rows loaded : {n_total}")
    print(f"      Legit       : {n_legit}")
    print(f"      Fraud       : {n_fraud}")
    print(f"      Class ratio : {n_legit/n_fraud:.1f}:1  (legit:fraud)")

    if 'fraud_type' in df.columns:
        print("\n      Fraud breakdown:")
        for ftype, count in df[df[TARGET] == 1]['fraud_type'].value_counts().items():
            print(f"        {ftype:<25} {count}")

    # 2. Features + Target
    print(f"\n[2/5] Extracting features: {FEATURES}")
    X = df[FEATURES]
    y = df[TARGET]

    # 3. Split
    print("\n[3/5] Train/Test split (80% / 20%, stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"      Train size : {len(X_train)}")
    print(f"      Test size  : {len(X_test)}")

    # 4. Train
    print("\n[4/5] Training RandomForest (200 trees, balanced weights)...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # 5. Evaluate
    print("\n[5/5] Evaluation on test set:")
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc     = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    cm      = confusion_matrix(y_test, y_pred)

    print(f"\n{'─'*55}")
    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  ROC-AUC   : {roc_auc:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"                Predicted Legit  Predicted Fraud")
    print(f"  Actual Legit     {cm[0][0]:<6}           {cm[0][1]:<6}")
    print(f"  Actual Fraud     {cm[1][0]:<6}           {cm[1][1]:<6}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Legit', 'Fraud']))

    # Feature importance
    importances = sorted(
        zip(FEATURES, model.feature_importances_),
        key=lambda x: x[1], reverse=True
    )
    print("  Feature Importances:")
    for feat, imp in importances:
        bar = 'I' * int(imp * 40)
        print(f"    {feat:<15} {imp:.4f}  {bar}")

    # Save model
    joblib.dump(model, MODEL_PATH)
    print(f"\n{'─'*55}")
    print(f"  Model saved to '{MODEL_PATH}'")
    print("=" * 55)
    print("  Next step: uvicorn main:app --reload")
    print("=" * 55)


if __name__ == '__main__':
    train()
