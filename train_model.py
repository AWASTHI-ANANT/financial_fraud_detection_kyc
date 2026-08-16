"""Trains the fraud detection ensemble and saves artifacts for app.py to serve.

The original notebook trained on a private ~6.3M row PaySim-style CSV
(`/Users/Task/Fraud.csv`) that isn't part of this repo. This script generates
a small synthetic dataset with the same schema and fraud patterns (fraud only
in TRANSFER/CASH_OUT, rare positive class, balance-mismatch signal) so the
pipeline can be trained and the API can be run end-to-end without that file.

Usage:
    python train_model.py
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
import shap

from fraud_model import FraudDetectionEnsemble, FEATURE_COLUMNS

MODELS_DIR = "models"
N_ROWS = 20000
FRAUD_RATE = 0.01
RANDOM_STATE = 42


def generate_synthetic_transactions(n_rows=N_ROWS, fraud_rate=FRAUD_RATE, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)

    types = rng.choice(
        ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"],
        size=n_rows,
        p=[0.35, 0.20, 0.25, 0.10, 0.10],
    )
    is_fraud = np.zeros(n_rows, dtype=int)
    fraud_eligible = np.where(np.isin(types, ["TRANSFER", "CASH_OUT"]))[0]
    n_fraud = int(n_rows * fraud_rate)
    fraud_idx = rng.choice(fraud_eligible, size=min(n_fraud, len(fraud_eligible)), replace=False)
    is_fraud[fraud_idx] = 1

    step = rng.integers(1, 744, size=n_rows)
    amount = rng.exponential(scale=5000, size=n_rows)
    amount[is_fraud == 1] *= rng.uniform(5, 20, size=is_fraud.sum())

    old_orig = rng.exponential(scale=8000, size=n_rows)
    new_orig = np.clip(old_orig - amount, 0, None)
    new_orig[is_fraud == 1] = 0  # fraud typically drains the account

    old_dest = rng.exponential(scale=4000, size=n_rows)
    new_dest = old_dest + amount

    dest_is_merchant = (rng.random(n_rows) < 0.3).astype(int)
    dest_is_merchant[is_fraud == 1] = 0  # fraud rarely targets merchants

    is_flagged_fraud = np.zeros(n_rows, dtype=int)

    df = pd.DataFrame(
        {
            "step": step,
            "type": types,
            "amount": amount,
            "oldbalanceOrg": old_orig,
            "newbalanceOrig": new_orig,
            "oldbalanceDest": old_dest,
            "newbalanceDest": new_dest,
            "isFraud": is_fraud,
            "isFlaggedFraud": is_flagged_fraud,
            "destIsMerchant": dest_is_merchant,
        }
    )
    return df


def engineer_features(df):
    df = df.copy()
    df["error_in_orig"] = df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]
    df["error_in_rec"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
    df["hour"] = df["step"] % 24
    df["day"] = df["step"] // 24
    df["type_TRANSFER"] = (df["type"] == "TRANSFER").astype(int)
    return df


def main():
    import os

    os.makedirs(MODELS_DIR, exist_ok=True)

    print(f"Generating {N_ROWS} synthetic transactions...")
    df = generate_synthetic_transactions()
    df = engineer_features(df)

    X = df[FEATURE_COLUMNS]
    y = df["isFraud"]

    x_train, x_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train: {x_train.shape}, Val: {x_val.shape}, fraud rate: {y.mean():.3%}")

    ensemble_model = FraudDetectionEnsemble()
    ensemble_model.fit(x_train, y_train, x_val, y_val)
    print("Per-model PR-AUC:", ensemble_model.model_scores)
    print("Ensemble weights:", ensemble_model.weights)

    y_hat = ensemble_model.predict_proba(x_val)
    from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

    roc_auc = roc_auc_score(y_val, y_hat)
    precision, recall, _ = precision_recall_curve(y_val, y_hat)
    pr_auc = auc(recall, precision)
    print(f"Ensemble ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}")

    print("Building SHAP explainer on the XGBoost sub-model...")
    explainer = shap.TreeExplainer(ensemble_model.models["xgboost"])

    joblib.dump(ensemble_model, f"{MODELS_DIR}/kyc_ensemble_model.pkl")
    joblib.dump(explainer, f"{MODELS_DIR}/shap_explainer.pkl")
    joblib.dump(FEATURE_COLUMNS, f"{MODELS_DIR}/feature_names.pkl")
    print(f"Saved artifacts to {MODELS_DIR}/")


if __name__ == "__main__":
    main()
