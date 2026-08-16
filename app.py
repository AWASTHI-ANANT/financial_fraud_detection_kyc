import os

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from fraud_model import FraudDetectionEnsemble  # noqa: F401  (needed for joblib.load to resolve the class)

MODELS_DIR = "models"

app = FastAPI(title="Financial Fraud Detection & KYC Risk Engine")

try:
    model = joblib.load(f"{MODELS_DIR}/kyc_ensemble_model.pkl")
    explainer = joblib.load(f"{MODELS_DIR}/shap_explainer.pkl")
    features = joblib.load(f"{MODELS_DIR}/feature_names.pkl")
except FileNotFoundError as exc:
    raise RuntimeError(
        "Model artifacts not found. Run `python train_model.py` first to train "
        "and save the ensemble before starting the API."
    ) from exc


class Transaction(BaseModel):
    step: float
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float
    isFlaggedFraud: int = 0
    error_in_orig: float
    error_in_rec: float
    hour: float
    destIsMerchant: int
    day: float
    type_TRANSFER: int


def assign_tier(score):
    if score < 30:
        return "Low"
    elif score < 60:
        return "Medium"
    elif score < 85:
        return "High"
    else:
        return "Critical"


def get_top_reasons(shap_vals, top_n=3):
    pairs = sorted(zip(features, shap_vals), key=lambda x: abs(x[1]), reverse=True)
    return [f"{f} ({'up' if v > 0 else 'down'})" for f, v in pairs[:top_n]]


def recommend_decision(risk_score, tier, reasons):
    if tier == "Low":
        decision, refer = "APPROVE", False
    elif tier == "Medium":
        decision, refer = "MANUAL_REVIEW", True
    elif tier == "High":
        decision, refer = "REJECT", True
    else:  # Critical
        decision, refer = "REJECT + FILE SAR", True

    return {
        "risk_score": risk_score,
        "tier": tier,
        "decision": decision,
        "refer_to_analyst": refer,
        "top_reasons": reasons,
        "explanation": f"{tier} risk due to: {', '.join(reasons)}",
    }


@app.post("/kyc/score")
def score_customer(txn: Transaction):
    try:
        df = pd.DataFrame([txn.model_dump()])[features]

        # FraudDetectionEnsemble.predict_proba returns P(fraud) directly per
        # row (shape (n_samples,)) -- unlike sklearn's (n_samples, 2) output.
        # Indexing it a second time for the "positive class" is the bug that
        # was crashing this endpoint with "invalid index to scalar variable".
        proba = model.predict_proba(df)
        score = round(float(proba[0]) * 100, 1)

        tier = assign_tier(score)
        shap_values = explainer.shap_values(df)[0]
        reasons = get_top_reasons(shap_values)

        return recommend_decision(score, tier, reasons)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/kyc/explain/{tier}")
def explain(tier: str):
    return {
        "message": f"Tier {tier} cases are flagged for: high balance errors, "
        "large TRANSFER amounts, odd-hour activity."
    }


@app.get("/health")
def health():
    return {"status": "ok", "artifacts_loaded": os.path.isdir(MODELS_DIR)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
