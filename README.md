# Financial Fraud Detection & KYC Risk Engine

End-to-end transaction fraud scoring pipeline handling extreme class imbalance (`PR-AUC` weighted ensemble) and serving real-time risk explanations via `FastAPI` and `SHAP`.

## Architecture & Workflow

1. **Ensemble Modeling (`fraud_model.py`, `notebooks/`)**
   - Combines gradient boosted decision trees (`XGBoost`) and `Logistic Regression` models in `FraudDetectionEnsemble`.
   - Each model is weighted by its Precision-Recall Area Under Curve (`PR-AUC`) validation performance, so the stronger model has more influence on the final score — important for handling severe class imbalance where fraud is rare (<1% of transactions).

2. **Training (`train_model.py`)**
   - The original notebook was trained on a private ~6.3M row PaySim-style CSV that isn't part of this repo.
   - `train_model.py` generates a small synthetic dataset with the same schema and fraud patterns (fraud concentrated in `TRANSFER`/`CASH_OUT`, drained origin balances, rare positive class), trains the ensemble on it, builds a SHAP explainer, and saves everything to `models/`.

3. **Real-time Explainable Microservice (`app.py`)**
   - A `FastAPI` service that loads the trained artifacts from `models/` and exposes `POST /kyc/score`, accepting a JSON transaction payload and returning a `0-100` risk score, a risk tier (`Low`/`Medium`/`High`/`Critical`), a recommended decision, and the top `SHAP` factors driving the score.

## Running locally

```bash
pip install -r requirements.txt

# 1. Train the ensemble and save model artifacts to models/
python train_model.py

# 2. Start the API
python app.py
```

Interactive docs: `http://localhost:8000/docs`

Example request:

```bash
curl -X POST http://localhost:8000/kyc/score -H "Content-Type: application/json" -d '{
  "step": 100, "amount": 95000.0, "oldbalanceOrg": 95000.0, "newbalanceOrig": 0.0,
  "oldbalanceDest": 500.0, "newbalanceDest": 500.0, "isFlaggedFraud": 0,
  "error_in_orig": 0.0, "error_in_rec": 94500.0, "hour": 3, "destIsMerchant": 0,
  "day": 4, "type_TRANSFER": 1
}'
```

Other endpoints: `GET /health`, `GET /kyc/explain/{tier}`.
