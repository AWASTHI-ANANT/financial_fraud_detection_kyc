# Financial Fraud Detection & KYC Risk Engine

End-to-end transaction fraud scoring pipeline handling extreme class imbalance (`PR-AUC` weighted ensemble) and serving real-time risk explanations via `FastAPI` and `SHAP`.

## Architecture & Workflow

1. **Ensemble Modeling (`notebooks/`)**
   - Combines gradient boosted decision trees (`XGBoost`) and `Logistic Regression` models.
   - Specifically weighted by Precision-Recall Area Under Curve (`PR-AUC`) validation performance ($W_{\text{XGB}} = 0.72, W_{\text{LR}} = 0.28$) to handle severe dataset imbalance where positive fraud instances are rare (<1%).

2. **Real-time Explainable Microservice (`app.py`)**
   - Deployed as a `FastAPI` microservice accepting JSON transaction payloads (`step`, `type_code`, `amount`, origin/destination balance differentials).
   - Calculates local `SHAP` attribution factors on the fly to return both a numeric risk score $[0, 1]$ and top contributing features for KYC/AML auditors.

## Running the API locally

```bash
pip install -r requirements.txt
python app.py
```

Check API interactive docs at `http://localhost:8080/docs`.
