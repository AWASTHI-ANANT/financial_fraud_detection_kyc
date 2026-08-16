"""Shared model definition for training and serving.

Kept in its own module (rather than inline in a notebook or app.py) so that
joblib can pickle/unpickle FraudDetectionEnsemble instances consistently
between train_model.py and app.py.
"""
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, auc

FEATURE_COLUMNS = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "isFlaggedFraud",
    "error_in_orig",
    "error_in_rec",
    "hour",
    "destIsMerchant",
    "day",
    "type_TRANSFER",
]


class FraudDetectionEnsemble:
    def __init__(self):
        self.models = {
            "xgboost": xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric="logloss",
            ),
            "logistic_regression": LogisticRegression(
                random_state=42,
                max_iter=1000,
            ),
        }
        self.weights = None
        self.model_scores = None
        self.is_fitted = False

    def fit(self, X_train, y_train, X_val, y_val):
        self.model_scores = {}

        for name, model in self.models.items():
            model.fit(X_train, y_train)

            y_pred_proba = model.predict_proba(X_val)[:, 1]
            precision, recall, _ = precision_recall_curve(y_val, y_pred_proba)
            pr_auc = auc(recall, precision)

            self.model_scores[name] = pr_auc

        total_score = sum(self.model_scores.values())
        self.weights = {name: score / total_score for name, score in self.model_scores.items()}

        self.is_fitted = True
        return self

    def predict_proba(self, X):
        """Returns P(fraud) per row as a 1-D array — NOT sklearn's (n, 2) shape."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet!")

        ensemble_prob = np.zeros(X.shape[0])
        for name, model in self.models.items():
            pred_prob = model.predict_proba(X)[:, 1]
            ensemble_prob += self.weights[name] * pred_prob

        return ensemble_prob
