import pandas as pd
import numpy as np
import joblib
import shap

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ----------------------------
# Load Dataset
# ----------------------------

import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "..",
    "data",
    "TelcoCustomerChurn.csv"
)

print("Loading dataset from:")
print(DATA_PATH)

df = pd.read_csv(DATA_PATH)

# ----------------------------
# Cleaning
# ----------------------------

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
).fillna(0)

# ----------------------------
# Target Variable
# ----------------------------

df["target"] = df["ChurnLabel"].map({
    "Yes": 1,
    "No": 0
})

# ----------------------------
# Feature Selection
# ----------------------------

drop_cols = [
    "CustomerID",

    "CustomerStatus",

    "ChurnLabel",
    "ChurnScore",
    "ChurnCategory",
    "ChurnReason",

    "target"
]

available_drop_cols = [
    c for c in drop_cols if c in df.columns
]

selected_features = [

    "Age",
    "TenureinMonths",
    "MonthlyCharge",
    "Contract",
    "InternetType",
    "PaymentMethod",
    "SatisfactionScore",
    "Number_of_Referrals"

]

X = df[selected_features]

y = df["target"]

# ----------------------------
# One Hot Encoding
# ----------------------------

X = pd.get_dummies(
    X,
    drop_first=True
)

feature_cols = X.columns.tolist()

# ----------------------------
# Train Test Split
# ----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ----------------------------
# Scaling
# ----------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------
# XGBoost Model
# ----------------------------

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    eval_metric="logloss"
)

model.fit(
    X_train_scaled,
    y_train
)
print("\nTop Features")
print("=" * 40)

for f, imp in sorted(
    zip(feature_cols, model.feature_importances_),
    key=lambda x: x[1],
    reverse=True
)[:20]:

    print(f, round(float(imp), 4))
from sklearn.metrics import roc_auc_score
probs = model.predict_proba(X_test_scaled)[:,1]

print(
    "ROC AUC:",
    round(
        roc_auc_score(y_test, probs),
        4
    )
)
# ----------------------------
# Evaluation
# ----------------------------

preds = model.predict(X_test_scaled)

print("\nMODEL METRICS")
print("=" * 40)

print(
    "Accuracy:",
    round(
        accuracy_score(y_test, preds),
        4
    )
)

print(
    "Precision:",
    round(
        precision_score(y_test, preds),
        4
    )
)

print(
    "Recall:",
    round(
        recall_score(y_test, preds),
        4
    )
)

print(
    "F1 Score:",
    round(
        f1_score(y_test, preds),
        4
    )
)

# ----------------------------
# SHAP Explainability
# ----------------------------

explainer = shap.TreeExplainer(model)

feature_importance = {}

for feature, score in zip(
    feature_cols,
    model.feature_importances_
):
    feature_importance[feature] = float(score)

# ----------------------------
# Save Artifacts
# ----------------------------

artifacts = {
    "model": model,
    "scaler": scaler,
    "feature_cols": feature_cols,
    "feature_importances": feature_importance,
    "explainer": explainer
}

joblib.dump(
    artifacts,
    "churn_model.pkl"
)

print("\nModel saved successfully.")