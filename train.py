"""
train.py
--------
Train final best model and save it to disk
"""

import pickle
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer


# =========================
# 1. Load data
# =========================
DATA_PATH = "data/insurance_claims.csv"   # adjust path if needed
MODEL_PATH = "models/fraud_model.pkl"

df = pd.read_csv(DATA_PATH)

# Target
TARGET = "fraud_reported"

X = df.drop(columns=[TARGET])
y = df[TARGET]

# =========================
# 2. Train-test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# 3. Preprocessing
# =========================
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# =========================
# 4. Model (Best Model)
# =========================
model = LGBMClassifier(
    objective="binary",
    learning_rate=0.01,
    n_estimators=100,
    num_leaves=31,
    random_state=42
)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

# =========================
# 5. Train model
# =========================
print("Training final model...")
pipeline.fit(X_train, y_train)

# =========================
# 6. Evaluation
# =========================
y_train_pred = pipeline.predict(X_train)
y_test_pred = pipeline.predict(X_test)

y_train_proba = pipeline.predict_proba(X_train)[:, 1]
y_test_proba = pipeline.predict_proba(X_test)[:, 1]

print("\nTraining Performance")
print("Accuracy:", accuracy_score(y_train, y_train_pred))
print("AUC:", roc_auc_score(y_train, y_train_proba))

print("\nTest Performance")
print("Accuracy:", accuracy_score(y_test, y_test_pred))
print("AUC:", roc_auc_score(y_test, y_test_proba))

print("\nClassification Report (Test)")
print(classification_report(y_test, y_test_pred))

# =========================
# 7. Save model
# =========================
with open(MODEL_PATH, "wb") as f:
    pickle.dump(pipeline, f)

print(f"\nModel saved to: {MODEL_PATH}")
