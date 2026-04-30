"""
preprocess.py
-------------
Handles all data cleaning, feature engineering, and pipeline creation.
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from typing import Tuple


# ── Column definitions ────────────────────────────────────────────────────────
NUMERIC_FEATURES = ["Tenure", "Age", "MonthlyCharges", "TotalCharges",
                    "NumComplaints", "SupportCalls"]

CATEGORICAL_FEATURES = [
    "Gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]

TARGET = "Churn"
DROP_COLS = ["CustomerID"]


def load_data(path: str) -> pd.DataFrame:
    """Load CSV and perform basic type coercion."""
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features that improve predictive power."""
    df = df.copy()
    # Charge per month efficiency
    df["ChargePerTenure"] = df["MonthlyCharges"] / (df["Tenure"] + 1)
    # Engagement score (number of services subscribed)
    service_cols = ["PhoneService", "MultipleLines", "OnlineSecurity",
                    "OnlineBackup", "DeviceProtection", "TechSupport",
                    "StreamingTV", "StreamingMovies"]
    df["ServiceCount"] = (df[service_cols] == "Yes").sum(axis=1)
    # Long-term customer flag
    df["IsLongTerm"] = (df["Tenure"] >= 24).astype(int)
    return df


def get_feature_lists() -> Tuple[list, list]:
    """Return (numeric_cols, categorical_cols) after feature engineering."""
    num = NUMERIC_FEATURES + ["ChargePerTenure", "ServiceCount", "IsLongTerm"]
    cat = CATEGORICAL_FEATURES.copy()
    return num, cat


def build_preprocessor() -> ColumnTransformer:
    """Sklearn ColumnTransformer: impute → scale / encode."""
    num_cols, cat_cols = get_feature_lists()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, num_cols),
        ("cat", categorical_pipeline, cat_cols),
    ], remainder="drop")
    return preprocessor


def prepare_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Drop irrelevant cols, engineer features, split X / y."""
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")
    df = engineer_features(df)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y
