"""Feature pipeline for the churn-risk model."""
from __future__ import annotations

import numpy as np
import pandas as pd

SERVICE_COLUMNS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


CONTRACT_RISK_MAP = {"Month-to-month": 2, "One year": 1, "Two year": 0}

ONEHOT_COLUMNS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "PaperlessBilling",
    "PaymentMethod",
]

CATEGORY_LEVELS = {
    "gender": ["Female", "Male"],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "PhoneService": ["No", "Yes"],
    "MultipleLines": ["No", "No phone service", "Yes"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "PaperlessBilling": ["No", "Yes"],
    "PaymentMethod": [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ],
}


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Fix known data-quality issues in the raw Telco churn export.

    TotalCharges loads as a string (object dtype) because ~11 brand-new
    customers (tenure == 0) have a blank-space value instead of a real
    NaN. Those rows are filled with 0 -- their true lifetime spend, not an
    imputed guess, since a customer with tenure == 0 has not been billed
    yet.
    """
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features and one-hot encode remaining categoricals.

    Assumes clean_raw() has already been applied (TotalCharges numeric,
    no missing values).
    """
    df = df.copy()

    df["contract_risk"] = df["Contract"].map(CONTRACT_RISK_MAP)

    df["num_services"] = (df[SERVICE_COLUMNS] == "Yes").sum(axis=1)

    safe_tenure = df["tenure"].replace(0, np.nan)
    df["charge_trend"] = (df["MonthlyCharges"] - (df["TotalCharges"] / safe_tenure)).fillna(0)

    df["is_electronic_check"] = (df["PaymentMethod"] == "Electronic check").astype(int)

    for col, levels in CATEGORY_LEVELS.items():
        df[col] = pd.Categorical(df[col], categories=levels)

    df = pd.get_dummies(df, columns=ONEHOT_COLUMNS, drop_first=True)

    if "Churn" in df.columns:
        df["churn_flag"] = (df["Churn"] == "Yes").astype(int)
        df = df.drop(columns=["Churn"])

    # Drop columns superseded by engineered features, plus the identifier.
    drop_cols = [c for c in ["customerID", "Contract"] + SERVICE_COLUMNS if c in df.columns]
    df = df.drop(columns=drop_cols)

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Raw Telco churn export -> model-ready DataFrame.

    Reproduces notebooks/02_feature_engineering.ipynb: the TotalCharges
    fix, contract_risk, num_services, charge_trend, is_electronic_check,
    and one-hot encoding of the remaining categoricals. Call this from
    both training and the serving API so identical logic runs in both
    places.
    """
    return engineer_features(clean_raw(df))
