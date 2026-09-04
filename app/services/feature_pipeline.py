"""Feature pipeline for the churn-risk model.

Moved out of notebooks/02_feature_engineering.ipynb into a reusable module
so the exact same transformation can be called from both model training
(Day 3) and the serving API -- the single biggest source of train/serve
skew is having two copies of this logic drift apart.

build_features(df) is the one public entry point the plan asked for, but
it's already split internally into clean_raw() and engineer_features()
rather than one large function, since a function should do one thing.
build_features() is fine as the starter public interface; if either half
grows its own edge cases later, each can be tested and extended on its own
without touching the other.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# The 6 optional add-on service columns. Folded into num_services rather
# than one-hot encoded individually -- encoding them separately would
# duplicate signal already captured by that engineered feature.
SERVICE_COLUMNS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

# Ordinal encoding of Contract: shorter commitment = easier to leave right
# now, i.e. higher churn risk. Plays the role a recency signal would play
# in a usage-based dataset.
CONTRACT_RISK_MAP = {"Month-to-month": 2, "One year": 1, "Two year": 0}

# Remaining categoricals, one-hot encoded directly. Contract and the
# service columns are deliberately excluded -- see SERVICE_COLUMNS and
# CONTRACT_RISK_MAP above.
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

# Fixed category levels for each one-hot column, taken from the full raw
# dataset. Required for correct scoring, not just cosmetic: pd.get_dummies
# only emits a column for a category actually present in whatever batch
# it's given. A small or single-row scoring batch can easily contain only
# one value for a column (e.g. one customer is just "Male", full stop) --
# with drop_first=True, get_dummies then produces *zero* dummy columns for
# it, indistinguishable from "this customer is the baseline category".
# Casting each column to a Categorical with these fixed levels before
# get_dummies forces it to always emit the full, consistent column set
# regardless of batch diversity, which is what makes a single customer
# score identically whether scored alone or as part of a larger batch.
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

    # contract_risk -- recency/commitment proxy.
    df["contract_risk"] = df["Contract"].map(CONTRACT_RISK_MAP)

    # num_services -- frequency/engagement proxy.
    df["num_services"] = (df[SERVICE_COLUMNS] == "Yes").sum(axis=1)

    # charge_trend -- current monthly rate vs. historical average monthly
    # rate. tenure == 0 rows have no billing history yet, so there's no
    # trend to measure: set to 0 (neutral) instead of dividing by zero.
    safe_tenure = df["tenure"].replace(0, np.nan)
    df["charge_trend"] = (df["MonthlyCharges"] - (df["TotalCharges"] / safe_tenure)).fillna(0)

    # is_electronic_check -- EDA showed this payment method churns far
    # more than any other (45.3% vs. 15-19%), worth a dedicated flag.
    df["is_electronic_check"] = (df["PaymentMethod"] == "Electronic check").astype(int)

    # Cast to the fixed category levels first (see CATEGORY_LEVELS) so
    # get_dummies emits a complete, batch-independent column set -- not
    # just whatever categories happen to appear in this particular df.
    for col, levels in CATEGORY_LEVELS.items():
        df[col] = pd.Categorical(df[col], categories=levels)

    # One-hot encode remaining categoricals. drop_first=True avoids the
    # dummy-variable trap; not required for a tree model like XGBoost, but
    # costs nothing to keep.
    df = pd.get_dummies(df, columns=ONEHOT_COLUMNS, drop_first=True)

    # Target: only present when the input still carries the raw label
    # (training data). Scoring/inference input for an active customer has
    # no Churn column, so this step is skipped there.
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
