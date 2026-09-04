"""Batch-score customers with the persisted churn model.

Moved out of notebooks/05_scoring_chain.ipynb into a reusable module so an
actual scheduled job (or API endpoint) can call score_all_customers()
directly, the same reason build_features() and model.py exist as modules
rather than staying notebook-only. The chain itself is unchanged from the
notebook: load_customers -> build_features -> predict_proba ->
assign_risk_tier -> output DataFrame, still four separate, individually
testable functions rather than one large one.

Thresholds (SCORE_THRESHOLD, HIGH_RISK_THRESHOLD, MEDIUM_RISK_THRESHOLD)
and the customer data source (CUSTOMER_DATA_PATH) all come from
app.core.config, not from literals here -- that's the point of this task:
a marketing team retuning what counts as "high risk" should mean editing
.env, not this file.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from app.core.config import CUSTOMER_DATA_PATH
from app.services.feature_pipeline import build_features
from app.services.model import align_features, assign_risk_tier, get_model


def load_customers(path: str | Path, customer_ids: list[str] | None = None) -> pd.DataFrame:
    """Load customer records ready for scoring.

    Drops Churn if present -- a real production customer table would never
    have this column at all (it's a future outcome, not a customer
    attribute), so scoring input should never carry it either.
    """
    df = pd.read_csv(path)
    if "Churn" in df.columns:
        df = df.drop(columns=["Churn"])
    if customer_ids is not None:
        df = df[df["customerID"].isin(customer_ids)].reset_index(drop=True)
    return df


def predict_proba(features: pd.DataFrame, artifact: dict) -> np.ndarray:
    """Run the model on already-engineered features.

    align_features() reindexes to the model's exact training-time columns
    first -- without it, a small or low-diversity batch can silently score
    against a different feature space than the model was trained on.
    """
    aligned = align_features(features, artifact["feature_columns"])
    return artifact["model"].predict_proba(aligned)[:, 1]


def score_all_customers(
    source: str | Path | None = None, customer_ids: list[str] | None = None
) -> pd.DataFrame:
    """Score customers and return customerID, churn_probability, risk_tier, last_scored_at.

    This is the production batch-scoring entry point. `source` defaults to
    CUSTOMER_DATA_PATH (config-driven, not hardcoded) and `customer_ids`
    is None by default, meaning "score every customer in `source`" -- pass
    a specific list for testing or a partial run.

    Every row in a given call gets the *same* last_scored_at (UTC, ISO
    8601) -- one timestamp per batch run, not per row -- so a downstream
    CRM can identify which batch a score came from, dedupe on
    (customerID, last_scored_at), and audit when a given score was
    actually produced.
    """
    if source is None:
        source = CUSTOMER_DATA_PATH

    customers = load_customers(source, customer_ids=customer_ids)
    features = build_features(customers)
    artifact = get_model()
    probabilities = predict_proba(features, artifact)
    tiers = [assign_risk_tier(p) for p in probabilities]
    scored_at = datetime.now(timezone.utc).isoformat()

    return pd.DataFrame(
        {
            "customerID": customers["customerID"],
            "churn_probability": probabilities,
            "risk_tier": tiers,
            "last_scored_at": scored_at,
        }
    )
