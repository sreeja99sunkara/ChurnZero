"""Train, evaluate, and persist the churn-risk model.
train_model() only fits, 
evaluate_model() only scores, 
save_model()/load_model() only handle disk I/O.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from app.core.config import HIGH_RISK_THRESHOLD, MEDIUM_RISK_THRESHOLD, MODEL_PATH, SCORE_THRESHOLD

logger = logging.getLogger(__name__)

MODEL_VERSION = "v1"


def train_model(
    X_train: pd.DataFrame, y_train: pd.Series, scale_pos_weight: float | None = None
) -> XGBClassifier:
    """Fit an XGBClassifier on already-engineered features.

    scale_pos_weight defaults to count(negative) / count(positive) on
    y_train, per the Day 3 imbalance comparison (notebooks/03_train_model.ipynb)
    that found this the best-performing, simplest approach on this dataset.
    Pass an explicit value to override, e.g. for cross-validated tuning
    later.
    """
    if scale_pos_weight is None:
        neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
        scale_pos_weight = neg / pos

    model = XGBClassifier(random_state=42, eval_metric="logloss", scale_pos_weight=scale_pos_weight)
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series, threshold: float = SCORE_THRESHOLD
) -> dict[str, float]:
    """Score a fitted model on a held-out set.

    roc_auc and pr_auc are threshold-independent (computed on predicted
    probabilities); precision/recall/f1 depend on `threshold`, which
    should be the actual deployment cutoff, not necessarily 0.5 -- see
    notebooks/03_train_model.ipynb for how the top-15%-riskiest threshold
    was chosen.
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    return {
        "threshold": threshold,
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
    }


def assign_risk_tier(
    probability: float, high_threshold: float = HIGH_RISK_THRESHOLD, medium_threshold: float = MEDIUM_RISK_THRESHOLD
) -> str:
    """Bucket a single churn probability into high/medium/low.

    Originally written and tested inline in notebooks/04_scoring_pipeline.ipynb;
    moved here once a second notebook (05_scoring_chain.ipynb) needed the
    same tier logic -- one place for the threshold rules rather than two
    copies that could quietly drift apart, same reasoning as build_features()
    living in feature_pipeline.py instead of each notebook re-deriving it.
    """
    if probability >= high_threshold:
        return "high"
    if probability >= medium_threshold:
        return "medium"
    return "low"


def align_features(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Reindex a build_features() output to the exact columns/order the
    model was trained on.

    Any training-time column missing from `df` (e.g. a one-hot category
    that didn't appear in this particular batch) is added as 0, which is
    the correct fill for a one-hot indicator that just didn't occur. Any
    extra column in `df` not seen at training time is dropped. Without
    this step, a small scoring batch would silently feed the model a
    different feature space than it was trained on.
    """
    return df.reindex(columns=feature_columns, fill_value=0)


def save_model(
    model: XGBClassifier,
    feature_columns: list[str],
    threshold: float,
    path: str | Path | None = None,
    version: str = MODEL_VERSION,
) -> Path:
    """Persist the fitted model together with the feature schema and
    deployment threshold it was trained/evaluated with -- not just the
    raw weights. See the module docstring for why.
    """
    path = Path(path) if path is not None else Path(MODEL_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": model,
        "feature_columns": feature_columns,
        "threshold": threshold,
        "version": version,
    }
    joblib.dump(artifact, path)
    return path


def load_model(path: str | Path | None = None) -> dict[str, Any]:
    """Load a persisted artifact (model + feature_columns + threshold + version)."""
    path = Path(path) if path is not None else Path(MODEL_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"No model found at {path}. Train and save_model() one first."
        )
    logger.info("Loading model artifact from %s", path)
    return joblib.load(path)


_cached_artifact: dict[str, Any] | None = None


def get_model(path: str | Path | None = None) -> dict[str, Any]:
    """Return the cached model artifact, loading it from disk on first call."""
    global _cached_artifact
    if _cached_artifact is None:
        _cached_artifact = load_model(path)
    return _cached_artifact


def reset_model_cache() -> None:
    """Clear the cached artifact. Mainly for tests, or after retraining a
    new version without restarting the process."""
    global _cached_artifact
    _cached_artifact = None
