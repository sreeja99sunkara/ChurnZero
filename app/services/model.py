"""Train, evaluate, and persist the churn-risk model.

Each function does one thing: train_model() only fits, evaluate_model()
only scores, save_model()/load_model() only handle disk I/O. Nothing here
reads a CSV or calls build_features() -- that's the caller's job (see
notebooks/03_train_model.ipynb for the reference pipeline this module
formalizes).

Why the persisted artifact bundles more than raw model weights: at predict
time, a fresh batch of customers run through build_features() (in
feature_pipeline.py) won't necessarily produce the same one-hot columns
the model was trained on -- pd.get_dummies() only emits columns for
categories actually present in whatever batch it's given, and a single
customer, or a small batch, can easily be missing a category that showed
up in the full training set. Saving feature_columns alongside the model
and reindexing to it at predict time (see align_features()) is what keeps
training and serving from silently drifting apart. The classification
threshold is bundled for the same reason: it's a business decision made
during evaluation (see notebooks/03_train_model.ipynb), not something that
should be re-derived, forgotten, or hardcoded separately in the API layer.
"""
from __future__ import annotations

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

from app.core.config import MODEL_PATH

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
    model: XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series, threshold: float = 0.5
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
    return joblib.load(path)


# --- Singleton: load the model once per process, not once per request ---
#
# The serving API should call get_model() from its request handlers; the
# first call loads from disk and caches the artifact in this module's
# global, every call after that just returns the cached object. Reloading
# a joblib pickle from disk on every request would add real, pointless
# latency to every prediction.
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
