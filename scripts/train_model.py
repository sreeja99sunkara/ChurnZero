"""Train and persist the churn-risk model from scratch.

The model file (models/churn_xgb_v1.pkl) is gitignored -- trained
artifacts don't belong in plain git (no diffing, bloats history). Run
this once after cloning the repo (or after data/raw/churn_data.csv
changes) to produce it locally; that's what makes "clone the repo, run
the API" actually true from a fresh checkout, rather than requiring a
model file nobody can reproduce.

Reuses the exact same pipeline as the notebooks -- build_features(),
train_model() with its Day 3 scale_pos_weight default, the 0.780
top-15%-riskiest deployment threshold chosen in Day 3's evaluation --
rather than reimplementing any of it. See notebooks/03_train_model.ipynb
for the full walkthrough with plots and the reasoning behind each choice.

Usage (from the repo root, with the venv active):
    python scripts/train_model.py
"""
import sys
from pathlib import Path

# Run directly (not via `python -m`), so put the repo root on sys.path
# ourselves rather than requiring PYTHONPATH to already be set up.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sklearn.model_selection import train_test_split

from app.core.config import CUSTOMER_DATA_PATH
from app.services.feature_pipeline import build_features
from app.services.model import evaluate_model, save_model, train_model

DEPLOYMENT_THRESHOLD = 0.780  # top-15%-riskiest cutoff -- see notebooks/03_train_model.ipynb


def main() -> None:
    raw = pd.read_csv(CUSTOMER_DATA_PATH)
    df = build_features(raw)  # raw still has Churn here, so build_features() includes churn_flag

    X = df.drop(columns=["churn_flag"])
    y = df["churn_flag"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test, threshold=DEPLOYMENT_THRESHOLD)

    print(f"Evaluation on held-out test set @ threshold={DEPLOYMENT_THRESHOLD}:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    path = save_model(model, feature_columns=list(X_train.columns), threshold=DEPLOYMENT_THRESHOLD)
    print(f"\nSaved model to {path}")


if __name__ == "__main__":
    main()
