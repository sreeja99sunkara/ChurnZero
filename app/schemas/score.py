"""Request/response schemas for the scoring endpoint.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.core.config import MAX_BATCH_SIZE

_EXAMPLE_FEATURES = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85,
}


class ScoreRequest(BaseModel):
    """A single customer to score."""

    customer_id: str = Field(
        description="Unique customer identifier, as used in the source customer table.",
        examples=["7590-VHVEG"],
    )
    features: dict[str, Any] = Field(
        description=(
            "Raw customer attributes -- the same columns build_features() expects "
            "(see app/services/feature_pipeline.py), e.g. tenure, Contract, "
            "MonthlyCharges, PaymentMethod. Not pre-engineered features: the API "
            "runs these through the same feature pipeline used at training time, "
            "so training and serving never drift apart."
        ),
        examples=[_EXAMPLE_FEATURES],
    )

    model_config = ConfigDict(
        json_schema_extra={"examples": [{"customer_id": "7590-VHVEG", "features": _EXAMPLE_FEATURES}]}
    )


class ScoreResponse(BaseModel):
    """A single customer's churn score and risk tier."""

    customer_id: str = Field(
        description="Echoes the customer_id from the request.",
        examples=["7590-VHVEG"],
    )
    churn_probability: float = Field(
        description="Predicted probability of churn.",
        ge=0.0,
        le=1.0,
        examples=[0.7269],
    )
    risk_tier: Literal["high", "medium", "low"] = Field(
        description=(
            "Risk bucket derived from churn_probability using HIGH_RISK_THRESHOLD "
            "and MEDIUM_RISK_THRESHOLD (see app/core/config.py)."
        ),
        examples=["high"],
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{"customer_id": "7590-VHVEG", "churn_probability": 0.7269, "risk_tier": "high"}]
        }
    )


class BatchScoreRequest(BaseModel):
    """A batch of customers to score, by ID."""

    customer_ids: list[str] = Field(
        min_length=1,
        max_length=MAX_BATCH_SIZE,
        description=(
            "Customer IDs to score, looked up from the customer data source "
            f"(CUSTOMER_DATA_PATH). Capped at {MAX_BATCH_SIZE} per request -- Pydantic "
            "rejects a longer list with a 422 automatically. For a full daily run "
            f"(e.g. ~7,000 customers), split the full ID list into pages of at most "
            f"{MAX_BATCH_SIZE} and call this endpoint once per page, rather than sending "
            "one unbounded request."
        ),
        examples=[["7590-VHVEG", "5575-GNVDE", "3668-QPYBK"]],
    )

    model_config = ConfigDict(
        json_schema_extra={"examples": [{"customer_ids": ["7590-VHVEG", "5575-GNVDE", "3668-QPYBK"]}]}
    )
