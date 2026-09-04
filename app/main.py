"""FastAPI application entry point.

Run with: uvicorn app.main:app --reload

uvicorn is the ASGI server that actually runs this app -- FastAPI defines
routes and validation, uvicorn is what accepts the TCP connections and
speaks HTTP/ASGI to it. --reload watches source files and restarts the
worker on change, which is what makes local development fast; it's a dev
convenience only -- a real deployment runs uvicorn (often behind gunicorn
as a process manager) without it, since restarting mid-request in
production would drop live requests.

FastAPI auto-generates interactive docs from the routes and Pydantic
schemas below, with zero extra code: /docs (Swagger UI) and /redoc. Once
there's more than a health check wired up, /docs is the fastest way to
exercise this API by hand -- no Postman or curl needed, it renders a form
that sends real requests to your running app.

Error handling: every endpoint below wraps its work in try/except and
raises HTTPException with one of three codes -- 400 (the caller's input
was semantically bad, e.g. a features dict missing a required column),
404 (an unknown customer_id), or 500 (something unexpected broke in
feature engineering or the model). A generic 500 with a raw Python
traceback is bad UX for an internal API a CRM system depends on -- it
gives the caller nothing to act on, and it can leak implementation
details (or, worse, a fragment of a pandas error string that happens to
include a customer's actual billing value) to whatever's on the other
end of the request. The rule followed throughout: log the full exception
internally (logger.exception, with exc_info), return only a fixed, generic
message externally -- never str(exc) in an HTTPException.detail for an
unexpected error.

Note FastAPI's own automatic Pydantic validation (a wrong type, a
too-long customer_ids list) still returns its native 422, not one of the
three codes above -- that's the framework's own "wrong types = automatic
error" behavior from Day 5, separate from the 400/404/500 this app raises
itself for errors Pydantic's schema check can't catch (e.g. a
`features` dict that's shaped correctly but missing a specific key).
"""
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request

from app.core.logger import configure_logging, get_logger
from app.schemas.score import BatchScoreRequest, ScoreRequest, ScoreResponse
from app.services.model import get_model
from app.services.scoring import score_all_customers, score_customer

configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup, not once per request.

    get_model() is already a singleton on its own (see model.py), so
    calling it again inside a request handler would just return the same
    cached object -- but loading it here, eagerly, at startup rather than
    lazily on the first request means the first real request isn't the
    one that pays for the disk read.
    """
    app.state.model_artifact = get_model()
    yield
    # No explicit teardown needed -- the model artifact is a plain
    # in-memory object with no open connection/file handle to release.


app = FastAPI(
    title="Churn Risk API",
    description="Scores customers for churn risk using a trained XGBoost model.",
    version="0.1.0",
    lifespan=lifespan,
)


def get_model_artifact(request: Request) -> dict:
    """Dependency: returns the model artifact loaded once at startup.

    FastAPI calls this on every request that declares it via Depends(),
    but it only ever reads request.app.state -- no disk I/O, no reload.
    """
    return request.app.state.model_artifact


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check: confirms the process is up and serving requests.

    Deliberately doesn't touch the model or any other dependency -- a load
    balancer/orchestrator polling this should learn "is the process
    alive," not "is the model loaded."
    """
    return {"status": "ok"}


@app.post("/score", response_model=ScoreResponse)
def score(payload: ScoreRequest, artifact: dict = Depends(get_model_artifact)) -> ScoreResponse:
    """Score a single customer.

    Runs payload.features through the exact same build_features() -> model
    -> assign_risk_tier chain used by batch scoring (Day 4), via
    score_customer() in app/services/scoring.py -- this endpoint only
    handles HTTP concerns (schemas, error mapping, logging), not the ML
    logic itself.
    """
    start = time.perf_counter()
    try:
        result = score_customer(payload.customer_id, payload.features, artifact)
    except KeyError as exc:
        # The caller's features dict is missing a column build_features()
        # needs -- a client input problem (400), not a server bug. The
        # exception message here is just a column *name* (e.g.
        # "TotalCharges"), never a customer's actual data value, so it's
        # safe to include in both the log and the response.
        logger.warning(
            "score_request rejected: missing feature",
            extra={"event": "score_request", "customer_id": payload.customer_id, "outcome": "missing_feature"},
        )
        raise HTTPException(status_code=400, detail=f"Missing required feature: {exc}") from exc
    except Exception as exc:
        # Anything else is unexpected -- a real bug in feature engineering
        # or the model, not something the caller did wrong. Log the full
        # trace internally for on-call debugging; never echo str(exc) back
        # to the client -- an unanticipated pandas/numpy error can end up
        # including a fragment of the actual input data in its message.
        logger.exception(
            "score_request failed unexpectedly",
            extra={"event": "score_request", "customer_id": payload.customer_id, "outcome": "error"},
        )
        raise HTTPException(status_code=500, detail="Internal error while scoring customer.") from exc

    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "score_request",
        extra={
            "event": "score_request",
            "customer_id": result["customer_id"],
            "risk_tier": result["risk_tier"],
            "churn_probability": round(result["churn_probability"], 4),
            "latency_ms": round(elapsed_ms, 2),
            "outcome": "ok",
        },
    )
    return ScoreResponse(**result)


@app.post("/batch-score", response_model=list[ScoreResponse])
def batch_score(payload: BatchScoreRequest) -> list[ScoreResponse]:
    """Score a batch of customers by ID -- the endpoint a daily cron job or
    CRM integration calls, rather than looping /score one customer at a
    time.

    Deliberately a plain `def`, not `async def`: build_features()/XGBoost
    prediction are synchronous, CPU-bound pandas/numpy work with no I/O to
    await. FastAPI runs a plain `def` endpoint in a worker thread
    automatically, so a slow batch run doesn't block the event loop from
    handling other requests (like /health) concurrently. Marking this
    `async def` without actually awaiting anything inside it would do the
    opposite -- freeze the whole server for the duration of the batch.

    Batch size is capped by BatchScoreRequest itself (Pydantic rejects a
    longer list with its own 422 before this function ever runs).
    """
    batch_size = len(payload.customer_ids)
    start = time.perf_counter()
    try:
        results = score_all_customers(customer_ids=payload.customer_ids)
    except Exception as exc:
        # A missing/corrupt data source, or an unexpected feature-pipeline
        # failure -- a model/feature error (500), not the caller's fault.
        logger.exception(
            "batch_score_request failed unexpectedly",
            extra={"event": "batch_score_request", "batch_size": batch_size, "outcome": "error"},
        )
        raise HTTPException(status_code=500, detail="Internal error while scoring batch.") from exc
    elapsed_ms = (time.perf_counter() - start) * 1000

    found_ids = set(results["customerID"])
    missing_ids = [cid for cid in payload.customer_ids if cid not in found_ids]
    if missing_ids:
        logger.warning(
            "batch_score_request rejected: unknown customer_id(s)",
            extra={
                "event": "batch_score_request",
                "batch_size": batch_size,
                "missing_count": len(missing_ids),
                "outcome": "not_found",
            },
        )
        raise HTTPException(status_code=404, detail=f"Customer(s) not found: {missing_ids}")

    # Score-distribution logging, not just individual scores: this is what
    # would make a sudden shift in average churn_probability -- model
    # drift, or a silently broken feature -- visible in the logs
    # immediately, rather than discovered days later from a business
    # symptom (e.g. a retention campaign budget spiking). Deliberately
    # logs only aggregate stats and customer_ids, never the underlying
    # feature values.
    probabilities = results["churn_probability"]
    tier_counts = results["risk_tier"].value_counts().to_dict()
    logger.info(
        "batch_score_request",
        extra={
            "event": "batch_score_request",
            "batch_size": batch_size,
            "latency_ms": round(elapsed_ms, 2),
            "customers_per_sec": round(batch_size / (elapsed_ms / 1000), 1) if elapsed_ms > 0 else None,
            "mean_probability": round(float(probabilities.mean()), 4),
            "median_probability": round(float(probabilities.median()), 4),
            "min_probability": round(float(probabilities.min()), 4),
            "max_probability": round(float(probabilities.max()), 4),
            "tier_counts": tier_counts,
            "outcome": "ok",
        },
    )

    return [
        ScoreResponse(
            customer_id=row.customerID,
            churn_probability=row.churn_probability,
            risk_tier=row.risk_tier,
        )
        for row in results.itertuples()
    ]
