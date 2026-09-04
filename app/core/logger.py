"""Structured logging setup for the churn-risk API.

What gets logged, per Day 6's plan: every scoring request (customer_id,
risk_tier, churn_probability, latency), batch size and score distribution
per batch run, and inference latency. The point of logging the
distribution, not just individual scores: if tomorrow's average churn
score jumps from ~0.27 to ~0.45, that's either real (something changed
for customers) or a bug (a feature broke) -- either way you want to know
from the logs immediately, not discover it days later when someone asks
why the retention campaign suddenly got three times more expensive.

What never gets logged: raw customer feature values (Contract,
MonthlyCharges, PaymentMethod, TotalCharges, etc.) alongside a churn
score. customer_id is an opaque identifier and is fine to log; the actual
billing/account attributes that produced a score are not -- a debug log
or an exception message that includes them would put PII/billing data in
plaintext log files. Every log call in this app logs customer_id + the
score/tier/latency, never the request's raw features dict.

Logs are structured as one JSON object per line, not free-form prose
sentences, so they're aggregable by key without regex -- "what was
average churn_probability across today's score_request events" is a
one-line jq/log-query over the day's logs, not a manual read-through.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone

# Every attribute a plain LogRecord has by default -- anything else on a
# record is a caller-supplied `extra={...}` field and belongs in the
# structured output, not the reserved/internal ones.
_RESERVED_RECORD_ATTRS = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()) | {
    "message",
    "asctime",
}


class JsonFormatter(logging.Formatter):
    """Renders each log record as one JSON object per line: timestamp,
    level, logger name, message, plus whatever structured fields the call
    site passed via extra=..., e.g.:

        logger.info("score_request", extra={"event": "score_request",
                                              "customer_id": cid,
                                              "risk_tier": tier,
                                              "churn_probability": prob,
                                              "latency_ms": elapsed_ms})
    """

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extras = {k: v for k, v in record.__dict__.items() if k not in _RESERVED_RECORD_ATTRS}
        payload.update(extras)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: int = logging.INFO) -> None:
    """Set up JSON-structured logging for the whole process. Call once, at
    startup (see app/main.py) -- other modules just call
    get_logger(__name__) and don't configure anything themselves.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers = [handler]  # replace uvicorn/default handlers to avoid duplicate/mixed-format lines
    root.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """Thin convenience wrapper so call sites don't need to import logging
    directly -- app/core/logger.py stays the one place logging is actually
    configured."""
    return logging.getLogger(name)
