"""Structured logging setup for the churn-risk API.
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
