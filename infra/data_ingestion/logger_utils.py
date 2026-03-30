"""
Structured logging utility for SPAR data ingestion pipeline.

Provides JSON-structured logging compatible with CloudWatch Logs Insights
and a human-readable fallback for local development.

Usage:
    from infra.data_ingestion.logger_utils import get_logger, log_event
    logger = get_logger(__name__)
    log_event(logger, "DATASET_LOADED", {"splits": 3, "rows": 40000})
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional

# Standard LogRecord attributes to skip when merging extras
_BUILTIN_ATTRS = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__)


class CloudWatchFormatter(logging.Formatter):
    """JSON formatter that outputs one JSON object per log line."""

    def format(self, record: logging.LogRecord) -> str:
        entry: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if record.exc_info and record.exc_info[0] is not None:
            entry["exception"] = self.formatException(record.exc_info)

        # Merge any extra fields the caller passed
        for key, val in record.__dict__.items():
            if key not in _BUILTIN_ATTRS and key not in (
                "message",
                "msg",
                "args",
                "exc_info",
                "exc_text",
                "stack_info",
            ):
                entry[key] = val

        return json.dumps(entry, default=str)


def get_logger(
    name: str,
    level: Optional[str] = None,
    json_format: bool = True,
) -> logging.Logger:
    """
    Return a configured logger with de-duplicated handlers.

    Args:
        name:        Logger name (typically ``__name__``).
        level:       Log level string. Falls back to ``LOG_LEVEL`` env var, then ``INFO``.
        json_format: ``True`` for CloudWatch JSON, ``False`` for human-readable.
    """
    log = logging.getLogger(name)

    resolved_level = (level or os.environ.get("LOG_LEVEL", "INFO")).upper()
    log.setLevel(getattr(logging, resolved_level, logging.INFO))

    if log.handlers:
        return log  # already configured — avoid duplicate output

    handler = logging.StreamHandler(sys.stderr)

    if json_format:
        handler.setFormatter(CloudWatchFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
            )
        )

    handler.setLevel(getattr(logging, resolved_level, logging.INFO))
    log.addHandler(handler)
    log.propagate = False
    return log


def log_event(
    logger: logging.Logger,
    event_type: str,
    details: Dict[str, Any],
    level: str = "INFO",
) -> None:
    """
    Emit a structured event log.

    These are designed to be easily queried in CloudWatch Logs Insights:
        fields @timestamp, event_type, @message
        | filter event_type = "DATASET_LOADED"
    """
    payload: Dict[str, Any] = {
        "event_type": event_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    payload.update(details)

    log_method = getattr(logger, level.lower(), None)
    if log_method is None:
        log_method = logger.info

    log_method(json.dumps(payload, default=str))
