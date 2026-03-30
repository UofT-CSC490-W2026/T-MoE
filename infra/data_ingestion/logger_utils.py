"""Structured logging for the SPAR data ingestion pipeline.

Provides JSON-formatted output compatible with CloudWatch Logs Insights
and a human-readable fallback for local development.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional

_BUILTIN_ATTRS = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__)


class CloudWatchFormatter(logging.Formatter):
    """Emits one JSON object per log line."""

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

        # Merge caller-supplied extra fields
        for key, val in record.__dict__.items():
            if key not in _BUILTIN_ATTRS and key not in ("message", "msg", "args", "exc_info", "exc_text", "stack_info"):
                entry[key] = val

        return json.dumps(entry, default=str)


def get_logger(
    name: str,
    level: Optional[str] = None,
    json_format: bool = True,
) -> logging.Logger:
    """Return a configured logger, avoiding duplicate handlers on re-import.

    Args:
        name: Logger name (typically ``__name__``).
        level: Log level; falls back to LOG_LEVEL env var, then INFO.
        json_format: True for CloudWatch JSON, False for human-readable.
    """
    log = logging.getLogger(name)
    resolved_level = (level or os.environ.get("LOG_LEVEL", "INFO")).upper()
    log.setLevel(getattr(logging, resolved_level, logging.INFO))

    if log.handlers:
        return log

    handler = logging.StreamHandler(sys.stderr)
    if json_format:
        handler.setFormatter(CloudWatchFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        ))
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
    """Emit a structured event log queryable in CloudWatch Logs Insights.

    Example query:
        fields @timestamp, event_type | filter event_type = "DATASET_LOADED"
    """
    payload: Dict[str, Any] = {
        "event_type": event_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    payload.update(details)
    log_method = getattr(logger, level.lower(), logger.info)
    log_method(json.dumps(payload, default=str))
