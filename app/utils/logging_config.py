"""Structured logging.

* ``logs/app.log`` receives JSON lines (one object per event) at DEBUG level,
  rotated at 10 MB (rotation is skipped safely if another process holds the file), so
  runs can be audited and parsed.
* The console receives concise messages at the configured level (default WARNING),
  so normal CLI output stays readable.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
from datetime import datetime, timezone
from pathlib import Path

_CONFIGURED_FLAG = "_fsa_configured"
_STANDARD_ATTRS = set(vars(logging.LogRecord("x", 0, "x", 0, "x", None, None))) | {"message"}


class SafeRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """Rotating handler that keeps logging when rotation is impossible.

    On Windows a log file cannot be renamed while another process (e.g. the dashboard
    server) has it open; the standard handler then fails on every record. Here the
    rotation is skipped and records keep going to the current file.
    """

    def doRollover(self) -> None:  # noqa: N802 (stdlib name)
        try:
            super().doRollover()
        except OSError:
            if self.stream is None:
                self.stream = self._open()
            self.maxBytes = 0  # stop retrying the rotation in this process


class JsonFormatter(logging.Formatter):
    """Format each record as a single JSON object, including ``extra=`` fields."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in vars(record).items():
            if key not in _STANDARD_ATTRS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=True)


def configure_logging(logs_dir: Path, console_level: str = "WARNING") -> None:
    """Configure the root logger once per process (idempotent)."""
    root = logging.getLogger()
    if getattr(root, _CONFIGURED_FLAG, False):
        for handler in root.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(
                    handler, logging.FileHandler):
                handler.setLevel(console_level)
        return

    logs_dir.mkdir(parents=True, exist_ok=True)
    root.setLevel(logging.DEBUG)

    file_handler = SafeRotatingFileHandler(
        logs_dir / "app.log", maxBytes=10_000_000, backupCount=5, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(JsonFormatter())

    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))

    root.addHandler(file_handler)
    root.addHandler(console_handler)
    # Third-party libraries are noisy at DEBUG.
    for noisy in ("yfinance", "urllib3", "peewee", "matplotlib", "PIL", "curl_cffi",
                  "alembic.runtime.migration"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    setattr(root, _CONFIGURED_FLAG, True)
