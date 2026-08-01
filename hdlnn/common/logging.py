"""Structured logging factory for HD-LNN architecture."""

import logging
import sys
from typing import Any


class RunIdAdapter(logging.LoggerAdapter):
    """LoggerAdapter that injects [run_id=...] into formatted log messages."""

    def process(self, msg: Any, kwargs: Any) -> tuple[str, dict]:
        run_id = self.extra.get("run_id", "N/A")
        return f"[run_id={run_id}] {msg}", kwargs


def get_logger(name: str, run_id: str | None = None) -> logging.LoggerAdapter:
    """Get a structured logger configured with run_id formatting.

    Args:
        name: Name of the module or logger.
        run_id: Optional run_id string to include in log lines.

    Returns:
        A LoggerAdapter that formats log messages with [run_id=...].
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return RunIdAdapter(logger, extra={"run_id": run_id or "N/A"})
