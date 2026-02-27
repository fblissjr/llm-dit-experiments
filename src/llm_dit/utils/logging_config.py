"""
Structured logging configuration with JSON file output and rotation.

Provides:
- Console logging with human-readable format
- JSON file logging with automatic rotation

last updated: 2026-02-06
"""

import json
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

# Default log directory
DEFAULT_LOG_DIR = Path.home() / ".llm_dit" / "logs"
DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10MB per file
DEFAULT_BACKUP_COUNT = 5  # Keep 5 rotated files


class JSONFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.

    Output format:
    {
        "timestamp": "2025-12-27T16:05:27.123456",
        "level": "INFO",
        "logger": "llm_dit.pipelines.z_image",
        "message": "Pipeline loaded successfully",
        "context": {...}  # Optional extra context
    }
    """

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add any extra context attached to the record
        if hasattr(record, "context"):
            log_entry["context"] = record.context

        # Add standard fields if present
        for field in ["prompt", "width", "height", "steps", "seed", "duration_ms"]:
            if hasattr(record, field):
                if "context" not in log_entry:
                    log_entry["context"] = {}
                log_entry["context"][field] = getattr(record, field)

        return json.dumps(log_entry, default=str)


def setup_logging(
    level: int = logging.INFO,
    log_dir: Optional[Path] = None,
    enable_json_file: bool = True,
    max_bytes: int = DEFAULT_MAX_BYTES,
    backup_count: int = DEFAULT_BACKUP_COUNT,
    console_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    date_format: str = "%H:%M:%S",
) -> None:
    """
    Configure logging with console and optional JSON file output.

    This is the single entry point for all logging configuration. Called from
    cli.py::setup_logging() during server startup. No other module should
    configure logging handlers.

    Args:
        level: Logging level (default INFO)
        log_dir: Directory for JSON log files (default ~/.llm_dit/logs)
        enable_json_file: Enable JSON file logging with rotation
        max_bytes: Max size per log file before rotation (default 10MB)
        backup_count: Number of backup files to keep (default 5)
        console_format: Format string for console output
        date_format: Date format for console timestamps
    """
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Console handler with human-readable format
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(
        logging.Formatter(console_format, datefmt=date_format)
    )
    root_logger.addHandler(console_handler)

    # JSON file handler with rotation
    if enable_json_file:
        if log_dir is None:
            log_dir = DEFAULT_LOG_DIR

        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / "llm_dit.jsonl"

        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(JSONFormatter())
        root_logger.addHandler(file_handler)

        # Log startup info to file
        root_logger.info(
            f"JSON logging enabled: {log_file} "
            f"(max {max_bytes // 1024 // 1024}MB, {backup_count} backups)"
        )
