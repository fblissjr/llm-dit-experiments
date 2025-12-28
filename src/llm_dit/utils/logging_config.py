"""
Structured logging configuration with JSON file output and rotation.

Provides:
- Console logging with human-readable format
- JSON file logging with automatic rotation
- Context injection for generation metadata

last updated: 2025-12-27
"""

import json
import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Optional

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


class ContextLogger(logging.LoggerAdapter):
    """
    Logger adapter that injects context into log records.

    Usage:
        logger = get_context_logger(__name__)
        logger.info("Generating image", extra={"prompt": "A cat", "steps": 9})
    """

    def process(self, msg: str, kwargs: dict) -> tuple:
        # Move context from extra to record attribute
        extra = kwargs.get("extra", {})
        if extra:
            kwargs["extra"] = {"context": extra}
        return msg, kwargs


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

    Args:
        level: Logging level (default INFO)
        log_dir: Directory for JSON log files (default ~/.llm_dit/logs)
        enable_json_file: Enable JSON file logging with rotation
        max_bytes: Max size per log file before rotation (default 10MB)
        backup_count: Number of backup files to keep (default 5)
        console_format: Format string for console output
        date_format: Date format for console timestamps

    Example:
        from llm_dit.utils.logging_config import setup_logging

        # Basic setup (console only)
        setup_logging(level=logging.DEBUG)

        # With JSON file logging
        setup_logging(
            level=logging.INFO,
            enable_json_file=True,
            log_dir=Path("/var/log/llm_dit"),
        )
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


def get_context_logger(name: str) -> ContextLogger:
    """
    Get a logger with context injection support.

    Args:
        name: Logger name (typically __name__)

    Returns:
        ContextLogger instance

    Usage:
        logger = get_context_logger(__name__)

        # Log with context
        logger.info("Generation complete", extra={
            "prompt": "A cat",
            "duration_ms": 1234,
            "seed": 42,
        })
    """
    return ContextLogger(logging.getLogger(name), {})


def log_generation(
    logger: logging.Logger,
    prompt: str,
    width: int,
    height: int,
    steps: int,
    duration_ms: float,
    seed: Optional[int] = None,
    **kwargs: Any,
) -> None:
    """
    Log a generation event with structured metadata.

    Args:
        logger: Logger instance
        prompt: Generation prompt
        width: Image width
        height: Image height
        steps: Number of inference steps
        duration_ms: Generation duration in milliseconds
        seed: Random seed (optional)
        **kwargs: Additional context fields

    Example:
        log_generation(
            logger,
            prompt="A cat sleeping",
            width=1024,
            height=1024,
            steps=9,
            duration_ms=1234.5,
            seed=42,
            model="z-image-turbo",
        )
    """
    context = {
        "prompt": prompt[:200],  # Truncate long prompts
        "width": width,
        "height": height,
        "steps": steps,
        "duration_ms": round(duration_ms, 2),
    }
    if seed is not None:
        context["seed"] = seed
    context.update(kwargs)

    # Create a log record with context
    record = logger.makeRecord(
        logger.name,
        logging.INFO,
        "(generation)",
        0,
        "Generation complete",
        (),
        None,
    )
    record.context = context
    logger.handle(record)
