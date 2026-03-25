"""
Structured logging for the HSCodeFinder application.

Purpose: Provides a consistent logging format across all modules.
Format: [TIMESTAMP] [MODULE] [LEVEL] message
"""

import logging
import sys


def get_logger(module_name: str) -> logging.Logger:
    """Create a structured logger for the given module.

    Args:
        module_name: Name of the calling module (e.g., 'embedding', 'vector_search').

    Returns:
        Configured logger instance with structured formatting.
    """
    logger = logging.getLogger(f"hscodefinder.{module_name}")

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger
