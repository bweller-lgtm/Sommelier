"""Logging configuration for the LLM Taste Cloner project."""
import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


# ANSI color codes for terminal output
class LogColors:
    """ANSI color codes for colored logging output."""
    RESET = "\033[0m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    BLUE = "\033[94m"
    GRAY = "\033[90m"


class ColoredFormatter(logging.Formatter):
    """
    Custom formatter that adds colors to log levels in terminal output.
    """

    COLORS = {
        logging.DEBUG: LogColors.GRAY,
        logging.INFO: LogColors.BLUE,
        logging.WARNING: LogColors.YELLOW,
        logging.ERROR: LogColors.RED,
        logging.CRITICAL: LogColors.RED,
    }

    def format(self, record):
        """Format log record with colors."""
        # Add color to level name
        levelname = record.levelname
        if record.levelno in self.COLORS:
            colored_levelname = f"{self.COLORS[record.levelno]}{levelname}{LogColors.RESET}"
            record.levelname = colored_levelname

        return super().format(record)


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
    enable_colors: bool = True
) -> logging.Logger:
    """
    Setup logging configuration for the project.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional path to log file. If None, logs only to console.
        enable_colors: Whether to enable colored console output.

    Returns:
        Configured logger instance.
    """
    # Get root logger
    logger = logging.getLogger("taste_cloner")
    logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)

    if enable_colors and sys.stdout.isatty():
        # Use colored formatter for terminals
        console_format = "%(levelname)s - %(name)s - %(message)s"
        console_formatter = ColoredFormatter(console_format)
    else:
        # Use plain formatter for non-terminals (e.g., file redirects)
        console_format = "%(levelname)s - %(name)s - %(message)s"
        console_formatter = logging.Formatter(console_format)

    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)

        # File logs should be more detailed and without colors
        file_format = "%(asctime)s - %(levelname)s - %(name)s - %(funcName)s:%(lineno)d - %(message)s"
        file_formatter = logging.Formatter(file_format)
        file_handler.setFormatter(file_formatter)

        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.

    Args:
        name: Module name (typically __name__).

    Returns:
        Logger instance.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started")
        >>> logger.warning("Cache miss for key: %s", key)
        >>> logger.error("Failed to load image: %s", path, exc_info=True)
    """
    # Ensure it's a child of the main logger
    if not name.startswith("taste_cloner"):
        name = f"taste_cloner.{name}"

    return logging.getLogger(name)


def configure_default_logging(verbose: bool = True) -> logging.Logger:
    """
    Configure default logging with sensible defaults for the project.

    Args:
        verbose: If True, sets log level to INFO. If False, sets to WARNING.

    Returns:
        Configured logger instance.
    """
    log_level = "INFO" if verbose else "WARNING"

    # Create logs directory
    log_dir = Path(".logs")
    log_dir.mkdir(exist_ok=True)

    # Generate timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"taste_cloner_{timestamp}.log"

    return setup_logging(
        log_level=log_level,
        log_file=log_file,
        enable_colors=True
    )


# Convenience functions for quick logging without logger instance
def log_info(message: str, *args, **kwargs):
    """Log an info message."""
    logger = get_logger("taste_cloner")
    logger.info(message, *args, **kwargs)


def log_warning(message: str, *args, **kwargs):
    """Log a warning message."""
    logger = get_logger("taste_cloner")
    logger.warning(message, *args, **kwargs)


def log_error(message: str, *args, **kwargs):
    """Log an error message."""
    logger = get_logger("taste_cloner")
    logger.error(message, *args, **kwargs)


def log_debug(message: str, *args, **kwargs):
    """Log a debug message."""
    logger = get_logger("taste_cloner")
    logger.debug(message, *args, **kwargs)
