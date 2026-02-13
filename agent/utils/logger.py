import logging
import os
import sys
from logging.handlers import RotatingFileHandler


def _parse_level(value: str) -> int:
    if not value:
        return logging.INFO
    v = str(value).strip().upper()
    return {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "WARN": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
    }.get(v, logging.INFO)

def setup_logger(name="Agent", level=logging.INFO):
    env_level = os.environ.get("AGENT_LOG_LEVEL")
    if env_level:
        level = _parse_level(env_level)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    log_file = os.environ.get("AGENT_LOG_FILE")
    if not log_file and os.environ.get("AGENT_LOG_TO_FILE") == "1":
        log_file = os.path.join(os.getcwd(), "sandbox", "agent.log")

    if log_file and not any(isinstance(h, RotatingFileHandler) and getattr(h, "baseFilename", None) == os.path.abspath(log_file) for h in logger.handlers):
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = RotatingFileHandler(log_file, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

logger = setup_logger()
