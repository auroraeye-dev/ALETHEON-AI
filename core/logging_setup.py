"""
core/logging_setup.py
======================
One simple, consistent logger for the whole project. Import `log` anywhere.
Named logging_setup (not logging) to avoid clashing with Python's stdlib logging.
"""

import logging
import sys


def _make_logger() -> logging.Logger:
    logger = logging.getLogger("aletheon")
    if logger.handlers:           # already configured — don't double-add
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s",
                            datefmt="%H:%M:%S")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    return logger


log = _make_logger()
