"""Default logging setup for CLI and script ``__main__`` entrypoints."""

from __future__ import annotations

import logging

_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def configure_cli_logging(level: int = logging.INFO) -> None:
    """Attach a stream handler to the root logger if none exist yet."""
    root = logging.getLogger()
    if root.handlers:
        return
    logging.basicConfig(level=level, format=_FORMAT, datefmt=_DATEFMT)
