"""Data loading utilities."""

import json
import logging

logger = logging.getLogger(__name__)


def load_jsonl(path: str):
    """Stream JSON lines from a file.

    Args:
        path: Path to a .jsonl file.

    Yields:
        Parsed JSON objects per non-empty line.
    """
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)
