"""Main script for searching game covers."""

import json as pyjson
import logging
import sys

from .config import config  # noqa: F401 - Import to setup logging
from .search.search import search_cover

logger = logging.getLogger(__name__)


def main():
    """CLI entry point for cover search."""
    if len(sys.argv) != 2:
        print("Usage: python -m src.search_cover <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    result = search_cover(image_path)
    logger.info("%s", pyjson.dumps(result, indent=2))


if __name__ == "__main__":
    main()
