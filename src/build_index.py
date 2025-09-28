"""Main script for building the cover embedding index."""

import logging

from src.config import config  # noqa: F401 - Import to setup logging
from src.indexing.build import build_index

logger = logging.getLogger(__name__)


def main():
    """CLI entry point for building the index."""
    build_index()


if __name__ == "__main__":
    main()
