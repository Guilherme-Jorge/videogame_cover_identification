"""Main script for fine-tuning the CLIP model."""

import logging

from src.config import config  # noqa: F401 - Import to setup logging
from src.training.train import main

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    main()
