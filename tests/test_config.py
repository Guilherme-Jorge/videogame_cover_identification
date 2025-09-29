"""Tests for configuration settings."""

import logging
import unittest.mock as mock
from dataclasses import dataclass

import pytest

from src.config import (
    AppConfig,
    IndexConfig,
    LoggingConfig,
    ModelConfig,
    PathConfig,
    SearchConfig,
    TrainingConfig,
    setup_logging,
)


class TestSearchConfig:
    """Test SearchConfig dataclass."""

    def test_default_values(self):
        """Test default values for SearchConfig."""
        config = SearchConfig()
        assert config.topk == 25
        assert config.rerank_k == 5
        assert config.accept_threshold == 0.25
        assert config.geom_score_threshold == 0.3

    def test_custom_values(self):
        """Test custom values for SearchConfig."""
        config = SearchConfig(topk=10, rerank_k=3, accept_threshold=0.5, geom_score_threshold=0.4)
        assert config.topk == 10
        assert config.rerank_k == 3
        assert config.accept_threshold == 0.5
        assert config.geom_score_threshold == 0.4


class TestIndexConfig:
    """Test IndexConfig dataclass."""

    def test_default_values(self):
        """Test default values for IndexConfig."""
        config = IndexConfig()
        assert config.jsonl_path == "metadata.jsonl"
        assert config.use_gpu is True
        assert config.index_path == "covers.faiss"
        assert config.npy_path == "covers.npy"
        assert config.meta_path == "covers_meta.json"

    def test_custom_values(self):
        """Test custom values for IndexConfig."""
        config = IndexConfig(
            jsonl_path="custom.jsonl",
            use_gpu=False,
            index_path="custom.faiss",
            npy_path="custom.npy",
            meta_path="custom.json",
        )
        assert config.jsonl_path == "custom.jsonl"
        assert config.use_gpu is False
        assert config.index_path == "custom.faiss"
        assert config.npy_path == "custom.npy"
        assert config.meta_path == "custom.json"


class TestTrainingConfig:
    """Test TrainingConfig dataclass."""

    def test_default_values(self):
        """Test default values for TrainingConfig."""
        config = TrainingConfig()
        assert config.jsonl_path == "metadata.jsonl"
        assert config.epochs == 5
        assert config.batch_size == 32
        assert config.lr == 5e-4
        assert config.workers == 8
        assert config.dim == 512
        assert config.amp == "fp16"
        assert config.out_path == "cover_encoder.pt"
        assert config.temperature == 0.07
        assert config.grad_accumulation_steps == 4

    def test_custom_values(self):
        """Test custom values for TrainingConfig."""
        config = TrainingConfig(
            jsonl_path="custom.jsonl",
            epochs=10,
            batch_size=16,
            lr=1e-4,
            workers=4,
            dim=256,
            amp="bf16",
            out_path="custom.pt",
            temperature=0.1,
            grad_accumulation_steps=2,
        )
        assert config.jsonl_path == "custom.jsonl"
        assert config.epochs == 10
        assert config.batch_size == 16
        assert config.lr == 1e-4
        assert config.workers == 4
        assert config.dim == 256
        assert config.amp == "bf16"
        assert config.out_path == "custom.pt"
        assert config.temperature == 0.1
        assert config.grad_accumulation_steps == 2


class TestModelConfig:
    """Test ModelConfig dataclass."""

    def test_default_values(self):
        """Test default values for ModelConfig."""
        config = ModelConfig()
        assert config.clip_model == "ViT-B-16"
        assert config.clip_pretrained == "laion2b_s34b_b88k"
        assert config.weights_path == "data/cover_encoder.pt"

    def test_custom_values(self):
        """Test custom values for ModelConfig."""
        config = ModelConfig(
            clip_model="ViT-L-14",
            clip_pretrained="openai",
            weights_path="custom/path.pt",
        )
        assert config.clip_model == "ViT-L-14"
        assert config.clip_pretrained == "openai"
        assert config.weights_path == "custom/path.pt"


class TestPathConfig:
    """Test PathConfig dataclass."""

    def test_default_values(self):
        """Test default values for PathConfig."""
        config = PathConfig()
        assert config.covers_root_env == "COVERS_ROOT"
        assert config.default_covers_dir == "covers"
        assert config.data_dir == "data"

    def test_custom_values(self):
        """Test custom values for PathConfig."""
        config = PathConfig(
            covers_root_env="CUSTOM_ROOT",
            default_covers_dir="custom_covers",
            data_dir="custom_data",
        )
        assert config.covers_root_env == "CUSTOM_ROOT"
        assert config.default_covers_dir == "custom_covers"
        assert config.data_dir == "custom_data"


class TestLoggingConfig:
    """Test LoggingConfig dataclass."""

    def test_default_values(self):
        """Test default values for LoggingConfig."""
        config = LoggingConfig()
        assert config.level == "INFO"
        assert config.format == "%(asctime)s %(levelname)s %(name)s: %(message)s"

    def test_custom_values(self):
        """Test custom values for LoggingConfig."""
        config = LoggingConfig(
            level="DEBUG",
            format="%(levelname)s: %(message)s",
        )
        assert config.level == "DEBUG"
        assert config.format == "%(levelname)s: %(message)s"


class TestAppConfig:
    """Test AppConfig dataclass."""

    @mock.patch("torch.cuda.is_available", return_value=True)
    def test_device_auto_cuda(self, mock_cuda):
        """Test device auto-detection when CUDA is available."""
        config = AppConfig(device=None)
        assert config.device == "cuda"
        mock_cuda.assert_called_once()

    @mock.patch("torch.cuda.is_available", return_value=False)
    def test_device_auto_cpu(self, mock_cuda):
        """Test device auto-detection when CUDA is not available."""
        config = AppConfig(device=None)
        assert config.device == "cpu"
        mock_cuda.assert_called_once()

    @mock.patch("torch.cuda.is_available", side_effect=ImportError)
    def test_device_auto_cpu_on_import_error(self, mock_cuda):
        """Test device auto-detection when torch import fails."""
        config = AppConfig(device=None)
        assert config.device == "cpu"
        mock_cuda.assert_called_once()

    def test_device_manual(self):
        """Test manual device setting."""
        config = AppConfig(device="cuda:1")
        assert config.device == "cuda:1"

    def test_default_config_instances(self):
        """Test that AppConfig creates default instances for all sub-configs."""
        config = AppConfig()
        assert isinstance(config.paths, PathConfig)
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.search, SearchConfig)
        assert isinstance(config.index, IndexConfig)
        assert isinstance(config.training, TrainingConfig)
        assert isinstance(config.logging, LoggingConfig)


class TestSetupLogging:
    """Test setup_logging function."""

    @mock.patch("logging.basicConfig")
    def test_setup_logging_with_config(self, mock_basic_config):
        """Test setup_logging with custom config."""
        config = LoggingConfig(level="DEBUG", format="%(levelname)s: %(message)s")
        setup_logging(config)
        mock_basic_config.assert_called_once_with(
            level=logging.DEBUG,
            format="%(levelname)s: %(message)s",
        )

    @mock.patch("logging.basicConfig")
    def test_setup_logging_default(self, mock_basic_config):
        """Test setup_logging with default config."""
        setup_logging(None)
        mock_basic_config.assert_called_once_with(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )


class TestGlobalConfig:
    """Test global configuration instance."""

    def test_global_config_exists(self):
        """Test that global config instance exists."""
        from src.config import config

        assert isinstance(config, AppConfig)
        assert isinstance(config.paths, PathConfig)
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.search, SearchConfig)
        assert isinstance(config.index, IndexConfig)
        assert isinstance(config.training, TrainingConfig)
        assert isinstance(config.logging, LoggingConfig)
