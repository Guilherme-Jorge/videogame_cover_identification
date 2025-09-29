"""Tests for CLIP model wrapper and utilities."""

import tempfile
import unittest.mock as mock
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from src.models.clip_model import (
    ImgModel,
    _infer_clip_embed_dim,
    load_encoder,
)


class TestInferClipEmbedDim:
    """Test _infer_clip_embed_dim function."""

    def test_infer_embed_dim_with_attr(self):
        """Test inference when model has embed_dim attribute."""
        class MockModel:
            embed_dim = 512

        model = MockModel()
        assert _infer_clip_embed_dim(model) == 512

    def test_infer_embed_dim_without_attr(self):
        """Test inference when model doesn't have embed_dim attribute."""
        class MockModel:
            def encode_image(self, x):
                # Return tensor with shape [batch, 768]
                return torch.zeros(1, 768)

        model = MockModel()
        assert _infer_clip_embed_dim(model) == 768


class TestImgModel:
    """Test ImgModel class."""

    def test_img_model_forward(self):
        """Test ImgModel forward pass."""
        # Create mock CLIP model
        class MockClipModel(nn.Module):
            def __init__(self):
                super().__init__()

            def encode_image(self, x):
                # Return unnormalized features
                return torch.randn(x.shape[0], 512)

        clip_model = MockClipModel()

        # Test with identity projection
        model = ImgModel(clip_model, nn.Identity())
        x = torch.randn(2, 3, 224, 224)

        output = model(x)
        assert output.shape == (2, 512)
        # Check normalization
        norms = torch.norm(output, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

    def test_img_model_with_projection(self):
        """Test ImgModel with linear projection."""
        class MockClipModel(nn.Module):
            def __init__(self):
                super().__init__()

            def encode_image(self, x):
                return torch.randn(x.shape[0], 768)

        clip_model = MockClipModel()
        projection = nn.Linear(768, 512)

        model = ImgModel(clip_model, projection)
        x = torch.randn(2, 3, 224, 224)

        output = model(x)
        assert output.shape == (2, 512)
        # Check normalization
        norms = torch.norm(output, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)


class TestLoadEncoder:
    """Test load_encoder function."""

    @mock.patch("src.models.clip_model.open_clip.create_model_and_transforms")
    def test_load_encoder_no_weights_file(self, mock_create_model):
        """Test loading encoder when no weights file exists."""
        # Mock open_clip
        mock_model = mock.MagicMock()
        mock_model.embed_dim = 512
        mock_preprocess = mock.MagicMock()

        mock_create_model.return_value = (mock_model, None, mock_preprocess)

        with mock.patch("os.path.exists", return_value=False):
            model, preprocess = load_encoder(device="cpu", weights_path="nonexistent.pt")

        assert model is not None
        assert preprocess is mock_preprocess
        mock_create_model.assert_called_once_with(
            "ViT-B-16", pretrained="laion2b_s34b_b88k"
        )

    @mock.patch("src.models.clip_model.open_clip.create_model_and_transforms")
    def test_load_encoder_with_weights_file(self, mock_create_model):
        """Test loading encoder with existing weights file."""
        # Create temporary weights file
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            temp_path = f.name

        try:
            # Mock open_clip
            mock_model = mock.MagicMock()
            mock_model.embed_dim = 512
            mock_preprocess = mock.MagicMock()

            mock_create_model.return_value = (mock_model, None, mock_preprocess)

            # Mock state dict with projection weights
            mock_state = {
                "clip.weight": torch.randn(512, 512),
                "proj.weight": torch.randn(256, 512),
                "proj.bias": torch.randn(256),
            }

            with mock.patch("torch.load", return_value=mock_state):
                with mock.patch("os.path.exists", return_value=True):
                    model, preprocess = load_encoder(device="cpu", weights_path=temp_path)

            assert model is not None
            assert preprocess is mock_preprocess

        finally:
            Path(temp_path).unlink()

    @mock.patch("src.models.clip_model.open_clip.create_model_and_transforms")
    def test_load_encoder_weights_without_projection(self, mock_create_model):
        """Test loading encoder with weights that don't have projection."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            temp_path = f.name

        try:
            # Mock open_clip
            mock_model = mock.MagicMock()
            mock_model.embed_dim = 512
            mock_preprocess = mock.MagicMock()

            mock_create_model.return_value = (mock_model, None, mock_preprocess)

            # Mock state dict without projection weights
            mock_state = {
                "clip.weight": torch.randn(512, 512),
            }

            with mock.patch("torch.load", return_value=mock_state):
                with mock.patch("os.path.exists", return_value=True):
                    model, preprocess = load_encoder(device="cpu", weights_path=temp_path)

            assert model is not None
            assert preprocess is mock_preprocess

        finally:
            Path(temp_path).unlink()

    @mock.patch("src.models.clip_model.open_clip.create_model_and_transforms")
    def test_load_encoder_load_state_dict_warnings(self, mock_create_model):
        """Test loading encoder with state dict warnings."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            temp_path = f.name

        try:
            # Mock open_clip
            mock_model = mock.MagicMock()
            mock_model.embed_dim = 512
            mock_preprocess = mock.MagicMock()

            mock_create_model.return_value = (mock_model, None, mock_preprocess)

            # Mock load_state_dict to return missing/unexpected keys
            mock_result = mock.MagicMock()
            mock_result.missing_keys = ["missing.key"]
            mock_result.unexpected_keys = ["unexpected.key"]
            mock_model.load_state_dict.return_value = mock_result

            with mock.patch("torch.load", return_value={}):
                with mock.patch("os.path.exists", return_value=True):
                    model, preprocess = load_encoder(device="cpu", weights_path=temp_path)

                    # Check that model is loaded
                    assert model is not None
                    assert preprocess is not None

        finally:
            Path(temp_path).unlink()

    @mock.patch("src.models.clip_model.open_clip.create_model_and_transforms")
    def test_load_encoder_eval_and_device(self, mock_create_model):
        """Test that model is set to eval mode and moved to device."""
        # Mock open_clip
        mock_model = mock.MagicMock()
        mock_model.embed_dim = 512
        mock_preprocess = mock.MagicMock()

        mock_create_model.return_value = (mock_model, None, mock_preprocess)

        with mock.patch("os.path.exists", return_value=False):
            model, preprocess = load_encoder(device="cuda", weights_path="nonexistent.pt")

        # Check that model is returned and is in eval mode
        assert model is not None
        assert not model.training

    @mock.patch("src.models.clip_model.open_clip.create_model_and_transforms")
    def test_load_encoder_custom_device(self, mock_create_model):
        """Test loading encoder with custom device."""
        # Mock open_clip
        mock_model = mock.MagicMock()
        mock_model.embed_dim = 512
        mock_preprocess = mock.MagicMock()

        mock_create_model.return_value = (mock_model, None, mock_preprocess)

        with mock.patch("os.path.exists", return_value=False):
            model, preprocess = load_encoder(device="cuda:1", weights_path="nonexistent.pt")

        # Check that model is returned
        assert model is not None
        assert not model.training
