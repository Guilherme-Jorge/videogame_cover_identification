"""Tests for index building functionality."""

import json
import tempfile
import unittest.mock as mock
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.indexing.build import build_index, embed_images


class TestEmbedImages:
    """Test embed_images function."""

    def test_embed_images_success(self):
        """Test successful embedding of images."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)

            # Create test JSONL file
            jsonl_path = temp_dir / "test.jsonl"
            with open(jsonl_path, "w") as f:
                json.dump({"id": 1, "name": "Game 1", "local_filename": "game1.jpg"}, f)
                f.write("\n")
                json.dump({"id": 2, "name": "Game 2", "local_filename": "game2.jpg"}, f)
                f.write("\n")
                json.dump({"id": 3, "name": "Game 3", "local_filename": "missing.jpg"}, f)
                f.write("\n")

            # Create test images
            img1_path = temp_dir / "game1.jpg"
            img2_path = temp_dir / "game2.jpg"

            # Create dummy images
            img = Image.new("RGB", (100, 100), color="red")
            img.save(img1_path)
            img.save(img2_path)

            # Mock the model and preprocess
            mock_model = mock.MagicMock()
            mock_model.return_value = torch.randn(1, 512)

            mock_preprocess = mock.MagicMock()
            mock_preprocess.return_value = torch.randn(3, 224, 224)

            with mock.patch("src.indexing.build.load_encoder", return_value=(mock_model, mock_preprocess)):
                with mock.patch("src.indexing.build.config") as mock_config:
                    mock_config.device = "cpu"
                    with mock.patch("torch.no_grad"):
                        metas, embeddings = embed_images(str(jsonl_path), root=str(temp_dir))

            assert len(metas) == 2  # Should skip missing image
            assert embeddings.shape == (2, 512)
            assert metas[0]["id"] == 1
            assert metas[1]["id"] == 2

    def test_embed_images_no_valid_images(self):
        """Test embedding when no valid images are found."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)

            # Create test JSONL file with no valid images
            jsonl_path = temp_dir / "test.jsonl"
            with open(jsonl_path, "w") as f:
                json.dump({"id": 1, "name": "Game 1"}, f)  # No local_filename
                f.write("\n")

            mock_model = mock.MagicMock()
            mock_model.return_value = torch.randn(1, 512)
            mock_preprocess = mock.MagicMock()

            with mock.patch("src.indexing.build.load_encoder", return_value=(mock_model, mock_preprocess)):
                with mock.patch("src.indexing.build.config") as mock_config:
                    mock_config.device = "cpu"
                    metas, embeddings = embed_images(str(jsonl_path), root=str(temp_dir))

            assert len(metas) == 0
            assert embeddings.shape == (0, 512)

    def test_embed_images_with_device(self):
        """Test embedding with specific device."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)

            # Create test JSONL file
            jsonl_path = temp_dir / "test.jsonl"
            with open(jsonl_path, "w") as f:
                json.dump({"id": 1, "name": "Game 1", "local_filename": "game1.jpg"}, f)
                f.write("\n")

            # Create test image
            img_path = temp_dir / "game1.jpg"
            img = Image.new("RGB", (100, 100), color="red")
            img.save(img_path)

            mock_model = mock.MagicMock()
            mock_model.return_value = torch.randn(1, 512)

            mock_preprocess = mock.MagicMock()
            mock_preprocess.return_value = torch.randn(3, 224, 224)

            with mock.patch("src.indexing.build.load_encoder", return_value=(mock_model, mock_preprocess)):
                with mock.patch("src.indexing.build.config"):
                    with mock.patch("torch.no_grad"):
                        metas, embeddings = embed_images(str(jsonl_path), root=str(temp_dir), device="cuda")

            assert len(metas) == 1
            assert embeddings.shape == (1, 512)


class TestBuildIndex:
    """Test build_index function."""

    @mock.patch("src.indexing.build.embed_images")
    @mock.patch("src.indexing.build.build_faiss")
    @mock.patch("src.indexing.build.save_index")
    @mock.patch("src.indexing.build.detect_covers_root")
    @mock.patch("os.path.isabs")
    @mock.patch("os.path.exists")
    @mock.patch("os.path.join")
    def test_build_index_default_params(self, mock_join, mock_exists, mock_isabs, mock_detect_root,
                                       mock_save_index, mock_build_faiss, mock_embed_images):
        """Test building index with default parameters."""
        # Setup mocks
        mock_detect_root.return_value = "/covers/root"
        mock_isabs.return_value = False
        mock_join.side_effect = lambda *args: "/".join(args)
        mock_exists.return_value = True

        # Mock embeddings and metadata
        test_metas = [{"id": 1, "name": "Game 1"}]
        test_embeddings = torch.randn(1, 512)
        mock_embed_images.return_value = (test_metas, test_embeddings)

        # Mock FAISS index
        mock_index = mock.MagicMock()
        mock_build_faiss.return_value = mock_index

        # Mock numpy conversion
        test_embeddings_np = np.random.randn(1, 512).astype(np.float32)
        with mock.patch.object(test_embeddings, "numpy", return_value=test_embeddings_np):
            build_index()

        # Verify calls
        mock_embed_images.assert_called_once()
        mock_build_faiss.assert_called_once_with(mock.ANY, use_gpu=True)
        mock_save_index.assert_called_once()

    @mock.patch("src.indexing.build.embed_images")
    @mock.patch("src.indexing.build.build_faiss")
    @mock.patch("src.indexing.build.save_index")
    @mock.patch("src.indexing.build.detect_covers_root")
    @mock.patch("os.path.isabs")
    @mock.patch("os.path.exists")
    def test_build_index_custom_params(self, mock_exists, mock_isabs, mock_detect_root,
                                      mock_save_index, mock_build_faiss, mock_embed_images):
        """Test building index with custom parameters."""
        # Setup mocks
        mock_detect_root.return_value = "/covers/root"
        mock_isabs.return_value = False
        mock_exists.return_value = True

        # Mock embeddings and metadata
        test_metas = [{"id": 1, "name": "Game 1"}]
        test_embeddings = torch.randn(1, 512)
        mock_embed_images.return_value = (test_metas, test_embeddings)

        # Mock FAISS index
        mock_index = mock.MagicMock()
        mock_build_faiss.return_value = mock_index

        # Mock numpy conversion
        test_embeddings_np = np.random.randn(1, 512).astype(np.float32)
        with mock.patch.object(test_embeddings, "numpy", return_value=test_embeddings_np):
            build_index(
                jsonl_path="custom.jsonl",
                root="/custom/root",
                use_gpu=False,
                device="cuda",
                index_path="custom.faiss",
                npy_path="custom.npy",
                meta_path="custom.json",
            )

        # Verify calls with custom parameters
        mock_embed_images.assert_called_once()
        mock_build_faiss.assert_called_once_with(mock.ANY, use_gpu=False)
        mock_save_index.assert_called_once()

    @mock.patch("src.indexing.build.embed_images")
    @mock.patch("src.indexing.build.build_faiss")
    @mock.patch("src.indexing.build.save_index")
    @mock.patch("os.path.isabs")
    def test_build_index_absolute_paths(self, mock_isabs, mock_save_index, mock_build_faiss, mock_embed_images):
        """Test building index with absolute paths."""
        # Setup mocks
        mock_isabs.return_value = True  # All paths are absolute

        # Mock embeddings and metadata
        test_metas = [{"id": 1, "name": "Game 1"}]
        test_embeddings = torch.randn(1, 512)
        mock_embed_images.return_value = (test_metas, test_embeddings)

        # Mock FAISS index
        mock_index = mock.MagicMock()
        mock_build_faiss.return_value = mock_index

        # Mock numpy conversion
        test_embeddings_np = np.random.randn(1, 512).astype(np.float32)
        with mock.patch.object(test_embeddings, "numpy", return_value=test_embeddings_np):
            build_index(
                jsonl_path="/absolute/custom.jsonl",
                root="/absolute/root",
                index_path="/absolute/custom.faiss",
                npy_path="/absolute/custom.npy",
                meta_path="/absolute/custom.json",
            )

        # Verify calls with absolute paths
        mock_embed_images.assert_called_once()
        mock_build_faiss.assert_called_once_with(mock.ANY, use_gpu=True)
        mock_save_index.assert_called_once()
