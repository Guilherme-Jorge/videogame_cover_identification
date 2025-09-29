"""Tests for training dataset classes."""

import json
import tempfile
import unittest.mock as mock
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.training.dataset import (
    CoverAugDataset,
    PerspectiveJitter,
    _persp_coeffs,
    convert_to_rgb,
    glare_overlay,
)


class TestPerspCoeffs:
    """Test _persp_coeffs function."""

    def test_persp_coeffs(self):
        """Test computing perspective transformation coefficients."""
        src = [(0, 0), (100, 0), (100, 100), (0, 100)]
        dst = [(10, 10), (90, 5), (95, 95), (5, 105)]

        coeffs = _persp_coeffs(src, dst)

        assert len(coeffs) == 8
        assert all(isinstance(c, (int, float, np.number)) for c in coeffs)


class TestPerspectiveJitter:
    """Test PerspectiveJitter class."""

    def test_perspective_jitter(self):
        """Test applying perspective jitter to images."""
        jitter = PerspectiveJitter(max_warp=0.1)

        # Create test image
        img = Image.new("RGB", (100, 100), color="red")

        result = jitter(img)

        assert isinstance(result, Image.Image)
        assert result.size == (100, 100)

    def test_perspective_jitter_no_warp(self):
        """Test perspective jitter with zero warp."""
        jitter = PerspectiveJitter(max_warp=0.0)

        img = Image.new("RGB", (100, 100), color="red")
        result = jitter(img)

        # Should return the same image
        assert result.size == img.size


class TestGlareOverlay:
    """Test glare_overlay function."""

    def test_glare_overlay(self):
        """Test adding glare overlay to image."""
        img = Image.new("RGB", (100, 100), color="black")

        result = glare_overlay(img)

        assert isinstance(result, Image.Image)
        assert result.size == (100, 100)
        assert result.mode == "RGB"

    def test_glare_overlay_custom_alpha(self):
        """Test glare overlay with custom alpha range."""
        img = Image.new("RGB", (100, 100), color="white")

        result = glare_overlay(img, alpha_range=(0.5, 0.5))

        assert isinstance(result, Image.Image)


class TestConvertToRgb:
    """Test convert_to_rgb function."""

    def test_convert_rgb_image(self):
        """Test converting RGB image."""
        img = Image.new("RGB", (50, 50), color="red")

        result = convert_to_rgb(img)

        assert result.mode == "RGB"
        assert result.size == (50, 50)

    def test_convert_grayscale_to_rgb(self):
        """Test converting grayscale image to RGB."""
        img = Image.new("L", (50, 50), color=128)

        result = convert_to_rgb(img)

        assert result.mode == "RGB"
        assert result.size == (50, 50)


class TestCoverAugDataset:
    """Test CoverAugDataset class."""

    def test_dataset_init_valid_files(self):
        """Test dataset initialization with valid image files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)

            # Create test JSONL file
            jsonl_path = temp_dir / "test.jsonl"
            with open(jsonl_path, "w") as f:
                json.dump({"id": 1, "name": "Game 1", "local_filename": "game1.jpg"}, f)
                f.write("\n")
                json.dump({"id": 2, "name": "Game 2", "local_filename": "game2.jpg"}, f)
                f.write("\n")
                json.dump({"id": 3, "name": "Game 3"}, f)  # No local_filename
                f.write("\n")

            # Create test images
            img1_path = temp_dir / "game1.jpg"
            img2_path = temp_dir / "game2.jpg"
            img3_path = temp_dir / "missing.jpg"  # This will be missing

            img = Image.new("RGB", (100, 100), color="red")
            img.save(img1_path)
            img.save(img2_path)

            dataset = CoverAugDataset(str(jsonl_path), str(temp_dir))

            assert len(dataset) == 2  # Should skip missing image and no local_filename
            assert dataset.items[0]["id"] == 1
            assert dataset.items[1]["id"] == 2

    def test_dataset_init_no_valid_files(self):
        """Test dataset initialization when no valid files are found."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)

            # Create test JSONL file with no valid images
            jsonl_path = temp_dir / "test.jsonl"
            with open(jsonl_path, "w") as f:
                json.dump({"id": 1, "name": "Game 1"}, f)  # No local_filename
                f.write("\n")

            dataset = CoverAugDataset(str(jsonl_path), str(temp_dir))

            assert len(dataset) == 0

    def test_dataset_getitem(self):
        """Test getting items from dataset."""
        # Suppress PIL deprecation warnings from torchvision transforms
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*mode.*parameter.*deprecated.*",
                category=DeprecationWarning
            )
            warnings.filterwarnings(
                "ignore",
                message=".*'mode'.*deprecated.*",
                category=DeprecationWarning
            )

            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir = Path(temp_dir)

                # Create test JSONL file
                jsonl_path = temp_dir / "test.jsonl"
                with open(jsonl_path, "w") as f:
                    json.dump({"id": 1, "name": "Game 1", "local_filename": "game1.jpg"}, f)
                    f.write("\n")

                # Create test image
                img_path = temp_dir / "game1.jpg"
                img = Image.new("RGB", (200, 200), color="red")
                img.save(img_path)

                dataset = CoverAugDataset(str(jsonl_path), str(temp_dir))

                v1, v2 = dataset[0]

                assert isinstance(v1, torch.Tensor)
                assert isinstance(v2, torch.Tensor)
                assert v1.shape == (3, 224, 224)  # After resizing and cropping
                assert v2.shape == (3, 224, 224)

                # Check that v1 and v2 are different (augmentation)
                assert not torch.equal(v1, v2)

    def test_dataset_load_img(self):
        """Test loading image from path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)

            # Create dummy JSONL file
            jsonl_path = temp_dir / "dummy.jsonl"
            with open(jsonl_path, "w") as f:
                json.dump({"local_filename": "test.jpg"}, f)
                f.write("\n")

            img_path = temp_dir / "test.jpg"
            img = Image.new("RGB", (100, 100), color="blue")
            img.save(img_path)

            dataset = CoverAugDataset(str(jsonl_path), str(temp_dir))

            loaded = dataset._load_img("test.jpg")

            assert isinstance(loaded, Image.Image)
            assert loaded.mode == "RGB"

    def test_dataset_view_augmentation(self):
        """Test the _view method with augmentations."""
        # Suppress PIL deprecation warnings from torchvision transforms
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*mode.*parameter.*deprecated.*",
                category=DeprecationWarning
            )
            warnings.filterwarnings(
                "ignore",
                message=".*'mode'.*deprecated.*",
                category=DeprecationWarning
            )

            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir = Path(temp_dir)

                # Create dummy JSONL file
                jsonl_path = temp_dir / "dummy.jsonl"
                with open(jsonl_path, "w") as f:
                    json.dump({"local_filename": "test.jpg"}, f)
                    f.write("\n")

                dataset = CoverAugDataset(str(jsonl_path), str(temp_dir))

                # Create test image
                img = Image.new("RGB", (300, 300), color="green")

                # Apply view transformation
                result = dataset._view(img)

                assert isinstance(result, torch.Tensor)
                assert result.shape == (3, 224, 224)

                # Check tensor has reasonable values
                assert not torch.isnan(result).any()
                assert not torch.isinf(result).any()

    def test_dataset_len(self):
        """Test dataset length."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)

            # Create test JSONL file
            jsonl_path = temp_dir / "test.jsonl"
            with open(jsonl_path, "w") as f:
                json.dump({"id": 1, "name": "Game 1", "local_filename": "game1.jpg"}, f)
                f.write("\n")
                json.dump({"id": 2, "name": "Game 2", "local_filename": "game2.jpg"}, f)
                f.write("\n")

            # Create test images
            img1_path = temp_dir / "game1.jpg"
            img2_path = temp_dir / "game2.jpg"

            img = Image.new("RGB", (100, 100), color="red")
            img.save(img1_path)
            img.save(img2_path)

            dataset = CoverAugDataset(str(jsonl_path), str(temp_dir))

            assert len(dataset) == 2

    def test_dataset_augmentations_applied(self):
        """Test that various augmentations are applied."""
        # Suppress PIL deprecation warnings from torchvision transforms
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*mode.*parameter.*deprecated.*",
                category=DeprecationWarning
            )
            warnings.filterwarnings(
                "ignore",
                message=".*'mode'.*deprecated.*",
                category=DeprecationWarning
            )

            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir = Path(temp_dir)

                # Create test JSONL file
                jsonl_path = temp_dir / "test.jsonl"
                with open(jsonl_path, "w") as f:
                    json.dump({"id": 1, "name": "Game 1", "local_filename": "game1.jpg"}, f)
                    f.write("\n")

                # Create test image
                img_path = temp_dir / "game1.jpg"
                img = Image.new("RGB", (300, 300), color="red")
                img.save(img_path)

                dataset = CoverAugDataset(str(jsonl_path), str(temp_dir))

                # Get multiple views to check augmentation variety
                views = [dataset[0][0] for _ in range(10)]

                # Check that not all views are identical (augmentation is working)
                all_identical = all(torch.equal(views[0], view) for view in views[1:])
                assert not all_identical  # Augmentations should create variation
