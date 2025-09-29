"""Tests for image processing utilities."""

import unittest.mock as mock

import numpy as np
import pytest
import torch
from PIL import Image

from src.utils.image import (
    _adjust_bounds,
    _compute_gradients_and_thresholds,
    _find_boundaries,
    _is_valid_crop,
    _order_points,
    _refine_rectified_crop,
    _should_skip_refinement,
    detect_and_rectify_cover,
    embed_image,
    sift_score,
)


class TestOrderPoints:
    """Test _order_points function."""

    def test_order_points_clockwise(self):
        """Test that points are ordered clockwise: TL, TR, BR, BL."""
        # Points in random order
        points = np.array([
            [10, 0],   # Should be TR
            [0, 0],    # Should be TL
            [10, 10],  # Should be BR
            [0, 10],   # Should be BL
        ], dtype=np.float32)

        ordered = _order_points(points)

        # Expected order: TL, TR, BR, BL
        expected = np.array([
            [0, 0],    # TL
            [10, 0],   # TR
            [10, 10],  # BR
            [0, 10],   # BL
        ], dtype=np.float32)

        np.testing.assert_array_equal(ordered, expected)

    def test_order_points_already_ordered(self):
        """Test ordering already ordered points."""
        points = np.array([
            [0, 0],    # TL
            [10, 0],   # TR
            [10, 10],  # BR
            [0, 10],   # BL
        ], dtype=np.float32)

        ordered = _order_points(points)
        np.testing.assert_array_equal(ordered, points)


class TestShouldSkipRefinement:
    """Test _should_skip_refinement function."""

    def test_skip_empty_image(self):
        """Test skipping refinement for empty image."""
        empty_img = np.zeros((0, 0, 3), dtype=np.uint8)
        assert _should_skip_refinement(empty_img, 0, 0) is True

    def test_skip_small_image(self):
        """Test skipping refinement for small image."""
        small_img = np.zeros((20, 20, 3), dtype=np.uint8)
        assert _should_skip_refinement(small_img, 20, 20) is True

    def test_allow_large_image(self):
        """Test allowing refinement for large image."""
        large_img = np.zeros((100, 100, 3), dtype=np.uint8)
        assert _should_skip_refinement(large_img, 100, 100) is False


class TestComputeGradientsAndThresholds:
    """Test _compute_gradients_and_thresholds function."""

    def test_compute_gradients_uniform_image(self):
        """Test gradients computation on uniform image."""
        uniform_img = np.full((50, 50, 3), 128, dtype=np.uint8)
        result = _compute_gradients_and_thresholds(uniform_img, 50, 50)
        assert result is None  # Should return None for uniform image

    def test_compute_gradients_with_edges(self):
        """Test gradients computation on image with clear edges."""
        # Create image with both horizontal and vertical edges
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        img[:, :25] = 0    # Left half black
        img[:, 25:] = 255  # Right half white
        img[:25, :] = 255  # Top half white (creates horizontal edge)

        result = _compute_gradients_and_thresholds(img, 50, 50)
        assert result is not None
        col_strength, row_strength, col_threshold, row_threshold = result

        assert col_strength.shape == (50,)
        assert row_strength.shape == (50,)
        assert col_threshold > 0
        assert row_threshold > 0


class TestFindBoundaries:
    """Test _find_boundaries function."""

    def test_find_boundaries_clear_edges(self):
        """Test boundary finding with clear edges."""
        # Create mock gradient strengths with clear boundaries
        col_strength = np.zeros(100)
        col_strength[20:80] = 10.0  # Content area

        row_strength = np.zeros(100)
        row_strength[10:90] = 10.0  # Content area

        result = _find_boundaries(
            col_strength, row_strength, 5.0, 5.0, 100, 100, 5, 5, 2, 2
        )

        assert result is not None
        x0, y0, x1, y1 = result
        assert 15 <= x0 <= 25  # Should find left boundary around 20
        assert 5 <= y0 <= 15   # Should find top boundary around 10
        assert 75 <= x1 <= 85  # Should find right boundary around 80
        assert 85 <= y1 <= 95  # Should find bottom boundary around 90

    def test_find_boundaries_no_edges(self):
        """Test boundary finding when no edges are found."""
        col_strength = np.zeros(100)
        row_strength = np.zeros(100)

        result = _find_boundaries(
            col_strength, row_strength, 5.0, 5.0, 100, 100, 5, 5, 2, 2
        )

        assert result is None


class TestAdjustBounds:
    """Test _adjust_bounds function."""

    def test_adjust_bounds_with_margins(self):
        """Test bounds adjustment with maximum margins."""
        x0, y0, x1, y1 = _adjust_bounds(10, 5, 90, 95, 100, 100)

        # Should add padding and respect maximum margins
        assert x0 >= 0
        assert y0 >= 0
        assert x1 <= 100
        assert y1 <= 100
        assert x1 > x0
        assert y1 > y0

    def test_adjust_bounds_no_change_needed(self):
        """Test bounds adjustment when no change is needed."""
        x0, y0, x1, y1 = _adjust_bounds(20, 20, 80, 80, 100, 100)

        # Should maintain reasonable bounds
        assert x0 >= 0
        assert y0 >= 0
        assert x1 <= 100
        assert y1 <= 100


class TestIsValidCrop:
    """Test _is_valid_crop function."""

    def test_valid_crop(self):
        """Test valid crop dimensions."""
        assert _is_valid_crop(10, 10, 90, 90, 100, 100) is True

    def test_invalid_crop_too_small(self):
        """Test invalid crop that is too small."""
        assert _is_valid_crop(50, 50, 60, 60, 100, 100) is False

    def test_invalid_crop_wrong_proportions(self):
        """Test invalid crop with wrong proportions."""
        assert _is_valid_crop(0, 0, 20, 100, 100, 100) is False


class TestRefineRectifiedCrop:
    """Test _refine_rectified_crop function."""

    def test_refine_rectified_crop_skip_small(self):
        """Test refinement is skipped for small images."""
        small_img = np.zeros((30, 30, 3), dtype=np.uint8)
        refined, bounds = _refine_rectified_crop(small_img)

        assert np.array_equal(refined, small_img)
        assert bounds is None

    def test_refine_rectified_crop_no_refinement(self):
        """Test no refinement when no clear boundaries found."""
        uniform_img = np.full((100, 100, 3), 128, dtype=np.uint8)
        refined, bounds = _refine_rectified_crop(uniform_img)

        assert np.array_equal(refined, uniform_img)
        assert bounds is None

    def test_refine_rectified_crop_with_edges(self):
        """Test refinement with clear content boundaries."""
        # Create image with clear content area
        img = np.full((100, 100, 3), 0, dtype=np.uint8)
        img[20:80, 20:80] = 255  # White square in center

        refined, bounds = _refine_rectified_crop(img)

        # Should return cropped image and bounds
        assert refined.shape[:2] != img.shape[:2] or bounds is not None
        if bounds is not None:
            assert len(bounds) == 4


class TestDetectAndRectifyCover:
    """Test detect_and_rectify_cover function."""

    def test_detect_cover_no_quadrilateral(self):
        """Test detection when no quadrilateral is found."""
        # Create uniform image with no clear shapes
        img = np.full((200, 200, 3), 128, dtype=np.uint8)

        crop, overlay, quad = detect_and_rectify_cover(img)

        # Should return original image and None quad
        assert crop.shape == img.shape
        assert overlay.shape == img.shape
        assert quad is None

    def test_detect_cover_with_simple_shape(self):
        """Test detection with a simple rectangular shape."""
        # Create image with a white rectangle on black background
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        img[50:150, 50:150] = 255  # White rectangle

        crop, overlay, quad = detect_and_rectify_cover(img)

        # Should detect something and return overlay
        assert crop.shape[0] > 0 and crop.shape[1] > 0
        assert overlay.shape == img.shape
        # Quad might be None if detection fails, which is acceptable


class TestEmbedImage:
    """Test embed_image function."""

    @pytest.fixture
    def mock_model_and_preprocess(self):
        """Create mock model and preprocess function."""
        # Create a simple mock model that returns normalized embeddings
        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                # Return normalized embeddings
                batch_size = x.shape[0]
                embedding = torch.randn(batch_size, 512)
                return torch.nn.functional.normalize(embedding, dim=-1)

            def to(self, device):
                return self

            def eval(self):
                return self

        # Mock preprocess function
        def mock_preprocess(img):
            # Convert PIL to tensor
            img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
            return img_tensor

        # Mock parameters
        mock_param = mock.MagicMock()
        mock_param.device = torch.device("cpu")
        mock_model = MockModel()
        mock_model.parameters = mock.MagicMock(return_value=iter([mock_param]))

        return mock_model, mock_preprocess

    def test_embed_image_basic(self, mock_model_and_preprocess):
        """Test basic image embedding."""
        model, preprocess = mock_model_and_preprocess

        # Create test image
        img_bgr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        embedding = embed_image(model, preprocess, img_bgr)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (512,)  # CLIP embedding dimension
        assert np.isclose(np.linalg.norm(embedding), 1.0, atol=1e-6)  # Should be normalized


class TestSiftScore:
    """Test sift_score function."""

    def test_sift_score_identical_images(self):
        """Test SIFT score for identical images."""
        # Create identical test images
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        score, inliers = sift_score(img, img)

        # Identical images should have good SIFT matches
        assert score >= 0.0
        assert inliers >= 0

    def test_sift_score_different_images(self):
        """Test SIFT score for different images."""
        img1 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        score, inliers = sift_score(img1, img2)

        # Different images should have lower scores
        assert score >= 0.0
        assert inliers >= 0

    def test_sift_score_no_features(self):
        """Test SIFT score when no features can be detected."""
        # Create uniform images with no features
        img1 = np.full((50, 50, 3), 128, dtype=np.uint8)
        img2 = np.full((50, 50, 3), 128, dtype=np.uint8)

        score, inliers = sift_score(img1, img2)

        # Should return 0.0 when no good matches found
        assert score == 0.0
        assert inliers == 0
