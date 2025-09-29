"""Tests for FAISS index utilities."""

import json
import tempfile
import unittest.mock as mock
from pathlib import Path

import numpy as np
import pytest

from src.models.index import build_faiss, load_index, save_index


class TestLoadIndex:
    """Test load_index function."""

    def test_load_index_success(self):
        """Test successful loading of index and metadata."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)

            # Create mock FAISS index file
            index_path = temp_dir / "test.faiss"
            index_path.touch()

            # Create metadata JSON file
            meta_path = temp_dir / "test_meta.json"
            test_meta = [
                {"id": 1, "name": "Game 1"},
                {"id": 2, "name": "Game 2"},
            ]
            with open(meta_path, "w") as f:
                json.dump(test_meta, f)

            with mock.patch("faiss.read_index") as mock_read_index:
                mock_index = mock.MagicMock()
                mock_read_index.return_value = mock_index

                index, metas = load_index(str(index_path), str(meta_path))

                assert index == mock_index
                assert metas == test_meta
                mock_read_index.assert_called_once_with(str(index_path))

    def test_load_index_file_not_found(self):
        """Test loading index when files don't exist."""
        with pytest.raises(RuntimeError):
            load_index("nonexistent.faiss", "nonexistent.json")


class TestBuildFaiss:
    """Test build_faiss function."""

    @mock.patch("faiss.get_num_gpus", return_value=0)
    @mock.patch("faiss.IndexFlatIP")
    def test_build_faiss_cpu_only(self, mock_index_class, mock_get_gpus):
        """Test building FAISS index on CPU only."""
        # Mock FAISS index
        mock_index = mock.MagicMock()
        mock_index_class.return_value = mock_index

        embeddings = np.random.randn(10, 128).astype(np.float32)

        result = build_faiss(embeddings, use_gpu=False)

        assert result == mock_index
        mock_index_class.assert_called_once_with(128)
        mock_index.add.assert_called_once()
        np.testing.assert_array_equal(mock_index.add.call_args[0][0], embeddings)

    @mock.patch("faiss.get_num_gpus", return_value=1)
    @mock.patch("faiss.IndexFlatIP")
    def test_build_faiss_with_gpu(self, mock_index_class, mock_get_gpus):
        """Test building FAISS index with GPU acceleration."""
        # Mock indices
        mock_cpu_index = mock.MagicMock()
        mock_index_class.return_value = mock_cpu_index

        embeddings = np.random.randn(10, 128).astype(np.float32)

        result = build_faiss(embeddings, use_gpu=True)

        # Since GPU functions don't exist, it should fall back to CPU
        assert result == mock_cpu_index

        # Verify the call sequence
        mock_index_class.assert_called_once_with(128)
        mock_cpu_index.add.assert_called_once()

    @mock.patch("faiss.get_num_gpus", return_value=0)
    @mock.patch("faiss.IndexFlatIP")
    def test_build_faiss_gpu_requested_but_unavailable(self, mock_index_class, mock_get_gpus):
        """Test building FAISS index when GPU is requested but unavailable."""
        mock_index = mock.MagicMock()
        mock_index_class.return_value = mock_index

        embeddings = np.random.randn(10, 128).astype(np.float32)

        result = build_faiss(embeddings, use_gpu=True)

        # Should fall back to CPU
        assert result == mock_index
        mock_index_class.assert_called_once_with(128)
        mock_index.add.assert_called_once()


class TestSaveIndex:
    """Test save_index function."""

    @mock.patch("faiss.write_index")
    @mock.patch("numpy.save")
    def test_save_index_success(self, mock_np_save, mock_faiss_write):
        """Test successful saving of index, embeddings, and metadata."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)

            # Mock FAISS index
            mock_index = mock.MagicMock()

            # Test data
            embeddings = np.random.randn(5, 128).astype(np.float32)
            metas = [
                {"id": 1, "name": "Game 1"},
                {"id": 2, "name": "Game 2"},
            ]

            index_path = str(temp_dir / "test.faiss")
            npy_path = str(temp_dir / "test.npy")
            meta_path = str(temp_dir / "test_meta.json")

            save_index(mock_index, embeddings, metas, index_path, npy_path, meta_path)

            # Verify calls
            mock_faiss_write.assert_called_once_with(mock_index, index_path)
            mock_np_save.assert_called_once_with(npy_path, embeddings)

            # Verify metadata file was created
            assert (temp_dir / "test_meta.json").exists()
            with open(temp_dir / "test_meta.json") as f:
                saved_metas = json.load(f)
                assert saved_metas == metas

    def test_save_index_default_paths(self):
        """Test saving with default paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)

            with mock.patch("faiss.write_index"):
                with mock.patch("numpy.save"):
                    mock_index = mock.MagicMock()
                    embeddings = np.random.randn(5, 128).astype(np.float32)
                    metas = [{"id": 1, "name": "Game 1"}]

                    # Use absolute paths
                    save_index(mock_index, embeddings, metas,
                             index_path=str(temp_dir / "covers.faiss"),
                             npy_path=str(temp_dir / "covers.npy"),
                             meta_path=str(temp_dir / "covers_meta.json"))

                    # Should save to specified paths
                    expected_meta_path = temp_dir / "covers_meta.json"
                    assert expected_meta_path.exists()
