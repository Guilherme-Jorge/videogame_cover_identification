"""Tests for path detection utilities."""

import os
import tempfile
import unittest.mock as mock
from pathlib import Path

import pytest

from src.utils.paths import detect_covers_root


class TestDetectCoversRoot:
    """Test detect_covers_root function."""

    def test_detect_covers_root_with_preferred_path(self):
        """Test detection with preferred path containing covers directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            covers_dir = Path(temp_dir) / "covers"
            covers_dir.mkdir()

            result = detect_covers_root(preferred=temp_dir)
            assert result == temp_dir

    def test_detect_covers_root_with_env_var(self):
        """Test detection using COVERS_ROOT environment variable."""
        with tempfile.TemporaryDirectory() as temp_dir:
            covers_dir = Path(temp_dir) / "covers"
            covers_dir.mkdir()

            with mock.patch.dict(os.environ, {"COVERS_ROOT": temp_dir}):
                result = detect_covers_root()
                assert result == temp_dir

    def test_detect_covers_root_in_script_dir(self):
        """Test detection in script directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            script_dir = Path(temp_dir) / "utils"
            script_dir.mkdir()
            covers_dir = script_dir / "covers"
            covers_dir.mkdir()

            # Mock the __file__ location to point to our script directory
            mock_file = str(script_dir / "paths.py")
            with mock.patch("src.utils.paths.__file__", mock_file):
                result = detect_covers_root()
                assert result == str(script_dir)

    def test_detect_covers_root_parent_dirs(self):
        """Test detection in parent directories."""
        with tempfile.TemporaryDirectory() as temp_base:
            temp_base = Path(temp_base)
            # Create nested structure: temp_base/utils/paths.py and temp_base/covers/
            utils_dir = temp_base / "utils"
            utils_dir.mkdir()
            covers_dir = temp_base / "covers"
            covers_dir.mkdir()

            mock_file = str(utils_dir / "paths.py")
            with mock.patch("src.utils.paths.__file__", mock_file):
                result = detect_covers_root()
                assert result == str(utils_dir)

    def test_detect_covers_root_igdb_extraction_dir(self):
        """Test detection in igdb-cover-extraction directory."""
        with tempfile.TemporaryDirectory() as temp_base:
            temp_base = Path(temp_base)
            igdb_dir = temp_base / "igdb-cover-extraction"
            igdb_dir.mkdir()
            covers_dir = igdb_dir / "covers"
            covers_dir.mkdir()

            mock_file = str(igdb_dir / "utils" / "paths.py")
            with mock.patch("src.utils.paths.__file__", mock_file):
                result = detect_covers_root()
                assert result == str(igdb_dir)

    def test_detect_covers_root_data_subdir(self):
        """Test detection in data subdirectory."""
        with tempfile.TemporaryDirectory() as temp_base:
            temp_base = Path(temp_base)
            data_dir = temp_base / "data"
            data_dir.mkdir()
            covers_dir = data_dir / "covers"
            covers_dir.mkdir()

            mock_file = str(temp_base / "utils" / "paths.py")
            with mock.patch("src.utils.paths.__file__", mock_file):
                result = detect_covers_root()
                assert result == str(temp_base / "utils")

    def test_detect_covers_root_no_covers_dir(self):
        """Test fallback when no covers directory is found."""
        with tempfile.TemporaryDirectory() as temp_dir:
            script_dir = str(Path(temp_dir) / "utils")
            mock_file = str(Path(temp_dir) / "utils" / "paths.py")

            with mock.patch("src.utils.paths.__file__", mock_file):
                result = detect_covers_root()
                assert result == script_dir  # Should fallback to script directory

    def test_detect_covers_root_preferred_takes_precedence(self):
        """Test that preferred path takes precedence over environment variable."""
        with tempfile.TemporaryDirectory() as temp_preferred:
            with tempfile.TemporaryDirectory() as temp_env:
                # Setup preferred directory with covers
                covers_preferred = Path(temp_preferred) / "covers"
                covers_preferred.mkdir()

                # Setup env directory with covers (should be ignored)
                covers_env = Path(temp_env) / "covers"
                covers_env.mkdir()

                with mock.patch.dict(os.environ, {"COVERS_ROOT": temp_env}):
                    result = detect_covers_root(preferred=temp_preferred)
                    assert result == temp_preferred

    def test_detect_covers_root_env_takes_precedence_over_script_dir(self):
        """Test that environment variable takes precedence over script directory."""
        with tempfile.TemporaryDirectory() as temp_env:
            with tempfile.TemporaryDirectory() as temp_script:
                # Setup env directory with covers
                covers_env = Path(temp_env) / "covers"
                covers_env.mkdir()

                mock_file = str(Path(temp_script) / "utils" / "paths.py")
                with mock.patch("src.utils.paths.__file__", mock_file):
                    with mock.patch.dict(os.environ, {"COVERS_ROOT": temp_env}):
                        result = detect_covers_root()
                        assert result == temp_env

    def test_detect_covers_root_handles_exceptions(self):
        """Test that exceptions during path checking are handled gracefully."""
        with tempfile.TemporaryDirectory() as temp_dir:
            covers_dir = Path(temp_dir) / "covers"
            covers_dir.mkdir()

            # Mock os.path.isdir to raise an exception sometimes
            original_isdir = os.path.isdir
            call_count = 0

            def mock_isdir(path):
                nonlocal call_count
                call_count += 1
                if call_count == 2:  # Fail on second call
                    raise PermissionError("Access denied")
                return original_isdir(path)

            with mock.patch("os.path.isdir", side_effect=mock_isdir):
                result = detect_covers_root(preferred=temp_dir)
                assert result == temp_dir  # Should still find it eventually
