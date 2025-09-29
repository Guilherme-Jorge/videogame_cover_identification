"""Tests for data loading utilities."""

import json
import tempfile
from pathlib import Path

import pytest

from src.utils.data import load_jsonl


class TestLoadJsonl:
    """Test load_jsonl function."""

    def test_load_valid_jsonl(self):
        """Test loading valid JSONL file."""
        data = [
            {"id": 1, "name": "Game 1", "local_filename": "game1.jpg"},
            {"id": 2, "name": "Game 2", "local_filename": "game2.jpg"},
            {"id": 3, "name": "Game 3"},
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for item in data:
                json.dump(item, f)
                f.write("\n")
            temp_path = f.name

        try:
            result = list(load_jsonl(temp_path))
            assert len(result) == 3
            assert result[0] == {"id": 1, "name": "Game 1", "local_filename": "game1.jpg"}
            assert result[1] == {"id": 2, "name": "Game 2", "local_filename": "game2.jpg"}
            assert result[2] == {"id": 3, "name": "Game 3"}
        finally:
            Path(temp_path).unlink()

    def test_load_jsonl_with_empty_lines(self):
        """Test loading JSONL file with empty lines."""
        data = [
            {"id": 1, "name": "Game 1"},
            "",  # Empty line
            {"id": 2, "name": "Game 2"},
            "   ",  # Whitespace only line
            {"id": 3, "name": "Game 3"},
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for item in data:
                if isinstance(item, dict):
                    json.dump(item, f)
                else:
                    f.write(item)
                f.write("\n")
            temp_path = f.name

        try:
            result = list(load_jsonl(temp_path))
            assert len(result) == 3
            assert result[0] == {"id": 1, "name": "Game 1"}
            assert result[1] == {"id": 2, "name": "Game 2"}
            assert result[2] == {"id": 3, "name": "Game 3"}
        finally:
            Path(temp_path).unlink()

    def test_load_empty_jsonl(self):
        """Test loading empty JSONL file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            temp_path = f.name

        try:
            result = list(load_jsonl(temp_path))
            assert len(result) == 0
        finally:
            Path(temp_path).unlink()

    def test_load_jsonl_with_mixed_content(self):
        """Test loading JSONL with various data types."""
        data = [
            {"id": 1, "name": "Game 1", "active": True, "score": 95.5},
            {"id": 2, "name": "Game 2", "tags": ["action", "adventure"]},
            {"id": 3, "name": "Game 3", "metadata": {"year": 2023, "platform": "PC"}},
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for item in data:
                json.dump(item, f)
                f.write("\n")
            temp_path = f.name

        try:
            result = list(load_jsonl(temp_path))
            assert len(result) == 3
            assert result[0]["active"] is True
            assert result[0]["score"] == 95.5
            assert result[1]["tags"] == ["action", "adventure"]
            assert result[2]["metadata"]["year"] == 2023
        finally:
            Path(temp_path).unlink()

    def test_load_jsonl_file_not_found(self):
        """Test loading non-existent file."""
        with pytest.raises(FileNotFoundError):
            list(load_jsonl("nonexistent_file.jsonl"))

    def test_load_jsonl_invalid_json(self):
        """Test loading file with invalid JSON."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"id": 1, "name": "Game 1"}\n')
            f.write('{"id": 2, "invalid": json}\n')
            f.write('{"id": 3, "name": "Game 3"}\n')
            temp_path = f.name

        try:
            with pytest.raises(json.JSONDecodeError):
                list(load_jsonl(temp_path))
        finally:
            Path(temp_path).unlink()
