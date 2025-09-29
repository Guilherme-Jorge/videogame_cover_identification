"""Tests for cover search functionality."""

import os
import tempfile
import unittest.mock as mock
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from src.search.search import (
    _collect_all_candidates,
    _extract_base_name,
    _extract_base_names_from_candidates,
    _find_base_game,
    _find_base_game_from_all_metas,
    _find_base_games_for_groups,
    _find_matching_games_in_metadata,
    _format_search_results,
    _group_games_by_base_name,
    _rerank_with_geometric_scoring,
    _search_index,
    _select_best_base_game,
    search_cover,
)


class TestExtractBaseName:
    """Test _extract_base_name function."""

    def test_extract_base_name_no_edition(self):
        """Test extracting base name from game without edition suffix."""
        assert _extract_base_name("The Legend of Zelda") == "The Legend of Zelda"
        assert _extract_base_name("Super Mario Bros") == "Super Mario Bros"

    def test_extract_base_name_with_edition(self):
        """Test extracting base name from game with edition suffix."""
        test_cases = [
            ("The Legend of Zelda: Breath of the Wild - Special Edition", "The Legend of Zelda: Breath of the Wild"),
            ("Super Mario Odyssey - Deluxe Edition", "Super Mario Odyssey"),
            ("Final Fantasy VII - Remastered", "Final Fantasy VII"),
            ("Halo: Combat Evolved - Anniversary Edition", "Halo: Combat Evolved"),
            ("Portal 2 - Game of the Year Edition", "Portal 2"),
            ("The Witcher 3: Wild Hunt - Complete Edition", "The Witcher 3: Wild Hunt"),
        ]

        for input_name, expected in test_cases:
            assert _extract_base_name(input_name) == expected

    def test_extract_base_name_case_insensitive(self):
        """Test that base name extraction is case insensitive."""
        assert _extract_base_name("GAME - SPECIAL EDITION") == "GAME"
        assert _extract_base_name("game - deluxe edition") == "game"

    def test_extract_base_name_multiple_editions(self):
        """Test extracting base name when multiple edition suffixes are present."""
        assert _extract_base_name("Game - Special Edition - Deluxe Edition") == "Game"


class TestFindBaseGame:
    """Test _find_base_game function."""

    def test_find_base_game_single_candidate(self):
        """Test finding base game with single candidate."""
        candidates = [{"name": "Super Mario Bros", "id": 1}]
        metas = [{"name": "Super Mario Bros", "id": 1}]

        result = _find_base_game(candidates, metas)
        assert result == candidates[0]

    def test_find_base_game_multiple_candidates_same_base(self):
        """Test finding base game among multiple candidates with same base name."""
        candidates = [
            {"name": "Super Mario Bros", "id": 1},
            {"name": "Super Mario Bros - Special Edition", "id": 2},
        ]
        metas = candidates.copy()

        result = _find_base_game(candidates, metas)
        assert result["id"] == 1  # Should pick the shortest name (base game)

    def test_find_base_game_different_groups(self):
        """Test finding base game when candidates have different base names."""
        candidates = [
            {"name": "Super Mario Bros", "id": 1},
            {"name": "The Legend of Zelda", "id": 2},
        ]
        metas = candidates.copy()

        result = _find_base_game(candidates, metas)
        # Should pick the largest group (both have 1, so picks first)
        assert result["id"] in [1, 2]

    def test_find_base_game_empty_candidates(self):
        """Test finding base game with empty candidates."""
        result = _find_base_game([], [])
        assert result is None


class TestExtractBaseNamesFromCandidates:
    """Test _extract_base_names_from_candidates function."""

    def test_extract_base_names(self):
        """Test extracting unique base names from candidates."""
        candidates = [
            {"name": "Game A - Special Edition"},
            {"name": "Game B"},
            {"name": "Game A - Deluxe Edition"},
            {"name": "Game C - Remastered"},
        ]

        result = _extract_base_names_from_candidates(candidates)
        expected = {"Game A", "Game B", "Game C"}
        assert result == expected

    def test_extract_base_names_empty(self):
        """Test extracting base names from empty candidates."""
        result = _extract_base_names_from_candidates([])
        assert result == set()


class TestFindMatchingGamesInMetadata:
    """Test _find_matching_games_in_metadata function."""

    def test_find_matching_games(self):
        """Test finding games that match base names."""
        base_names = {"Super Mario", "Zelda"}
        metas = [
            {"id": 1, "name": "Super Mario Bros"},
            {"id": 2, "name": "Super Mario World"},
            {"id": 3, "name": "The Legend of Zelda"},
            {"id": 4, "name": "Pokemon Red"},
        ]

        result = _find_matching_games_in_metadata(base_names, metas)

        assert len(result) == 3
        names = {game["name"] for game in result}
        assert "Super Mario Bros" in names
        assert "Super Mario World" in names
        assert "The Legend of Zelda" in names

    def test_find_matching_games_empty(self):
        """Test finding matching games with empty inputs."""
        result = _find_matching_games_in_metadata(set(), [])
        assert result == []


class TestGroupGamesByBaseName:
    """Test _group_games_by_base_name function."""

    def test_group_games(self):
        """Test grouping games by base name."""
        games = [
            {"base_name": "Mario", "id": 1, "name": "Super Mario Bros"},
            {"base_name": "Mario", "id": 2, "name": "Super Mario World"},
            {"base_name": "Zelda", "id": 3, "name": "The Legend of Zelda"},
        ]

        result = _group_games_by_base_name(games)

        assert "Mario" in result
        assert "Zelda" in result
        assert len(result["Mario"]) == 2
        assert len(result["Zelda"]) == 1


class TestFindBaseGamesForGroups:
    """Test _find_base_games_for_groups function."""

    def test_find_base_games(self):
        """Test finding base game for each group."""
        groups = {
            "Mario": [
                {"id": 1, "name": "Super Mario Bros Deluxe", "base_name": "Mario"},
                {"id": 2, "name": "Super Mario Bros", "base_name": "Mario"},
            ],
            "Zelda": [
                {"id": 3, "name": "The Legend of Zelda", "base_name": "Zelda"},
            ]
        }

        result = _find_base_games_for_groups(groups)

        assert len(result) == 2
        # Should pick shortest name for Mario group
        mario_games = [g for g in result if g["base_name"] == "Mario"]
        assert len(mario_games) == 1
        assert mario_games[0]["id"] == 2  # "Super Mario Bros" is shorter


class TestSelectBestBaseGame:
    """Test _select_best_base_game function."""

    def test_select_best_base_game_with_candidates(self):
        """Test selecting best base game when candidates are available."""
        base_games = [
            {"id": 1, "name": "Super Mario Bros", "base_name": "Mario"},
            {"id": 2, "name": "Super Mario World", "base_name": "Mario"},
        ]
        base_names = {"Mario"}
        candidates = [{"name": "Super Mario Bros - Special Edition"}]

        result = _select_best_base_game(base_games, base_names, candidates)

        assert result["id"] == 1  # Should match the base name of the best candidate

    def test_select_best_base_game_fallback(self):
        """Test fallback selection when no direct match."""
        base_games = [
            {"id": 1, "name": "Super Mario Bros", "base_name": "Mario"},
            {"id": 2, "name": "Long Game Name With Many Words", "base_name": "Long Game"},
        ]

        result = _select_best_base_game(base_games, set(), [])

        assert result["id"] == 1  # Should pick shortest name

    def test_select_best_base_game_empty(self):
        """Test selecting base game with empty inputs."""
        result = _select_best_base_game([], set(), [])
        assert result is None


class TestSearchIndex:
    """Test _search_index function."""

    def test_search_index(self):
        """Test searching the FAISS index."""
        # Mock FAISS index
        mock_index = mock.MagicMock()
        mock_index.search.return_value = (
            np.array([[0.9, 0.8, 0.7]]),  # distances
            np.array([[0, 1, 2]])        # indices
        )

        metas = [
            {"id": 1, "name": "Game 1"},
            {"id": 2, "name": "Game 2"},
            {"id": 3, "name": "Game 3"},
        ]

        query_embedding = np.random.randn(1, 512).astype(np.float32)
        topk = 3

        result = _search_index(mock_index, metas, query_embedding, topk)

        assert len(result) == 3
        assert result[0] == (0.9, 0, metas[0])
        assert result[1] == (0.8, 1, metas[1])
        assert result[2] == (0.7, 2, metas[2])


class TestRerankWithGeometricScoring:
    """Test _rerank_with_geometric_scoring function."""

    def test_rerank_with_valid_images(self):
        """Test reranking with valid candidate images."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)

            # Create test images
            img1_path = temp_dir / "game1.jpg"
            img2_path = temp_dir / "game2.jpg"

            # Create dummy images
            test_img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(img1_path), test_img)
            cv2.imwrite(str(img2_path), test_img)

            candidates = [
                (0.9, 0, {"local_filename": "game1.jpg"}),
                (0.8, 1, {"local_filename": "game2.jpg"}),
                (0.7, 2, {"local_filename": "missing.jpg"}),
            ]

            query_bgr = np.zeros((100, 100, 3), dtype=np.uint8)
            rerank_k = 3

            with mock.patch("src.search.search.detect_covers_root", return_value=str(temp_dir)):
                with mock.patch("src.search.search.sift_score", return_value=(0.5, 10)):
                    result = _rerank_with_geometric_scoring(candidates, query_bgr, rerank_k)

            assert len(result) == 3
            # Should be sorted by geometric score, then cosine
            assert result[0][1] >= result[1][1]  # Higher geometric scores first

    def test_rerank_with_missing_images(self):
        """Test reranking when candidate images are missing."""
        candidates = [
            (0.9, 0, {"local_filename": "missing1.jpg"}),
            (0.8, 1, {"local_filename": "missing2.jpg"}),
        ]

        query_bgr = np.zeros((100, 100, 3), dtype=np.uint8)

        with mock.patch("src.search.search.detect_covers_root", return_value="/fake/path"):
            with mock.patch("os.path.exists", return_value=False):
                result = _rerank_with_geometric_scoring(candidates, query_bgr, 2)

        assert len(result) == 2
        # Missing images should get geometric score of 0.0
        assert all(r[1] == 0.0 for r in result)


class TestCollectAllCandidates:
    """Test _collect_all_candidates function."""

    def test_collect_candidates(self):
        """Test collecting all candidates for base game identification."""
        reranked = [
            (0.9, 0.8, 0, {"id": 1, "name": "Game 1", "cover_id": 101, "cover_url": "url1", "local_filename": "file1.jpg"}),
            (0.8, 0.7, 1, {"id": 2, "name": "Game 2", "cover_id": 102, "cover_url": "url2", "local_filename": "file2.jpg"}),
        ]

        result = _collect_all_candidates(reranked)

        assert len(result) == 2
        assert result[0]["cosine"] == 0.9
        assert result[0]["id"] == 1
        assert result[1]["cosine"] == 0.8
        assert result[1]["id"] == 2


class TestFormatSearchResults:
    """Test _format_search_results function."""

    def test_format_results(self):
        """Test formatting search results."""
        sim = 0.9
        geo = 0.8
        meta = {"id": 1, "name": "Game 1", "cover_id": 101, "cover_url": "url1", "local_filename": "file1.jpg"}
        reranked = [
            (0.9, 0.8, 0, meta),
            (0.8, 0.7, 1, {"id": 2, "name": "Game 2", "cover_id": 102, "cover_url": "url2", "local_filename": "file2.jpg"}),
        ]
        base_game = {"id": 1, "name": "Game 1"}

        result = _format_search_results(sim, geo, meta, reranked, base_game)

        assert result["cosine"] == 0.9
        assert result["geom_score"] == 0.8
        assert result["match"]["id"] == 1
        assert len(result["alternatives"]) == 1
        assert result["base"]["id"] == 1

    def test_format_results_no_base_game(self):
        """Test formatting results without base game."""
        sim = 0.9
        geo = 0.8
        meta = {"id": 1, "name": "Game 1", "cover_id": 101, "cover_url": "url1", "local_filename": "file1.jpg"}
        reranked = [(0.9, 0.8, 0, meta)]
        base_game = None

        result = _format_search_results(sim, geo, meta, reranked, base_game)

        assert result["base"] is None


class TestSearchCoverIntegration:
    """Integration tests for search_cover function."""

    @mock.patch("src.search.search.load_index")
    @mock.patch("src.search.search.load_encoder")
    @mock.patch("src.search.search.detect_covers_root")
    @mock.patch("cv2.imread")
    @mock.patch("cv2.imwrite")
    @mock.patch("src.search.search.detect_and_rectify_cover")
    def test_search_cover_basic(self, mock_detect_cover, mock_imwrite, mock_imread,
                               mock_detect_root, mock_load_encoder, mock_load_index):
        """Test basic search_cover functionality."""
        # Setup mocks
        mock_detect_root.return_value = "/covers/root"

        mock_index = mock.MagicMock()
        # Return arrays with shape (1, 25) to match default topk
        distances = np.random.rand(1, 25).astype(np.float32)
        indices = np.random.randint(0, 1, (1, 25)).astype(np.int64)  # Only index 0 since metas has 1 element
        mock_index.search.return_value = (distances, indices)
        mock_index.d = 512

        metas = [{"id": 1, "name": "Game 1", "cover_id": 101, "cover_url": "url1", "local_filename": "file1.jpg"}]
        mock_load_index.return_value = (mock_index, metas)

        mock_model = mock.MagicMock()
        mock_model.return_value = torch.randn(1, 512)
        mock_preprocess = mock.MagicMock()
        mock_load_encoder.return_value = (mock_model, mock_preprocess)

        # Mock image reading and processing
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_imread.return_value = test_img

        mock_detect_cover.return_value = (test_img, test_img, np.array([[0, 0], [100, 0], [100, 100], [0, 100]]))

        # Mock embedding
        with mock.patch("src.search.search.embed_image", return_value=np.random.randn(512)):
            # Mock reranking
            with mock.patch("src.search.search._rerank_with_geometric_scoring", return_value=[(0.9, 0.8, 0, metas[0])]):
                with mock.patch("src.search.search._find_base_game_from_candidates", return_value=None):
                    result = search_cover("test_image.jpg")

        assert "cosine" in result
        assert "geom_score" in result
        assert "match" in result
        assert "confident" in result
