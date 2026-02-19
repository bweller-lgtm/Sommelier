"""Extended tests for Router covering uncovered code paths."""
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock

from taster.core.config import load_config
from taster.core.profiles import TasteProfile, CategoryDefinition
from taster.classification.routing import Router


@pytest.fixture
def config():
    return load_config(Path("config.yaml"))


@pytest.fixture
def router(config):
    return Router(config)



# ── Singleton routing ────────────────────────────────────────────────


class TestSingletonRouting:

    def test_ignore_category(self, router):
        assert router.route_singleton({"classification": "Ignore"}) == "Ignore"

    def test_no_children(self, router):
        assert router.route_singleton({
            "classification": "Share", "score": 5,
            "contains_children": False,
        }) == "Ignore"

    def test_inappropriate(self, router):
        assert router.route_singleton({
            "classification": "Share", "score": 5,
            "contains_children": True, "is_appropriate": False,
        }) == "Review"

    def test_share_high_score(self, router):
        assert router.route_singleton({
            "classification": "Share", "score": 4,
            "contains_children": True, "is_appropriate": True,
        }) == "Share"

    def test_share_mid_score_to_review(self, router):
        assert router.route_singleton({
            "classification": "Share", "score": 3,
            "contains_children": True, "is_appropriate": True,
        }) == "Review"

    def test_storage(self, router):
        assert router.route_singleton({
            "classification": "Storage", "score": 2,
            "contains_children": True, "is_appropriate": True,
        }) == "Storage"

    def test_very_low_score(self, router):
        assert router.route_singleton({
            "classification": "Share", "score": 1,
            "contains_children": True, "is_appropriate": True,
        }) == "Storage"

    def test_default_review(self, router):
        """Mid-score non-Share, non-Storage ends up in Review."""
        assert router.route_singleton({
            "classification": "Review", "score": 3,
            "contains_children": True, "is_appropriate": True,
        }) == "Review"


# ── Video routing ────────────────────────────────────────────────────


class TestVideoRouting:

    def test_delegates_to_singleton(self, router):
        result = router.route_video({
            "classification": "Share", "score": 5,
            "contains_children": True, "is_appropriate": True,
        })
        assert result == "Share"


# ── Burst routing ────────────────────────────────────────────────────


class TestBurstRouting:

    def test_mismatch_raises(self, router):
        with pytest.raises(ValueError, match="Mismatch"):
            router.route_burst(
                [Path("a.jpg")],
                [{"rank": 1}, {"rank": 2}],
            )

    def test_ignore_in_burst(self, router):
        result = router.route_burst(
            [Path("a.jpg")],
            [{"classification": "Ignore", "rank": 1, "score": 5}],
        )
        assert result == ["Ignore"]

    def test_no_children_in_burst(self, router):
        result = router.route_burst(
            [Path("a.jpg")],
            [{"classification": "Share", "rank": 1, "score": 5, "contains_children": False}],
        )
        assert result == ["Ignore"]

    def test_inappropriate_in_burst(self, router):
        result = router.route_burst(
            [Path("a.jpg")],
            [{"classification": "Share", "rank": 1, "score": 5, "is_appropriate": False}],
        )
        assert result == ["Review"]

    def test_top_ranked_share(self, router):
        result = router.route_burst(
            [Path("a.jpg"), Path("b.jpg")],
            [
                {"classification": "Share", "score": 5, "rank": 1},
                {"classification": "Share", "score": 4, "rank": 2},
            ],
        )
        assert result[0] == "Share"
        assert result[1] == "Share"

    def test_mid_ranked_review(self, config):
        config.classification.burst_rank_consider = 1
        config.classification.burst_rank_review = 3
        r = Router(config)

        result = r.route_burst(
            [Path("a.jpg"), Path("b.jpg"), Path("c.jpg")],
            [
                {"classification": "Share", "score": 5, "rank": 1},
                {"classification": "Share", "score": 4, "rank": 2},
                {"classification": "Share", "score": 4, "rank": 3},
            ],
        )
        assert result[0] == "Share"
        assert result[1] == "Review"
        assert result[2] == "Review"

    def test_low_ranked_storage(self, config):
        config.classification.burst_rank_consider = 1
        config.classification.burst_rank_review = 2
        r = Router(config)

        result = r.route_burst(
            [Path("a.jpg"), Path("b.jpg"), Path("c.jpg")],
            [
                {"classification": "Share", "score": 5, "rank": 1},
                {"classification": "Share", "score": 4, "rank": 2},
                {"classification": "Share", "score": 2, "rank": 5},
            ],
        )
        assert result[2] == "Storage"

    def test_top_ranked_low_score(self, router):
        result = router.route_burst(
            [Path("a.jpg")],
            [{"classification": "Share", "score": 1, "rank": 1}],
        )
        assert result[0] == "Storage"

    def test_diversity_check_rejects_similar(self, config):
        config.classification.enable_diversity_check = True
        config.classification.diversity_similarity_threshold = 0.9
        r = Router(config)

        # Two very similar embeddings
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.99, 0.01, 0.0],  # Very similar to first
        ])
        # Normalize
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        result = r.route_burst(
            [Path("a.jpg"), Path("b.jpg")],
            [
                {"classification": "Share", "score": 5, "rank": 1},
                {"classification": "Share", "score": 4, "rank": 2},
            ],
            embeddings=embeddings,
        )
        assert result[0] == "Share"
        assert result[1] == "Review"  # Too similar, but score 4 >= review threshold

    def test_diversity_check_accepts_different(self, config):
        config.classification.enable_diversity_check = True
        config.classification.diversity_similarity_threshold = 0.95
        r = Router(config)

        # Two very different embeddings
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],  # Orthogonal
        ])

        result = r.route_burst(
            [Path("a.jpg"), Path("b.jpg")],
            [
                {"classification": "Share", "score": 5, "rank": 1},
                {"classification": "Share", "score": 4, "rank": 2},
            ],
            embeddings=embeddings,
        )
        assert result[0] == "Share"
        assert result[1] == "Share"

    def test_no_embeddings_assumes_diverse(self, config):
        config.classification.enable_diversity_check = True
        r = Router(config)

        result = r.route_burst(
            [Path("a.jpg"), Path("b.jpg")],
            [
                {"classification": "Share", "score": 5, "rank": 1},
                {"classification": "Share", "score": 4, "rank": 2},
            ],
            embeddings=None,
        )
        assert result[0] == "Share"
        assert result[1] == "Share"


# ── Diversity check (standalone) ─────────────────────────────────────


class TestDiversityCheck:

    def test_disabled(self, router):
        is_diverse, conf, reason = router.check_diversity(Path("a.jpg"), Path("b.jpg"))
        assert is_diverse is True

    def test_no_client(self, config):
        r = Router(config, gemini_client=None)
        is_diverse, _, _ = r.check_diversity(Path("a.jpg"), Path("b.jpg"), gemini_check=True)
        assert is_diverse is True

    def test_with_client(self, config):
        mock_client = MagicMock()
        mock_client.generate_json.return_value = {
            "is_diverse": False, "score": 5, "reasoning": "identical"
        }
        r = Router(config, gemini_client=mock_client)
        with MagicMock() as mock_utils:
            from unittest.mock import patch
            with patch("taster.classification.routing.ImageUtils") as mock_iu:
                mock_iu.load_and_fix_orientation.return_value = MagicMock()
                is_diverse, conf, reason = r.check_diversity(
                    Path("a.jpg"), Path("b.jpg"), gemini_check=True
                )
                assert is_diverse is False

    def test_load_failure(self, config):
        mock_client = MagicMock()
        r = Router(config, gemini_client=mock_client)
        from unittest.mock import patch
        with patch("taster.classification.routing.ImageUtils") as mock_iu:
            mock_iu.load_and_fix_orientation.return_value = None
            is_diverse, _, _ = r.check_diversity(
                Path("a.jpg"), Path("b.jpg"), gemini_check=True
            )
            assert is_diverse is True


# ── Document routing ─────────────────────────────────────────────────


class TestDocumentRouting:

    def test_no_profile_delegates_to_singleton(self, router):
        result = router.route_document({
            "classification": "Share", "score": 5,
            "contains_children": True, "is_appropriate": True,
        })
        assert result == "Share"

    def test_profile_valid_category(self, config):
        profile = TasteProfile(
            name="test", description="test", media_types=["document"],
            categories=[
                CategoryDefinition("Strong", "good"),
                CategoryDefinition("Weak", "bad"),
            ],
            default_category="Weak",
            thresholds={"Strong": 4},
        )
        r = Router(config, profile=profile)
        result = r.route_document({"classification": "Strong", "score": 4})
        assert result == "Strong"

    def test_profile_below_threshold(self, config):
        profile = TasteProfile(
            name="test", description="test", media_types=["document"],
            categories=[
                CategoryDefinition("Strong", "good"),
                CategoryDefinition("Weak", "bad"),
            ],
            default_category="Weak",
            thresholds={"Strong": 4},
        )
        r = Router(config, profile=profile)
        result = r.route_document({"classification": "Strong", "score": 3})
        assert result == "Weak"  # Falls to default

    def test_profile_invalid_category(self, config):
        profile = TasteProfile(
            name="test", description="test", media_types=["document"],
            categories=[
                CategoryDefinition("Strong", "good"),
                CategoryDefinition("Weak", "bad"),
            ],
            default_category="Weak",
        )
        r = Router(config, profile=profile)
        result = r.route_document({"classification": "Bogus", "score": 5})
        assert result == "Weak"

    def test_profile_no_threshold_for_category(self, config):
        profile = TasteProfile(
            name="test", description="test", media_types=["document"],
            categories=[
                CategoryDefinition("A", "a"),
                CategoryDefinition("B", "b"),
            ],
            default_category="B",
            thresholds={},
        )
        r = Router(config, profile=profile)
        result = r.route_document({"classification": "A", "score": 1})
        assert result == "A"  # No threshold means any score accepted
