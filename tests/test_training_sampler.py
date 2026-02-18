"""Tests for ComparisonSampler."""

import pytest

from taster.training.session import TrainingSession
from taster.training.sampler import ComparisonSampler


def _make_session(
    bursts=None, singletons=None, pairwise_labels=None
) -> TrainingSession:
    """Helper to create a session with given photo layout."""
    if bursts is None:
        bursts = []
    if singletons is None:
        singletons = []
    session = TrainingSession.create(
        profile_name="test",
        folder_path="/photos",
        bursts=bursts,
        singletons=singletons,
    )
    if pairwise_labels:
        for label in pairwise_labels:
            session.add_pairwise(*label)
    return session


class TestComparisonSampler:
    def test_empty_session_returns_none(self):
        session = _make_session(bursts=[], singletons=[])
        sampler = ComparisonSampler(session)
        assert sampler.get_next() is None

    def test_single_photo_returns_none(self):
        session = _make_session(singletons=["/a.jpg"])
        sampler = ComparisonSampler(session)
        assert sampler.get_next() is None

    def test_two_singletons_returns_between(self):
        session = _make_session(singletons=["/a.jpg", "/b.jpg"])
        sampler = ComparisonSampler(session)
        result = sampler.get_next()
        assert result is not None
        assert result["type"] == "pairwise_between"
        assert result["comparison_type"] == "between_burst"
        assert set([result["photo_a"], result["photo_b"]]) == {"/a.jpg", "/b.jpg"}

    def test_small_burst_returns_within(self):
        session = _make_session(
            bursts=[["/a.jpg", "/b.jpg", "/c.jpg"]],
            singletons=["/d.jpg", "/e.jpg"],
        )
        sampler = ComparisonSampler(session, seed=42)

        # Run many times to check we get within-burst pairs
        within_count = 0
        for _ in range(50):
            result = sampler.get_next()
            if result and result["type"] == "pairwise_within":
                within_count += 1
                assert result["photo_a"] in ["/a.jpg", "/b.jpg", "/c.jpg"]
                assert result["photo_b"] in ["/a.jpg", "/b.jpg", "/c.jpg"]
                assert result["photo_a"] != result["photo_b"]

        # Should get a mix, with within being dominant (~50%)
        assert within_count > 10  # at least 20% should be within

    def test_large_burst_generates_gallery(self):
        large_burst = [f"/{i}.jpg" for i in range(10)]
        session = _make_session(
            bursts=[large_burst],
            singletons=["/s1.jpg", "/s2.jpg"],
        )
        sampler = ComparisonSampler(session, seed=42)

        # Run many times to check gallery appears
        gallery_count = 0
        for _ in range(100):
            result = sampler.get_next()
            if result and result["type"] == "gallery":
                gallery_count += 1
                assert len(result["photos"]) <= 20
                assert all(p in large_burst for p in result["photos"])

        assert gallery_count > 0  # gallery should appear

    def test_no_bursts_all_singletons(self):
        singletons = [f"/{i}.jpg" for i in range(10)]
        session = _make_session(singletons=singletons)
        sampler = ComparisonSampler(session, seed=42)

        result = sampler.get_next()
        assert result is not None
        # Should be between-burst since no bursts exist
        assert result["type"] == "pairwise_between"
        assert result["photo_a"] in singletons
        assert result["photo_b"] in singletons

    def test_prefers_unlabeled_photos(self):
        burst = ["/a.jpg", "/b.jpg", "/c.jpg", "/d.jpg"]
        session = _make_session(
            bursts=[burst],
            singletons=["/e.jpg", "/f.jpg"],
        )
        # Label a and b
        session.add_pairwise("/a.jpg", "/b.jpg", "left", "test", "within_burst")

        sampler = ComparisonSampler(session, seed=42)

        # Run many times, check unlabeled photos appear more often
        unlabeled_appearances = 0
        labeled_appearances = 0
        for _ in range(50):
            result = sampler.get_next()
            if result and result["type"] in ("pairwise_within", "pairwise_between"):
                for p in [result["photo_a"], result["photo_b"]]:
                    if p in ("/a.jpg", "/b.jpg"):
                        labeled_appearances += 1
                    elif p in ("/c.jpg", "/d.jpg", "/e.jpg", "/f.jpg"):
                        unlabeled_appearances += 1

        # Unlabeled should appear more
        assert unlabeled_appearances > labeled_appearances

    def test_all_labeled_exhausted(self):
        """When all photos in a small session are labeled, returns None."""
        session = _make_session(
            bursts=[["/a.jpg", "/b.jpg"]],
            singletons=[],
        )
        session.add_pairwise("/a.jpg", "/b.jpg", "left", "test", "within_burst")

        sampler = ComparisonSampler(session, seed=42)
        result = sampler.get_next()
        assert result is None

    def test_partially_labeled_still_works(self):
        """With enough unlabeled photos remaining, comparisons continue."""
        session = _make_session(
            bursts=[["/a.jpg", "/b.jpg", "/c.jpg", "/d.jpg"]],
            singletons=["/e.jpg", "/f.jpg"],
        )
        session.add_pairwise("/a.jpg", "/b.jpg", "left", "test", "within_burst")

        sampler = ComparisonSampler(session, seed=42)
        result = sampler.get_next()
        assert result is not None

    def test_comparison_types_are_valid(self):
        session = _make_session(
            bursts=[[f"/{i}.jpg" for i in range(8)]],
            singletons=[f"/s{i}.jpg" for i in range(5)],
        )
        sampler = ComparisonSampler(session, seed=42)

        for _ in range(20):
            result = sampler.get_next()
            assert result is not None
            assert result["type"] in ("pairwise_within", "pairwise_between", "gallery")
            if result["type"] in ("pairwise_within", "pairwise_between"):
                assert "photo_a" in result
                assert "photo_b" in result
                assert "comparison_type" in result
            elif result["type"] == "gallery":
                assert "photos" in result
                assert len(result["photos"]) >= 1

    def test_context_message_included(self):
        session = _make_session(
            bursts=[["/a.jpg", "/b.jpg"]],
            singletons=["/c.jpg", "/d.jpg"],
        )
        sampler = ComparisonSampler(session, seed=42)

        for _ in range(10):
            result = sampler.get_next()
            assert result is not None
            assert "context" in result
            assert isinstance(result["context"], str)
            assert len(result["context"]) > 0

    def test_ratio_distribution_approximate(self):
        """With many sources, ratios should be approximately correct."""
        bursts = [[f"/b{i}_{j}.jpg" for j in range(3)] for i in range(10)]
        large_bursts = [[f"/lb{i}_{j}.jpg" for j in range(8)] for i in range(5)]
        singletons = [f"/s{i}.jpg" for i in range(20)]

        session = _make_session(
            bursts=bursts + large_bursts,
            singletons=singletons,
        )
        sampler = ComparisonSampler(session, seed=42)

        counts = {"pairwise_within": 0, "pairwise_between": 0, "gallery": 0}
        n = 200
        for _ in range(n):
            result = sampler.get_next()
            if result:
                counts[result["type"]] += 1

        total = sum(counts.values())
        assert total > 0

        # Check ratios are in the right ballpark (within 20% of target)
        within_pct = counts["pairwise_within"] / total
        between_pct = counts["pairwise_between"] / total
        gallery_pct = counts["gallery"] / total

        assert 0.25 < within_pct < 0.75, f"within={within_pct:.2f}"
        assert 0.10 < between_pct < 0.55, f"between={between_pct:.2f}"
        assert gallery_pct > 0.01, f"gallery={gallery_pct:.2f}"
