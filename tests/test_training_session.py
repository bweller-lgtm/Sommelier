"""Tests for TrainingSession, PairwiseComparison, GallerySelection."""

import json
import pytest
from pathlib import Path

from sommelier.training.session import (
    TrainingSession,
    PairwiseComparison,
    GallerySelection,
)


class TestPairwiseComparison:
    def test_create_with_defaults(self):
        comp = PairwiseComparison(
            photo_a="/a.jpg",
            photo_b="/b.jpg",
            choice="left",
            reason="sharper",
            comparison_type="within_burst",
            weight=2.0,
        )
        assert comp.photo_a == "/a.jpg"
        assert comp.choice == "left"
        assert comp.weight == 2.0
        assert comp.timestamp  # auto-set

    def test_timestamp_preserved_if_set(self):
        comp = PairwiseComparison(
            photo_a="/a.jpg",
            photo_b="/b.jpg",
            choice="both",
            reason="",
            comparison_type="between_burst",
            weight=1.0,
            timestamp="2024-01-01T00:00:00",
        )
        assert comp.timestamp == "2024-01-01T00:00:00"


class TestGallerySelection:
    def test_create_with_defaults(self):
        sel = GallerySelection(
            photos=["/a.jpg", "/b.jpg", "/c.jpg"],
            selected_indices=[0, 2],
            reason="best expressions",
        )
        assert len(sel.photos) == 3
        assert sel.selected_indices == [0, 2]
        assert sel.weight == 1.5
        assert sel.timestamp


class TestTrainingSession:
    def test_create(self):
        session = TrainingSession.create(
            profile_name="test-profile",
            folder_path="/photos",
            bursts=[["/a.jpg", "/b.jpg"], ["/c.jpg", "/d.jpg", "/e.jpg"]],
            singletons=["/f.jpg"],
        )
        assert session.session_id
        assert len(session.session_id) == 12
        assert session.profile_name == "test-profile"
        assert session.total_photos == 6
        assert session.status == "active"
        assert session.created_at
        assert session.pairwise == []
        assert session.gallery == []

    def test_add_pairwise(self):
        session = TrainingSession.create(
            profile_name="test",
            folder_path="/photos",
            bursts=[["/a.jpg", "/b.jpg"]],
            singletons=[],
        )
        comp = session.add_pairwise(
            photo_a="/a.jpg",
            photo_b="/b.jpg",
            choice="left",
            reason="better focus",
            comparison_type="within_burst",
        )
        assert comp.weight == 2.0
        assert len(session.pairwise) == 1

    def test_add_pairwise_between_weight(self):
        session = TrainingSession.create(
            profile_name="test",
            folder_path="/photos",
            bursts=[],
            singletons=["/a.jpg", "/b.jpg"],
        )
        comp = session.add_pairwise(
            photo_a="/a.jpg",
            photo_b="/b.jpg",
            choice="right",
            reason="more engaging",
            comparison_type="between_burst",
        )
        assert comp.weight == 1.0

    def test_add_gallery(self):
        session = TrainingSession.create(
            profile_name="test",
            folder_path="/photos",
            bursts=[["/a.jpg", "/b.jpg", "/c.jpg", "/d.jpg", "/e.jpg"]],
            singletons=[],
        )
        sel = session.add_gallery(
            photos=["/a.jpg", "/b.jpg", "/c.jpg", "/d.jpg", "/e.jpg"],
            selected_indices=[0, 2, 4],
            reason="best expressions and focus",
        )
        assert sel.weight == 1.5
        assert len(session.gallery) == 1

    def test_get_labeled_photos(self):
        session = TrainingSession.create(
            profile_name="test",
            folder_path="/photos",
            bursts=[["/a.jpg", "/b.jpg"]],
            singletons=["/c.jpg", "/d.jpg"],
        )
        session.add_pairwise("/a.jpg", "/b.jpg", "left", "sharper", "within_burst")
        session.add_gallery(["/c.jpg", "/d.jpg"], [0], "c is better")

        labeled = session.get_labeled_photos()
        assert labeled == {"/a.jpg", "/b.jpg", "/c.jpg", "/d.jpg"}

    def test_get_stats(self):
        session = TrainingSession.create(
            profile_name="test",
            folder_path="/photos",
            bursts=[["/a.jpg", "/b.jpg"], ["/c.jpg", "/d.jpg"]],
            singletons=["/e.jpg"],
        )
        session.add_pairwise("/a.jpg", "/b.jpg", "left", "sharper", "within_burst")
        session.add_pairwise("/c.jpg", "/e.jpg", "both", "both good", "between_burst")

        stats = session.get_stats()
        assert stats["pairwise_count"] == 2
        assert stats["within_burst"] == 1
        assert stats["between_burst"] == 1
        assert stats["choices"]["left"] == 1
        assert stats["choices"]["both"] == 1
        assert stats["total_labeled"] == 2
        assert stats["total_photos"] == 5
        assert not stats["ready_to_synthesize"]

    def test_ready_to_synthesize_at_15(self):
        session = TrainingSession.create(
            profile_name="test",
            folder_path="/photos",
            bursts=[],
            singletons=[f"/{i}.jpg" for i in range(30)],
        )
        for i in range(15):
            session.add_pairwise(
                f"/{i*2}.jpg", f"/{i*2+1}.jpg", "left", "test", "between_burst"
            )
        stats = session.get_stats()
        assert stats["ready_to_synthesize"]

    def test_save_and_load(self, tmp_path):
        session = TrainingSession.create(
            profile_name="test-profile",
            folder_path="/photos",
            bursts=[["/a.jpg", "/b.jpg"]],
            singletons=["/c.jpg"],
        )
        session.add_pairwise("/a.jpg", "/b.jpg", "left", "sharper", "within_burst")
        session.add_gallery(["/c.jpg"], [0], "good")

        session.save(tmp_path)

        loaded = TrainingSession.load(session.session_id, tmp_path)
        assert loaded.session_id == session.session_id
        assert loaded.profile_name == "test-profile"
        assert len(loaded.pairwise) == 1
        assert loaded.pairwise[0].choice == "left"
        assert loaded.pairwise[0].reason == "sharper"
        assert loaded.pairwise[0].weight == 2.0
        assert len(loaded.gallery) == 1
        assert loaded.gallery[0].selected_indices == [0]

    def test_load_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            TrainingSession.load("nonexistent", tmp_path)

    def test_list_sessions(self, tmp_path):
        s1 = TrainingSession.create("p1", "/photos", [], ["/a.jpg"])
        s1.save(tmp_path)

        s2 = TrainingSession.create("p2", "/photos", [], ["/b.jpg"])
        s2.add_pairwise("/a.jpg", "/b.jpg", "left", "test", "between_burst")
        s2.save(tmp_path)

        sessions = TrainingSession.list_sessions(tmp_path)
        assert len(sessions) == 2
        ids = {s["session_id"] for s in sessions}
        assert s1.session_id in ids
        assert s2.session_id in ids

    def test_list_sessions_empty(self, tmp_path):
        sessions = TrainingSession.list_sessions(tmp_path)
        assert sessions == []

    def test_round_trip_preserves_types(self, tmp_path):
        """Ensure all fields survive serialization round-trip."""
        session = TrainingSession.create(
            profile_name="round-trip",
            folder_path="/photos",
            bursts=[["/a.jpg", "/b.jpg", "/c.jpg"]],
            singletons=["/d.jpg"],
        )
        session.add_pairwise("/a.jpg", "/b.jpg", "neither", "both blurry", "within_burst")
        session.add_gallery(
            ["/a.jpg", "/b.jpg", "/c.jpg"], [1], "only b is sharp"
        )
        session.current_comparison = {"type": "pairwise_within", "photo_a": "/a.jpg", "photo_b": "/c.jpg"}
        session.comparisons_served = 5

        session.save(tmp_path)
        loaded = TrainingSession.load(session.session_id, tmp_path)

        assert loaded.total_photos == 4
        assert loaded.comparisons_served == 5
        assert loaded.current_comparison["type"] == "pairwise_within"
        assert loaded.pairwise[0].choice == "neither"
        assert loaded.gallery[0].selected_indices == [1]
