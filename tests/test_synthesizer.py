"""Tests for ProfileSynthesizer."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.core.profiles import ProfileManager
from src.training.session import TrainingSession
from src.training.synthesizer import ProfileSynthesizer


@pytest.fixture
def pm(tmp_path):
    d = tmp_path / "profiles"
    d.mkdir()
    return ProfileManager(d)


@pytest.fixture
def mock_client():
    client = MagicMock()
    return client


@pytest.fixture
def session():
    s = TrainingSession.create(
        profile_name="synth-test",
        folder_path="/photos",
        bursts=[["/a.jpg", "/b.jpg", "/c.jpg"]],
        singletons=["/d.jpg", "/e.jpg"],
    )
    s.add_pairwise("/a.jpg", "/b.jpg", "left", "a is sharper", "within_burst")
    s.add_pairwise("/c.jpg", "/d.jpg", "right", "d has better expression", "between_burst")
    s.add_pairwise("/a.jpg", "/e.jpg", "both", "both are good", "between_burst")
    s.add_pairwise("/b.jpg", "/c.jpg", "neither", "both blurry", "within_burst")
    s.add_gallery(
        ["/a.jpg", "/b.jpg", "/c.jpg"],
        [0, 2],
        "a and c are the keepers",
    )
    return s


class TestLabelConversion:
    def test_left_choice_mapping(self, mock_client, pm):
        synth = ProfileSynthesizer(mock_client, pm)
        session = TrainingSession.create("t", "/p", [], ["/a.jpg", "/b.jpg"])
        session.add_pairwise("/a.jpg", "/b.jpg", "left", "a is better", "within_burst")

        share, storage, share_r, storage_r = synth._convert_labels(session)
        assert "/a.jpg" in share
        assert share["/a.jpg"] == 0.95
        assert "/b.jpg" in storage
        assert storage["/b.jpg"] == 0.85
        assert "a is better" in share_r

    def test_right_choice_mapping(self, mock_client, pm):
        synth = ProfileSynthesizer(mock_client, pm)
        session = TrainingSession.create("t", "/p", [], ["/a.jpg", "/b.jpg"])
        session.add_pairwise("/a.jpg", "/b.jpg", "right", "b is better", "within_burst")

        share, storage, share_r, storage_r = synth._convert_labels(session)
        assert "/b.jpg" in share
        assert share["/b.jpg"] == 0.95
        assert "/a.jpg" in storage

    def test_both_choice_mapping(self, mock_client, pm):
        synth = ProfileSynthesizer(mock_client, pm)
        session = TrainingSession.create("t", "/p", [], ["/a.jpg", "/b.jpg"])
        session.add_pairwise("/a.jpg", "/b.jpg", "both", "both great", "between_burst")

        share, storage, _, _ = synth._convert_labels(session)
        assert "/a.jpg" in share
        assert "/b.jpg" in share
        assert share["/a.jpg"] == 0.9
        assert share["/b.jpg"] == 0.9

    def test_neither_choice_mapping(self, mock_client, pm):
        synth = ProfileSynthesizer(mock_client, pm)
        session = TrainingSession.create("t", "/p", [], ["/a.jpg", "/b.jpg"])
        session.add_pairwise("/a.jpg", "/b.jpg", "neither", "both bad", "within_burst")

        share, storage, share_r, storage_r = synth._convert_labels(session)
        assert "/a.jpg" in storage
        assert "/b.jpg" in storage
        assert storage["/a.jpg"] == 0.9
        assert "both bad" in storage_r

    def test_gallery_mapping(self, mock_client, pm):
        synth = ProfileSynthesizer(mock_client, pm)
        session = TrainingSession.create("t", "/p", [], [])
        session.add_gallery(
            ["/a.jpg", "/b.jpg", "/c.jpg"],
            [0, 2],
            "a and c are best",
        )

        share, storage, share_r, _ = synth._convert_labels(session)
        assert "/a.jpg" in share  # selected
        assert "/c.jpg" in share  # selected
        assert "/b.jpg" in storage  # not selected

    def test_conflict_resolution(self, mock_client, pm):
        synth = ProfileSynthesizer(mock_client, pm)
        session = TrainingSession.create("t", "/p", [], ["/a.jpg", "/b.jpg", "/c.jpg"])
        # First: a=share(0.95), b=storage(0.85)
        session.add_pairwise("/a.jpg", "/b.jpg", "left", "", "within_burst")
        # Second: b=share(0.95), c=storage(0.85)
        session.add_pairwise("/b.jpg", "/c.jpg", "left", "", "within_burst")

        share, storage, _, _ = synth._convert_labels(session)
        # b appears as both share(0.95) and storage(0.85)
        # Share confidence (0.95) >= Storage (0.85), so b should be in share
        assert "/b.jpg" in share
        assert "/b.jpg" not in storage

    def test_reasons_collected_correctly(self, mock_client, pm, session):
        synth = ProfileSynthesizer(mock_client, pm)
        share, storage, share_r, storage_r = synth._convert_labels(session)

        # left/right/both reasons go to share_reasons
        assert any("sharper" in r for r in share_r)
        assert any("better expression" in r for r in share_r)
        assert any("both are good" in r for r in share_r)
        # neither reasons go to storage_reasons
        assert any("both blurry" in r for r in storage_r)


class TestAnalyzeReasoning:
    def test_empty_reasons_returns_none(self, mock_client, pm):
        synth = ProfileSynthesizer(mock_client, pm)
        result = synth._analyze_reasoning([], [])
        assert result is None

    def test_calls_ai_with_correct_prompt(self, mock_client, pm):
        mock_client.generate_json.return_value = {
            "valued_qualities": ["sharpness"],
            "reject_criteria": ["blur"],
        }
        synth = ProfileSynthesizer(mock_client, pm)

        result = synth._analyze_reasoning(
            ["sharper image", "better expressions"],
            ["too blurry", "eyes closed"],
        )

        mock_client.generate_json.assert_called_once()
        prompt = mock_client.generate_json.call_args[1]["prompt"]
        assert "sharper image" in prompt
        assert "too blurry" in prompt
        assert result is not None
        assert "valued_qualities" in result


class TestSynthesize:
    def test_synthesize_creates_profile(self, mock_client, pm, session):
        # Mock all AI calls
        mock_client.generate_json.side_effect = [
            # reasoning analysis
            {"valued_qualities": ["sharpness"], "reject_criteria": ["blur"]},
            # profile synthesis
            {
                "description": "Family photo sorter",
                "media_types": ["image"],
                "categories": [
                    {"name": "Share", "description": "Worth sharing"},
                    {"name": "Storage", "description": "Archive"},
                ],
                "default_category": "Storage",
                "top_priorities": ["Expression quality"],
                "positive_criteria": {"must_have": ["Clear faces"]},
                "negative_criteria": {"deal_breakers": ["Blurry"]},
                "specific_guidance": ["Check expressions"],
                "philosophy": "Share the best family moments",
            },
        ]
        mock_response = MagicMock()
        mock_response.parse_json.return_value = None
        mock_client.generate.return_value = mock_response

        synth = ProfileSynthesizer(mock_client, pm)
        profile = synth.synthesize(session, "test-synth")

        assert profile.name == "test-synth"
        assert any(c.name.lower() == "share" for c in profile.categories)

    def test_synthesize_refines_existing(self, mock_client, pm, session):
        # Create existing profile
        existing = pm.create_profile(
            name="existing",
            description="Old profile",
            media_types=["image"],
            categories=[
                {"name": "Share", "description": "Share"},
                {"name": "Storage", "description": "Storage"},
            ],
        )

        # Only one generate_json call happens (_synthesize_profile) since
        # _analyze_reasoning is skipped (no reasons) and visual analysis
        # is skipped (photos don't exist on disk).
        mock_client.generate_json.return_value = {
            "description": "Updated profile",
            "top_priorities": ["Updated priority"],
            "positive_criteria": {"must_have": ["New criteria"]},
            "negative_criteria": {"deal_breakers": ["New breaker"]},
            "specific_guidance": ["New guidance"],
            "philosophy": "Updated philosophy",
        }
        mock_response = MagicMock()
        mock_response.parse_json.return_value = None
        mock_client.generate.return_value = mock_response

        synth = ProfileSynthesizer(mock_client, pm)
        # Use session with no reasons (empty string)
        empty_session = TrainingSession.create("t", "/p", [], ["/a.jpg", "/b.jpg"])
        empty_session.add_pairwise("/a.jpg", "/b.jpg", "left", "", "between_burst")

        profile = synth.synthesize(empty_session, "existing", existing)
        assert profile.name == "existing"


class TestRefineFromCorrections:
    def test_refine_updates_profile(self, mock_client, pm):
        pm.create_profile(
            name="to-refine",
            description="Test",
            media_types=["image"],
            categories=[{"name": "Share", "description": "Share"}],
            top_priorities=["Quality"],
            positive_criteria={"must_have": ["Sharp"]},
            negative_criteria={"deal_breakers": ["Blurry"]},
        )

        mock_client.generate_json.return_value = {
            "top_priorities": ["Expression", "Quality"],
            "positive_criteria": {"must_have": ["Sharp", "Good expressions"]},
            "negative_criteria": {"deal_breakers": ["Blurry", "Eyes closed"]},
            "specific_guidance": ["Check eye contact"],
            "changes_made": ["Added expression criteria"],
        }

        synth = ProfileSynthesizer(mock_client, pm)
        updated = synth.refine_from_corrections("to-refine", [
            {
                "file_path": "/photo.jpg",
                "original_category": "Storage",
                "correct_category": "Share",
                "reason": "Great expressions",
            },
        ])

        assert "Expression" in updated.top_priorities
        changes = getattr(updated, "_refinement_changes", [])
        assert len(changes) > 0

    def test_refine_ai_failure_raises(self, mock_client, pm):
        pm.create_profile(
            name="fail-refine",
            description="Test",
            media_types=["image"],
            categories=[{"name": "A", "description": "A"}],
        )

        mock_client.generate_json.return_value = None

        synth = ProfileSynthesizer(mock_client, pm)
        with pytest.raises(RuntimeError, match="AI failed"):
            synth.refine_from_corrections("fail-refine", [
                {"file_path": "/a.jpg", "original_category": "A", "correct_category": "B"},
            ])
