"""Tests for ProfileSynthesizer."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from taster.core.config import TrainingConfig
from taster.core.profiles import ProfileManager
from taster.training.session import TrainingSession
from taster.training.synthesizer import ProfileSynthesizer


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

    def test_refine_with_dimension_scores(self, mock_client, pm):
        """Corrections with dimension scores should be included in the prompt."""
        pm.create_profile(
            name="dim-refine",
            description="Test",
            media_types=["image"],
            categories=[{"name": "Share", "description": "Share"}],
            top_priorities=["Quality"],
        )

        mock_client.generate_json.return_value = {
            "top_priorities": ["Expression", "Quality"],
            "specific_guidance": ["Pay more attention to expressions"],
            "changes_made": ["Adjusted expression scoring"],
            "dimension_adjustments": ["composition dimension was too lenient"],
        }

        synth = ProfileSynthesizer(mock_client, pm)
        updated = synth.refine_from_corrections("dim-refine", [
            {
                "file_path": "/photo.jpg",
                "original_category": "Storage",
                "correct_category": "Share",
                "reason": "Great expressions",
                "dimensions": {"composition": 3, "expression": 2},
            },
        ])

        # Verify the prompt sent to AI included dimension scores
        call_args = mock_client.generate_json.call_args
        prompt_text = call_args[1]["prompt"]
        assert "composition=3" in prompt_text
        assert "expression=2" in prompt_text
        assert "dimension" in prompt_text.lower()

        dim_adj = getattr(updated, "_dimension_adjustments", [])
        assert len(dim_adj) > 0

    def test_format_correction_dimensions(self):
        assert ProfileSynthesizer._format_correction_dimensions({}) == ""
        assert ProfileSynthesizer._format_correction_dimensions(None) == ""
        result = ProfileSynthesizer._format_correction_dimensions({"a": 3, "b": None, "c": 5})
        assert "a=3" in result
        assert "c=5" in result
        assert "b=" not in result  # None values excluded


class TestBalanceExamples:
    """Tests for _balance_examples()."""

    def test_no_balancing_when_within_ratio(self, mock_client, pm):
        """3 share + 9 storage (exactly 3:1) → unchanged."""
        synth = ProfileSynthesizer(mock_client, pm)
        share = {f"/{i}.jpg": 0.9 for i in range(3)}
        storage = {f"/s{i}.jpg": 0.8 for i in range(9)}
        s_reasons = ["good"] * 3
        t_reasons = ["bad"] * 9

        s, t, sr, tr = synth._balance_examples(share, storage, s_reasons, t_reasons)
        assert len(s) == 3
        assert len(t) == 9
        assert len(sr) == 3
        assert len(tr) == 9

    def test_undersample_reduces_storage(self, mock_client, pm):
        """3 share + 30 storage → 9 storage (3:1 with default max_negative_per_positive=3)."""
        synth = ProfileSynthesizer(mock_client, pm)
        share = {f"/{i}.jpg": 0.9 for i in range(3)}
        storage = {f"/s{i}.jpg": 0.5 + i * 0.01 for i in range(30)}

        s, t, sr, tr = synth._balance_examples(share, storage, [], [])
        assert len(s) == 3
        assert len(t) == 9  # 3 * 3

    def test_keeps_highest_confidence(self, mock_client, pm):
        """When undersampling, top-confidence items are kept."""
        synth = ProfileSynthesizer(mock_client, pm)
        share = {"/a.jpg": 0.9}
        # 10 storage items with distinct confidences
        storage = {f"/s{i}.jpg": 0.1 * (i + 1) for i in range(10)}

        s, t, sr, tr = synth._balance_examples(share, storage, [], [])
        assert len(t) == 3  # 1 * 3
        # Top-3 confidences: 1.0, 0.9, 0.8
        assert "/s9.jpg" in t  # 1.0
        assert "/s8.jpg" in t  # 0.9
        assert "/s7.jpg" in t  # 0.8

    def test_reasons_capped_proportionally(self, mock_client, pm):
        """Storage reasons scale with storage count reduction."""
        synth = ProfileSynthesizer(mock_client, pm)
        share = {"/a.jpg": 0.9}
        storage = {f"/s{i}.jpg": 0.5 for i in range(30)}
        t_reasons = [f"reason_{i}" for i in range(30)]

        s, t, sr, tr = synth._balance_examples(share, storage, [], t_reasons)
        assert len(t) == 3
        # 30 reasons → ~3 (3/30 * 30), at least 1
        assert 1 <= len(tr) <= 5

    def test_empty_share_skips_balancing(self, mock_client, pm):
        """0 share + 10 storage → unchanged (nothing to ratio against)."""
        synth = ProfileSynthesizer(mock_client, pm)
        share: dict = {}
        storage = {f"/s{i}.jpg": 0.8 for i in range(10)}

        s, t, sr, tr = synth._balance_examples(share, storage, [], ["r"] * 10)
        assert len(t) == 10
        assert len(tr) == 10

    def test_default_config(self, mock_client, pm):
        """No config passed → defaults apply (max_negative_per_positive=3)."""
        synth = ProfileSynthesizer(mock_client, pm)
        assert synth.training_config.max_negative_per_positive == 3

    def test_custom_config_ratio(self, mock_client, pm):
        """Custom config with ratio=5 allows more storage."""
        tc = TrainingConfig(max_negative_per_positive=5)
        synth = ProfileSynthesizer(mock_client, pm, training_config=tc)
        share = {f"/{i}.jpg": 0.9 for i in range(2)}
        storage = {f"/s{i}.jpg": 0.5 for i in range(20)}

        s, t, sr, tr = synth._balance_examples(share, storage, [], [])
        assert len(t) == 10  # 2 * 5
