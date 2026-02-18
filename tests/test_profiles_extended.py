"""Extended tests for profiles.py covering ProfileManager CRUD, defaults, and migration."""
import json
import pytest
from pathlib import Path

from sommelier.core.profiles import (
    TasteProfile,
    CategoryDefinition,
    PhotoProfileSettings,
    DocumentProfileSettings,
    ProfileManager,
)


@pytest.fixture
def tmp_profiles_dir(tmp_path):
    """Create a temporary profiles directory."""
    d = tmp_path / "profiles"
    d.mkdir()
    return d


@pytest.fixture
def pm(tmp_profiles_dir):
    """ProfileManager with temp dir."""
    return ProfileManager(str(tmp_profiles_dir))


# ── TasteProfile dataclass ───────────────────────────────────────────


class TestTasteProfile:

    def test_auto_created_at(self):
        p = TasteProfile(name="t", description="t", media_types=["image"])
        assert p.created_at != ""
        assert p.updated_at == p.created_at

    def test_dict_categories_converted(self):
        p = TasteProfile(
            name="t", description="t", media_types=["image"],
            categories=[{"name": "A", "description": "a"}],
        )
        assert isinstance(p.categories[0], CategoryDefinition)
        assert p.categories[0].name == "A"

    def test_dict_settings_converted(self):
        p = TasteProfile(
            name="t", description="t", media_types=["image"],
            photo_settings={"enable_burst_detection": False},
            document_settings={"extract_text": False, "similarity_threshold": 0.5},
        )
        assert isinstance(p.photo_settings, PhotoProfileSettings)
        assert p.photo_settings.enable_burst_detection is False
        assert isinstance(p.document_settings, DocumentProfileSettings)
        assert p.document_settings.extract_text is False

    def test_category_names(self):
        p = TasteProfile(
            name="t", description="t", media_types=["image"],
            categories=[
                CategoryDefinition("X", "x"),
                CategoryDefinition("Y", "y"),
            ],
        )
        assert p.category_names == ["X", "Y"]

    def test_to_dict_roundtrip(self):
        p = TasteProfile(
            name="roundtrip",
            description="test roundtrip",
            media_types=["document"],
            categories=[CategoryDefinition("A", "a", "#fff")],
            default_category="A",
            top_priorities=["p1"],
            positive_criteria={"must_have": ["x"]},
            negative_criteria={"deal_breakers": ["y"]},
            specific_guidance=["g1"],
            philosophy="test",
            thresholds={"A": 0.5},
            photo_settings=PhotoProfileSettings(),
            document_settings=DocumentProfileSettings(),
        )
        d = p.to_dict()
        p2 = TasteProfile.from_dict(d)
        assert p2.name == "roundtrip"
        assert p2.philosophy == "test"
        assert p2.category_names == ["A"]
        assert p2.thresholds == {"A": 0.5}

    def test_from_dict(self):
        data = {
            "name": "test",
            "description": "desc",
            "media_types": ["image"],
            "categories": [{"name": "A", "description": "a"}],
            "default_category": "A",
        }
        p = TasteProfile.from_dict(data)
        assert p.name == "test"
        assert p.category_names == ["A"]


# ── ProfileManager CRUD ──────────────────────────────────────────────


class TestProfileManagerCRUD:

    def test_create_and_load(self, pm):
        profile = pm.create_profile(
            name="test-profile",
            description="A test profile",
            media_types=["image"],
            categories=[{"name": "Good", "description": "good stuff"}],
        )
        assert profile.name == "test-profile"
        assert pm.profile_exists("test-profile")

        loaded = pm.load_profile("test-profile")
        assert loaded.name == "test-profile"
        assert loaded.description == "A test profile"
        assert loaded.category_names == ["Good"]

    def test_load_nonexistent_raises(self, pm):
        with pytest.raises(FileNotFoundError, match="not found"):
            pm.load_profile("nonexistent")

    def test_list_profiles(self, pm):
        pm.create_profile(
            name="alpha", description="a", media_types=["image"],
            categories=[{"name": "A", "description": "a"}],
        )
        pm.create_profile(
            name="beta", description="b", media_types=["document"],
            categories=[{"name": "B", "description": "b"}],
        )
        profiles = pm.list_profiles()
        names = [p.name for p in profiles]
        assert "alpha" in names
        assert "beta" in names

    def test_list_profiles_empty(self, pm):
        assert pm.list_profiles() == []

    def test_update_profile(self, pm):
        pm.create_profile(
            name="updatable", description="original", media_types=["image"],
            categories=[{"name": "A", "description": "a"}],
        )
        updated = pm.update_profile("updatable", description="updated desc", philosophy="new philosophy")
        assert updated.description == "updated desc"
        assert updated.philosophy == "new philosophy"
        assert updated.version == 2

        # Verify persisted
        loaded = pm.load_profile("updatable")
        assert loaded.description == "updated desc"
        assert loaded.version == 2

    def test_delete_profile(self, pm):
        pm.create_profile(
            name="deleteme", description="d", media_types=["image"],
            categories=[{"name": "A", "description": "a"}],
        )
        assert pm.profile_exists("deleteme")
        result = pm.delete_profile("deleteme")
        assert result is True
        assert not pm.profile_exists("deleteme")

    def test_delete_nonexistent(self, pm):
        result = pm.delete_profile("nope")
        assert result is False

    def test_profile_exists(self, pm):
        assert not pm.profile_exists("nope")
        pm.create_profile(
            name="exists", description="d", media_types=["image"],
            categories=[{"name": "A", "description": "a"}],
        )
        assert pm.profile_exists("exists")

    def test_create_with_all_fields(self, pm):
        profile = pm.create_profile(
            name="full",
            description="full profile",
            media_types=["document"],
            categories=[
                {"name": "Keep", "description": "keep it"},
                {"name": "Toss", "description": "toss it"},
            ],
            default_category="Toss",
            top_priorities=["relevance", "quality"],
            positive_criteria={"must_have": ["clarity"]},
            negative_criteria={"deal_breakers": ["spam"]},
            specific_guidance=["Be strict"],
            philosophy="Quality over quantity",
            thresholds={"Keep": 0.7},
            photo_settings={"enable_burst_detection": False},
            document_settings={"similarity_threshold": 0.9},
        )
        assert profile.default_category == "Toss"
        assert profile.top_priorities == ["relevance", "quality"]
        assert profile.thresholds == {"Keep": 0.7}
        assert profile.photo_settings.enable_burst_detection is False
        assert profile.document_settings.similarity_threshold == 0.9


# ── Default profiles ─────────────────────────────────────────────────


class TestDefaultProfiles:

    def test_get_default_photos(self, pm):
        profile = pm.get_default_profile("image")
        assert profile.name == "default-photos"
        assert "Share" in profile.category_names
        assert "Storage" in profile.category_names

    def test_get_default_documents(self, pm):
        profile = pm.get_default_profile("document")
        assert profile.name == "default-documents"
        assert "Exemplary" in profile.category_names
        assert "Discard" in profile.category_names

    def test_get_default_video_falls_to_photos(self, pm):
        profile = pm.get_default_profile("video")
        assert profile.name == "default-photos"

    def test_get_default_mixed_falls_to_photos(self, pm):
        profile = pm.get_default_profile("mixed")
        assert profile.name == "default-photos"

    def test_get_default_unknown_falls_to_photos(self, pm):
        profile = pm.get_default_profile("unknown_type")
        assert profile.name == "default-photos"

    def test_default_reuses_existing(self, pm):
        """If default already exists, don't recreate."""
        p1 = pm.get_default_profile("image")
        p2 = pm.get_default_profile("image")
        assert p1.name == p2.name
        assert p1.created_at == p2.created_at


# ── Migration ────────────────────────────────────────────────────────


class TestMigration:

    def test_migrate_from_taste_preferences(self, pm, tmp_path):
        prefs = {
            "share_philosophy": "Capture genuine moments",
            "top_priorities": ["Baby face visible", "Good lighting"],
            "share_criteria": {
                "must_have": ["Baby visible"],
                "highly_valued": ["Parent engagement"],
            },
            "reject_criteria": {
                "deal_breakers": ["Blurry photos"],
            },
            "specific_guidance": ["Focus on interactions"],
        }
        prefs_path = tmp_path / "taste_preferences.json"
        prefs_path.write_text(json.dumps(prefs), encoding="utf-8")

        profile = pm.migrate_from_taste_preferences(prefs_path, "migrated")
        assert profile.name == "migrated"
        assert profile.philosophy == "Capture genuine moments"
        assert "Baby face visible" in profile.top_priorities
        assert profile.positive_criteria["must_have"] == ["Baby visible"]
        assert profile.negative_criteria["deal_breakers"] == ["Blurry photos"]
        assert "Share" in profile.category_names

    def test_migrate_minimal_prefs(self, pm, tmp_path):
        prefs = {}
        prefs_path = tmp_path / "minimal.json"
        prefs_path.write_text(json.dumps(prefs), encoding="utf-8")

        profile = pm.migrate_from_taste_preferences(prefs_path, "minimal")
        assert profile.name == "minimal"
        assert profile.philosophy == ""
        assert profile.top_priorities == []

    def test_list_profiles_skips_corrupt(self, pm, tmp_profiles_dir):
        """Corrupt profile JSON should be skipped, not crash list_profiles."""
        (tmp_profiles_dir / "corrupt.json").write_text("not valid json", encoding="utf-8")
        pm.create_profile(
            name="valid", description="v", media_types=["image"],
            categories=[{"name": "A", "description": "a"}],
        )
        profiles = pm.list_profiles()
        names = [p.name for p in profiles]
        assert "valid" in names
        assert "corrupt" not in names
