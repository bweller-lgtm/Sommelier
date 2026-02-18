"""Profile management system for taste classification.

Supports multiple named taste profiles with user-defined output categories,
enabling classification of any media type (photos, videos, documents, mixed).
"""
import json
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime


@dataclass
class CategoryDefinition:
    """Definition of an output category."""
    name: str                          # e.g., "Exemplary"
    description: str                   # e.g., "Outstanding examples worth showcasing"
    color: Optional[str] = None        # For UI display, e.g., "#4CAF50"


@dataclass
class PhotoProfileSettings:
    """Photo/video-specific profile settings."""
    enable_burst_detection: bool = True
    enable_face_detection: bool = True
    contains_children_check: bool = True
    appropriateness_check: bool = True


@dataclass
class DocumentProfileSettings:
    """Document-specific profile settings."""
    extract_text: bool = True
    extract_metadata: bool = True
    enable_similarity_grouping: bool = True
    similarity_threshold: float = 0.85
    max_pages_to_analyze: int = 10


@dataclass
class TasteProfile:
    """A named taste profile defining classification preferences."""
    name: str
    description: str
    media_types: List[str]             # ["image", "video", "document", "mixed"]

    # User-defined output categories
    categories: List[CategoryDefinition] = field(default_factory=list)
    default_category: str = "Review"

    # Taste preferences
    top_priorities: List[str] = field(default_factory=list)
    positive_criteria: Dict[str, List[str]] = field(default_factory=dict)
    negative_criteria: Dict[str, List[str]] = field(default_factory=dict)
    specific_guidance: List[str] = field(default_factory=list)
    philosophy: str = ""

    # Thresholds mapping category names to confidence ranges
    thresholds: Dict[str, float] = field(default_factory=dict)

    # Optional media-type-specific settings
    photo_settings: Optional[PhotoProfileSettings] = None
    document_settings: Optional[DocumentProfileSettings] = None

    # Metadata
    created_at: str = ""
    updated_at: str = ""
    version: int = 1

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.updated_at:
            self.updated_at = self.created_at
        # Convert dict categories to CategoryDefinition objects
        if self.categories and isinstance(self.categories[0], dict):
            self.categories = [
                CategoryDefinition(**c) for c in self.categories
            ]
        # Convert dict settings to dataclass objects
        if isinstance(self.photo_settings, dict):
            self.photo_settings = PhotoProfileSettings(**self.photo_settings)
        if isinstance(self.document_settings, dict):
            self.document_settings = DocumentProfileSettings(**self.document_settings)

    @property
    def category_names(self) -> List[str]:
        """Get list of category names."""
        return [c.name for c in self.categories]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = {}
        data["name"] = self.name
        data["description"] = self.description
        data["media_types"] = self.media_types
        data["categories"] = [asdict(c) for c in self.categories]
        data["default_category"] = self.default_category
        data["top_priorities"] = self.top_priorities
        data["positive_criteria"] = self.positive_criteria
        data["negative_criteria"] = self.negative_criteria
        data["specific_guidance"] = self.specific_guidance
        data["philosophy"] = self.philosophy
        data["thresholds"] = self.thresholds
        data["photo_settings"] = asdict(self.photo_settings) if self.photo_settings else None
        data["document_settings"] = asdict(self.document_settings) if self.document_settings else None
        data["created_at"] = self.created_at
        data["updated_at"] = self.updated_at
        data["version"] = self.version
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TasteProfile":
        """Create TasteProfile from dictionary."""
        return cls(**data)


class ProfileManager:
    """Manages taste profiles stored as JSON files."""

    def __init__(self, profiles_dir: str = "profiles"):
        self.profiles_dir = Path(profiles_dir)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)

    def create_profile(
        self,
        name: str,
        description: str,
        media_types: List[str],
        categories: List[Dict[str, str]],
        default_category: str = "Review",
        top_priorities: Optional[List[str]] = None,
        positive_criteria: Optional[Dict[str, List[str]]] = None,
        negative_criteria: Optional[Dict[str, List[str]]] = None,
        specific_guidance: Optional[List[str]] = None,
        philosophy: str = "",
        thresholds: Optional[Dict[str, float]] = None,
        photo_settings: Optional[Dict] = None,
        document_settings: Optional[Dict] = None,
    ) -> TasteProfile:
        """Create and save a new taste profile."""
        profile = TasteProfile(
            name=name,
            description=description,
            media_types=media_types,
            categories=[CategoryDefinition(**c) for c in categories],
            default_category=default_category,
            top_priorities=top_priorities or [],
            positive_criteria=positive_criteria or {},
            negative_criteria=negative_criteria or {},
            specific_guidance=specific_guidance or [],
            philosophy=philosophy,
            thresholds=thresholds or {},
            photo_settings=PhotoProfileSettings(**photo_settings) if photo_settings else None,
            document_settings=DocumentProfileSettings(**document_settings) if document_settings else None,
        )
        self._save_profile(profile)
        return profile

    def load_profile(self, name: str) -> TasteProfile:
        """Load a profile by name."""
        path = self.profiles_dir / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(f"Profile not found: {name} (looked at {path})")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return TasteProfile.from_dict(data)

    def list_profiles(self) -> List[TasteProfile]:
        """List all available profiles."""
        profiles = []
        for path in sorted(self.profiles_dir.glob("*.json")):
            try:
                profiles.append(self.load_profile(path.stem))
            except Exception as e:
                print(f"Warning: Failed to load profile {path.name}: {e}")
        return profiles

    def update_profile(self, name: str, **kwargs) -> TasteProfile:
        """Update an existing profile with new values."""
        profile = self.load_profile(name)

        for key, value in kwargs.items():
            if hasattr(profile, key):
                setattr(profile, key, value)

        profile.updated_at = datetime.now().isoformat()
        profile.version += 1
        self._save_profile(profile)
        return profile

    def delete_profile(self, name: str) -> bool:
        """Delete a profile by name."""
        path = self.profiles_dir / f"{name}.json"
        if path.exists():
            path.unlink()
            return True
        return False

    def profile_exists(self, name: str) -> bool:
        """Check if a profile exists."""
        return (self.profiles_dir / f"{name}.json").exists()

    def get_default_profile(self, media_type: str = "image") -> TasteProfile:
        """Get the default profile for a given media type."""
        defaults = {
            "image": "default-photos",
            "video": "default-photos",
            "document": "default-documents",
            "mixed": "default-photos",
        }
        profile_name = defaults.get(media_type, "default-photos")

        if self.profile_exists(profile_name):
            return self.load_profile(profile_name)

        # Create default if it doesn't exist
        return self._create_default_profile(profile_name, media_type)

    def _save_profile(self, profile: TasteProfile):
        """Save a profile to disk."""
        path = self.profiles_dir / f"{profile.name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(profile.to_dict(), f, indent=2, ensure_ascii=False)

    def _create_default_profile(self, name: str, media_type: str) -> TasteProfile:
        """Create a default profile for the given media type."""
        if name == "default-photos":
            return self._create_default_photos_profile()
        elif name == "default-documents":
            return self._create_default_documents_profile()
        else:
            return self._create_default_photos_profile()

    def _create_default_photos_profile(self) -> TasteProfile:
        """Create the default family photos profile."""
        profile = TasteProfile(
            name="default-photos",
            description="Family photo and video classification - sorts by share-worthiness",
            media_types=["image", "video"],
            categories=[
                CategoryDefinition("Share", "Photos worth sharing with family", "#4CAF50"),
                CategoryDefinition("Storage", "Keep but don't share", "#2196F3"),
                CategoryDefinition("Review", "Uncertain, needs manual review", "#FF9800"),
                CategoryDefinition("Ignore", "No children or inappropriate", "#9E9E9E"),
            ],
            default_category="Review",
            top_priorities=[
                "Parent-child interaction quality",
                "Baby's expression (mischief, joy, engagement)",
                "Parent expressions (should be engaged, not vacant)",
                "Baby's face clearly visible",
                "Genuine emotional moments",
            ],
            positive_criteria={
                "must_have": [
                    "Baby's face clearly visible",
                    "Good technical quality (sharp, well-lit)",
                ],
                "highly_valued": [
                    "Parent and baby both have good expressions",
                    "Subjects engaged with each other (eye contact, shared activity)",
                    "Emotional moments: mischief, joy, intimacy, playfulness",
                    "Natural interactions (not forced/posed)",
                ],
                "bonus_points": [
                    "Candid, unaware moments",
                    "Funny expressions or situations",
                    "Physical closeness/affection",
                ],
            },
            negative_criteria={
                "deal_breakers": [
                    "Can't see baby's face clearly",
                    "Unclear what's happening in the photo",
                    "Parent has vacant/disengaged expression",
                    "Major technical issues (very blurry, bad lighting)",
                ],
                "negative_factors": [
                    "Subjects looking in different directions",
                    "Too zoomed out to see expressions",
                    "Ordinary moment without special quality",
                    "Awkward composition",
                ],
            },
            specific_guidance=[
                "Parent expressions matter as much as baby's.",
                "Photos showing interaction between parent and child are preferred over solo baby shots.",
                "Technical quality is important but not sufficient - a technically perfect photo of an ordinary moment is still Storage.",
                "In burst mode, only keep the absolute best 1-2 photos.",
            ],
            philosophy="Photos that capture genuine moments of connection, emotion, and personality - especially between parent and child.",
            thresholds={
                "Share": 0.60,
                "Review": 0.35,
            },
            photo_settings=PhotoProfileSettings(
                enable_burst_detection=True,
                enable_face_detection=True,

                contains_children_check=True,
                appropriateness_check=True,
            ),
        )
        self._save_profile(profile)
        return profile

    def _create_default_documents_profile(self) -> TasteProfile:
        """Create the default document quality profile."""
        profile = TasteProfile(
            name="default-documents",
            description="Document quality classification - sorts by relevance and quality",
            media_types=["document"],
            categories=[
                CategoryDefinition("Exemplary", "Outstanding, high-quality documents", "#4CAF50"),
                CategoryDefinition("Acceptable", "Adequate quality, keep for reference", "#2196F3"),
                CategoryDefinition("Review", "Uncertain quality, needs manual review", "#FF9800"),
                CategoryDefinition("Discard", "Low quality or irrelevant", "#F44336"),
            ],
            default_category="Review",
            top_priorities=[
                "Content relevance and accuracy",
                "Writing quality and clarity",
                "Completeness of information",
                "Professional formatting",
                "Actionable insights or value",
            ],
            positive_criteria={
                "must_have": [
                    "Clear, readable content",
                    "Relevant to the intended purpose",
                ],
                "highly_valued": [
                    "Well-structured with headings and sections",
                    "Accurate and up-to-date information",
                    "Professional tone and formatting",
                    "Actionable content or clear takeaways",
                ],
            },
            negative_criteria={
                "deal_breakers": [
                    "Unreadable or corrupted content",
                    "Completely irrelevant to purpose",
                    "Severely outdated information",
                ],
                "negative_factors": [
                    "Poor formatting or structure",
                    "Excessive errors or typos",
                    "Redundant with other documents",
                    "Missing key information",
                ],
            },
            specific_guidance=[
                "Evaluate documents based on their intended purpose and audience.",
                "Consider both content quality and presentation.",
                "Group similar documents and identify the best version.",
            ],
            philosophy="Keep documents that provide clear value through accurate, well-presented information relevant to their purpose.",
            thresholds={
                "Exemplary": 0.70,
                "Acceptable": 0.40,
            },
            document_settings=DocumentProfileSettings(
                extract_text=True,
                extract_metadata=True,
                enable_similarity_grouping=True,
                similarity_threshold=0.85,
                max_pages_to_analyze=10,
            ),
        )
        self._save_profile(profile)
        return profile

    def migrate_from_taste_preferences(
        self,
        preferences_path: Path,
        profile_name: str = "default-photos",
    ) -> TasteProfile:
        """Migrate an existing taste_preferences.json to a TasteProfile.

        Args:
            preferences_path: Path to taste_preferences.json or taste_preferences_generated.json.
            profile_name: Name for the new profile.

        Returns:
            The created TasteProfile.
        """
        with open(preferences_path, "r", encoding="utf-8") as f:
            prefs = json.load(f)

        # Map old format to new profile format
        profile = TasteProfile(
            name=profile_name,
            description=prefs.get("description", "Migrated from taste_preferences.json"),
            media_types=["image", "video"],
            categories=[
                CategoryDefinition("Share", "Photos worth sharing with family", "#4CAF50"),
                CategoryDefinition("Storage", "Keep but don't share", "#2196F3"),
                CategoryDefinition("Review", "Uncertain, needs manual review", "#FF9800"),
                CategoryDefinition("Ignore", "No children or inappropriate", "#9E9E9E"),
            ],
            default_category="Review",
            top_priorities=prefs.get("top_priorities", []),
            positive_criteria=prefs.get("share_criteria", {}),
            negative_criteria=prefs.get("reject_criteria", {}),
            specific_guidance=prefs.get("specific_guidance", []),
            philosophy=prefs.get("share_philosophy", ""),
            thresholds={"Share": 0.60, "Review": 0.35},
            photo_settings=PhotoProfileSettings(
                enable_burst_detection=True,
                enable_face_detection=True,

                contains_children_check=True,
                appropriateness_check=True,
            ),
        )

        self._save_profile(profile)
        return profile
