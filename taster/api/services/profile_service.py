"""Service layer for taste profile management.

Wraps ProfileManager to provide a clean API-friendly interface that returns
plain dictionaries instead of dataclass instances.
"""
import logging
from typing import List, Dict, Any

from ...core.profiles import ProfileManager, TasteProfile

logger = logging.getLogger(__name__)


class ProfileService:
    """High-level service for managing taste profiles."""

    def __init__(self, profiles_dir: str = "profiles"):
        """
        Initialize ProfileService.

        Args:
            profiles_dir: Directory where profile JSON files are stored.
        """
        self.manager = ProfileManager(profiles_dir=profiles_dir)

    def list_profiles(self) -> List[dict]:
        """
        List all available taste profiles.

        Returns:
            List of profile dictionaries with name, description, media_types,
            and category names.
        """
        profiles = self.manager.list_profiles()
        return [
            {
                "name": p.name,
                "description": p.description,
                "media_types": p.media_types,
                "categories": p.category_names,
                "created_at": p.created_at,
                "updated_at": p.updated_at,
                "version": p.version,
            }
            for p in profiles
        ]

    def get_profile(self, name: str) -> dict:
        """
        Load a single profile by name.

        Args:
            name: Profile name (matches the JSON filename stem).

        Returns:
            Full profile dictionary.

        Raises:
            FileNotFoundError: If the profile does not exist.
        """
        profile = self.manager.load_profile(name)
        return profile.to_dict()

    def create_profile(self, data: dict) -> dict:
        """
        Create a new taste profile.

        Args:
            data: Dictionary containing at minimum ``name``, ``description``,
                ``media_types``, and ``categories``.  Additional fields
                (``default_category``, ``top_priorities``, ``positive_criteria``,
                ``negative_criteria``, ``specific_guidance``, ``philosophy``,
                ``thresholds``, ``photo_settings``, ``document_settings``) are
                optional.

        Returns:
            The newly created profile as a dictionary.

        Raises:
            ValueError: If required fields are missing.
        """
        required = ("name", "description", "media_types", "categories")
        missing = [f for f in required if f not in data]
        if missing:
            raise ValueError(f"Missing required fields: {', '.join(missing)}")

        profile = self.manager.create_profile(
            name=data["name"],
            description=data["description"],
            media_types=data["media_types"],
            categories=data["categories"],
            default_category=data.get("default_category", "Review"),
            top_priorities=data.get("top_priorities"),
            positive_criteria=data.get("positive_criteria"),
            negative_criteria=data.get("negative_criteria"),
            specific_guidance=data.get("specific_guidance"),
            philosophy=data.get("philosophy", ""),
            thresholds=data.get("thresholds"),
            photo_settings=data.get("photo_settings"),
            document_settings=data.get("document_settings"),
        )
        logger.info("Created profile: %s", profile.name)
        return profile.to_dict()

    def update_profile(self, name: str, data: dict) -> dict:
        """
        Update an existing profile with the provided fields.

        Only keys present in *data* are changed; all other fields are preserved.

        Args:
            name: Name of the profile to update.
            data: Dictionary of fields to update.

        Returns:
            The updated profile as a dictionary.

        Raises:
            FileNotFoundError: If the profile does not exist.
        """
        # Filter out keys that are not valid TasteProfile attributes
        valid_keys = {
            "description", "media_types", "categories", "default_category",
            "top_priorities", "positive_criteria", "negative_criteria",
            "specific_guidance", "philosophy", "thresholds",
            "photo_settings", "document_settings",
        }
        update_kwargs = {k: v for k, v in data.items() if k in valid_keys}

        profile = self.manager.update_profile(name, **update_kwargs)
        logger.info("Updated profile: %s (v%d)", profile.name, profile.version)
        return profile.to_dict()

    def delete_profile(self, name: str) -> bool:
        """
        Delete a profile by name.

        Args:
            name: Profile name.

        Returns:
            True if the profile was deleted, False if it did not exist.
        """
        deleted = self.manager.delete_profile(name)
        if deleted:
            logger.info("Deleted profile: %s", name)
        return deleted

    def ensure_defaults(self) -> None:
        """
        Create the default profiles if they do not already exist.

        Ensures both ``default-photos`` and ``default-documents`` are present.
        """
        for media_type, profile_name in [
            ("image", "default-photos"),
            ("document", "default-documents"),
        ]:
            if not self.manager.profile_exists(profile_name):
                self.manager.get_default_profile(media_type)
                logger.info("Created default profile: %s", profile_name)
