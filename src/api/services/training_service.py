"""Service layer for collecting user feedback and generating profiles.

Stores feedback entries as a JSON file on disk and provides helpers to
aggregate statistics and bootstrap a new taste profile from accumulated
corrections.
"""
import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

from ...core.profiles import ProfileManager

logger = logging.getLogger(__name__)

FEEDBACK_FILENAME = "training_feedback.json"


class TrainingService:
    """Collects classification feedback and can generate profiles from it."""

    def __init__(self, profiles_dir: str = "profiles"):
        """
        Initialize TrainingService.

        Args:
            profiles_dir: Directory where profile JSON files are stored.
                The feedback file is also written here.
        """
        self.profiles_dir = Path(profiles_dir)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        self.feedback_path = self.profiles_dir / FEEDBACK_FILENAME
        self.manager = ProfileManager(profiles_dir=str(self.profiles_dir))

    # ------------------------------------------------------------------
    # Feedback persistence helpers
    # ------------------------------------------------------------------

    def _load_feedback(self) -> List[Dict[str, Any]]:
        """Load the feedback list from disk."""
        if not self.feedback_path.exists():
            return []
        try:
            with open(self.feedback_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            return []
        except (json.JSONDecodeError, OSError):
            logger.warning("Could not read feedback file; starting fresh")
            return []

    def _save_feedback(self, feedback: List[Dict[str, Any]]) -> None:
        """Persist the feedback list to disk."""
        with open(self.feedback_path, "w", encoding="utf-8") as f:
            json.dump(feedback, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit_feedback(
        self,
        file_path: str,
        correct_category: str,
        reasoning: str = "",
    ) -> dict:
        """
        Record a user correction for a classified file.

        Args:
            file_path: Path to the file that was misclassified.
            correct_category: The category the user believes is correct.
            reasoning: Optional explanation of why this is the right category.

        Returns:
            The stored feedback entry as a dictionary.
        """
        entry = {
            "file_path": str(file_path),
            "correct_category": correct_category,
            "reasoning": reasoning,
            "submitted_at": datetime.utcnow().isoformat(),
        }

        feedback = self._load_feedback()
        feedback.append(entry)
        self._save_feedback(feedback)

        logger.info(
            "Feedback recorded: %s -> %s", file_path, correct_category
        )
        return entry

    def get_stats(self) -> dict:
        """
        Return aggregate statistics about collected feedback.

        Returns:
            Dictionary with ``total_feedback`` count and ``by_category``
            mapping of category name to count.
        """
        feedback = self._load_feedback()
        by_category: Dict[str, int] = defaultdict(int)
        for entry in feedback:
            by_category[entry.get("correct_category", "unknown")] += 1

        return {
            "total_feedback": len(feedback),
            "by_category": dict(by_category),
        }

    def generate_profile_from_feedback(self, profile_name: str) -> dict:
        """
        Generate a new taste profile seeded by accumulated feedback.

        The method analyses feedback entries to derive initial criteria:
        positive criteria come from reasoning on highly-rated categories,
        and negative criteria from corrections away from lower categories.

        Args:
            profile_name: Name for the new profile.

        Returns:
            The created profile as a dictionary.

        Raises:
            ValueError: If there is no feedback to learn from.
        """
        feedback = self._load_feedback()
        if not feedback:
            raise ValueError(
                "No feedback has been submitted yet. "
                "Submit corrections before generating a profile."
            )

        # Aggregate categories and reasoning
        by_category: Dict[str, List[str]] = defaultdict(list)
        for entry in feedback:
            cat = entry.get("correct_category", "Review")
            reason = entry.get("reasoning", "")
            if reason:
                by_category[cat].append(reason)

        # Build category definitions from feedback
        categories = [
            {"name": cat, "description": f"User-defined via feedback ({len(reasons)} examples)"}
            for cat, reasons in by_category.items()
        ]
        if not categories:
            categories = [
                {"name": "Good", "description": "High-quality items"},
                {"name": "Review", "description": "Needs manual review"},
            ]

        # Derive criteria from reasoning text
        positive_reasons = []
        negative_reasons = []
        for cat, reasons in by_category.items():
            cat_lower = cat.lower()
            if cat_lower in ("share", "good", "exemplary", "keep"):
                positive_reasons.extend(reasons)
            elif cat_lower in ("ignore", "discard", "delete", "reject"):
                negative_reasons.extend(reasons)

        positive_criteria = {}
        if positive_reasons:
            positive_criteria["user_feedback"] = positive_reasons[:20]

        negative_criteria = {}
        if negative_reasons:
            negative_criteria["user_feedback"] = negative_reasons[:20]

        # Determine media types from file extensions in feedback
        media_types = set()
        image_exts = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".tif", ".tiff", ".bmp"}
        video_exts = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
        doc_exts = {".pdf", ".docx", ".xlsx", ".pptx", ".html", ".txt", ".md"}

        for entry in feedback:
            ext = Path(entry.get("file_path", "")).suffix.lower()
            if ext in image_exts:
                media_types.add("image")
            elif ext in video_exts:
                media_types.add("video")
            elif ext in doc_exts:
                media_types.add("document")

        if not media_types:
            media_types = {"image"}

        profile = self.manager.create_profile(
            name=profile_name,
            description=f"Auto-generated from {len(feedback)} feedback entries",
            media_types=sorted(media_types),
            categories=categories,
            default_category="Review",
            top_priorities=[
                "Match user corrections from training feedback",
            ],
            positive_criteria=positive_criteria or None,
            negative_criteria=negative_criteria or None,
            specific_guidance=[
                f"This profile was generated from {len(feedback)} user corrections.",
                "Refine criteria as more feedback is collected.",
            ],
            philosophy="Learned from user feedback to match their taste preferences.",
        )

        logger.info(
            "Generated profile '%s' from %d feedback entries",
            profile_name,
            len(feedback),
        )
        return profile.to_dict()
