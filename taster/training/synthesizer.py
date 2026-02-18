"""Convert pairwise training data into a TasteProfile via AI synthesis."""

import json
from pathlib import Path
from typing import Optional, Union

from PIL import Image

from ..core.ai_client import AIClient
from ..core.file_utils import FileTypeRegistry
from ..core.media_prep import VideoFrameExtractor, PDFPageRenderer
from ..core.profiles import ProfileManager, TasteProfile
from ..features.document_features import DocumentFeatureExtractor
from .session import TrainingSession


class ProfileSynthesizer:
    """
    Synthesize pairwise training data into a TasteProfile.

    Pipeline:
    1. Convert pairwise/gallery labels to Share/Storage example sets
    2. Analyze all reasoning text with AI
    3. Visually analyze Share photos with AI
    4. Visually analyze Storage photos with AI
    5. Synthesize all analyses into TasteProfile JSON
    6. Optionally merge with existing profile for refinement
    """

    MAX_VISUAL_SAMPLES = 15

    def __init__(self, ai_client: AIClient, profile_manager: ProfileManager):
        self.ai_client = ai_client
        self.pm = profile_manager

    def synthesize(
        self,
        session: TrainingSession,
        profile_name: str,
        existing_profile: Optional[TasteProfile] = None,
    ) -> TasteProfile:
        """
        Convert training session data into a TasteProfile.

        Args:
            session: Completed training session with pairwise/gallery labels.
            profile_name: Name for the generated profile.
            existing_profile: If provided, merge/refine rather than replace.

        Returns:
            Created or updated TasteProfile.
        """
        # Step 1: Convert labels to Share/Storage sets
        share_photos, storage_photos, share_reasons, storage_reasons = (
            self._convert_labels(session)
        )

        # Step 2: Analyze reasoning text
        reasoning_analysis = self._analyze_reasoning(
            share_reasons, storage_reasons
        )

        # Step 3: Analyze Share photos visually
        share_visual = self._analyze_share_photos(share_photos)

        # Step 4: Analyze Storage photos visually
        storage_visual = self._analyze_storage_photos(storage_photos)

        # Step 5: Synthesize into profile
        profile_data = self._synthesize_profile(
            reasoning_analysis, share_visual, storage_visual, existing_profile
        )

        if profile_data is None:
            raise RuntimeError("AI failed to synthesize profile data.")

        # Step 6: Create or update profile
        if existing_profile and self.pm.profile_exists(profile_name):
            profile = self.pm.update_profile(
                profile_name,
                description=profile_data.get(
                    "description", existing_profile.description
                ),
                top_priorities=profile_data.get("top_priorities", []),
                positive_criteria=profile_data.get("positive_criteria", {}),
                negative_criteria=profile_data.get("negative_criteria", {}),
                specific_guidance=profile_data.get("specific_guidance", []),
                philosophy=profile_data.get(
                    "philosophy",
                    profile_data.get("share_philosophy", ""),
                ),
            )
        else:
            profile = self.pm.create_profile(
                name=profile_name,
                description=profile_data.get(
                    "description",
                    f"Generated from pairwise training ({len(session.pairwise)} comparisons)",
                ),
                media_types=profile_data.get("media_types", ["image"]),
                categories=profile_data.get("categories", [
                    {"name": "Share", "description": "Worth sharing"},
                    {"name": "Storage", "description": "Keep but don't share"},
                    {"name": "Review", "description": "Needs manual review"},
                ]),
                default_category=profile_data.get("default_category", "Review"),
                top_priorities=profile_data.get("top_priorities", []),
                positive_criteria=profile_data.get("positive_criteria", {}),
                negative_criteria=profile_data.get("negative_criteria", {}),
                specific_guidance=profile_data.get("specific_guidance", []),
                philosophy=profile_data.get(
                    "philosophy",
                    profile_data.get("share_philosophy", ""),
                ),
            )

        return profile

    def _convert_labels(
        self, session: TrainingSession
    ) -> tuple[dict, dict, list, list]:
        """
        Convert pairwise/gallery labels to Share/Storage example sets.

        Returns (share_photos, storage_photos, share_reasons, storage_reasons)
        where each photos dict maps path -> confidence.
        """
        share: dict[str, float] = {}
        storage: dict[str, float] = {}
        share_reasons: list[str] = []
        storage_reasons: list[str] = []

        for comp in session.pairwise:
            if comp.choice == "left":
                share[comp.photo_a] = max(
                    share.get(comp.photo_a, 0), 0.95
                )
                storage[comp.photo_b] = max(
                    storage.get(comp.photo_b, 0), 0.85
                )
                if comp.reason:
                    share_reasons.append(comp.reason)
            elif comp.choice == "right":
                share[comp.photo_b] = max(
                    share.get(comp.photo_b, 0), 0.95
                )
                storage[comp.photo_a] = max(
                    storage.get(comp.photo_a, 0), 0.85
                )
                if comp.reason:
                    share_reasons.append(comp.reason)
            elif comp.choice == "both":
                share[comp.photo_a] = max(
                    share.get(comp.photo_a, 0), 0.9
                )
                share[comp.photo_b] = max(
                    share.get(comp.photo_b, 0), 0.9
                )
                if comp.reason:
                    share_reasons.append(comp.reason)
            elif comp.choice == "neither":
                storage[comp.photo_a] = max(
                    storage.get(comp.photo_a, 0), 0.9
                )
                storage[comp.photo_b] = max(
                    storage.get(comp.photo_b, 0), 0.9
                )
                if comp.reason:
                    storage_reasons.append(comp.reason)

        for sel in session.gallery:
            for i, photo in enumerate(sel.photos):
                if i in sel.selected_indices:
                    share[photo] = max(share.get(photo, 0), 0.95)
                else:
                    storage[photo] = max(storage.get(photo, 0), 0.85)
            if sel.reason:
                share_reasons.append(sel.reason)

        # Remove conflicts (if a photo is in both, keep the higher-confidence one)
        for photo in list(share.keys()):
            if photo in storage:
                if share[photo] >= storage[photo]:
                    del storage[photo]
                else:
                    del share[photo]

        return share, storage, share_reasons, storage_reasons

    def _analyze_reasoning(
        self, share_reasons: list[str], storage_reasons: list[str]
    ) -> Optional[dict]:
        """Analyze reasoning text to extract taste patterns."""
        if not share_reasons and not storage_reasons:
            return None

        share_text = "\n".join(f"- {r}" for r in share_reasons if r)
        storage_text = "\n".join(f"- {r}" for r in storage_reasons if r)

        prompt = f"""Analyze this user's photo sorting reasoning and extract their taste preferences.

**SHARE PHOTOS - User's reasoning for photos they SHARE:**
{share_text or "(no reasoning provided)"}

**STORAGE PHOTOS - User's reasoning for photos they REJECT:**
{storage_text or "(no reasoning provided)"}

Analyze the patterns and respond with JSON:
{{
    "valued_qualities": [
        "list of qualities, themes, and attributes the user values",
        "e.g., parent-child interaction, joyful expressions, engagement"
    ],
    "reject_criteria": [
        "list of qualities, themes, and attributes the user rejects",
        "e.g., can't see face, vacant expressions, poor framing"
    ],
    "specific_preferences": [
        "specific, concrete preferences extracted from reasoning",
        "e.g., both subjects should be engaged, not looking different directions"
    ],
    "emotional_keywords": [
        "emotional qualities the user values",
        "e.g., mischief, joy, intimacy, engagement"
    ],
    "technical_priorities": [
        "technical aspects the user prioritizes",
        "e.g., sharpness, lighting, visible faces, framing"
    ],
    "contextual_insights": [
        "higher-level insights about user's taste",
        "e.g., values interaction over solo shots, parent expressions matter as much as baby"
    ]
}}

Extract ONLY what's clearly stated or strongly implied. Be specific and concrete."""

        return self.ai_client.generate_json(prompt=prompt, fallback=None)

    def _prepare_file_for_analysis(
        self, path: Path,
    ) -> Optional[Union[Path, Image.Image, str]]:
        """Prepare a file for AI analysis.

        Returns a Path (for images/native uploads), PIL Image (for rendered
        previews), or a text string (for documents/code).  Returns None if
        the file cannot be prepared.
        """
        if FileTypeRegistry.is_image(path):
            return path
        elif FileTypeRegistry.is_video(path):
            frames = VideoFrameExtractor.extract_frames(path, max_frames=1)
            return frames[0] if frames else None
        elif path.suffix.lower() == ".pdf":
            pages = PDFPageRenderer.render_pages(path, max_pages=1)
            return pages[0] if pages else None
        else:
            try:
                extractor = DocumentFeatureExtractor(max_chars=3000)
                text = extractor.extract_text(path)
                if text:
                    return f"[{path.suffix.upper().lstrip('.')}] {path.name}:\n{text[:2000]}"
            except Exception:
                pass
            return None

    def _analyze_share_photos(
        self, share_photos: dict[str, float]
    ) -> Optional[dict]:
        """Analyze Share files to identify positive patterns."""
        if not share_photos:
            return None

        visual_items: list[Union[Path, Image.Image]] = []
        text_items: list[str] = []
        for path_str in list(share_photos.keys()):
            p = Path(path_str)
            if not p.exists():
                continue
            item = self._prepare_file_for_analysis(p)
            if item is None:
                continue
            if isinstance(item, str):
                text_items.append(item)
            else:
                visual_items.append(item)
            if len(visual_items) + len(text_items) >= self.MAX_VISUAL_SAMPLES:
                break

        if not visual_items and not text_items:
            return None

        total = len(visual_items) + len(text_items)
        text_context = ""
        if text_items:
            text_context = (
                "\n\n**TEXT-BASED FILES (documents/code) the user CHOSE TO SHARE:**\n"
                + "\n---\n".join(text_items)
            )

        prompt_parts: list = [
            f"""Analyze these {total} files that the user CHOSE TO SHARE.

What patterns do you see? What makes these files share-worthy?

Consider:
- Subject matter, content quality, and composition
- For images: expressions, interactions, technical quality, framing
- For documents/code: clarity, structure, relevance, quality
- Common themes or patterns across all files

Respond with JSON:
{{
    "visual_patterns": [
        "list of common visual or content characteristics"
    ],
    "compositional_patterns": [
        "structural patterns: framing, organization, layout"
    ],
    "expression_patterns": [
        "for images: expressions and emotions; for text: tone and voice"
    ],
    "interaction_patterns": [
        "patterns in how subjects interact or content connects"
    ],
    "quality_baseline": "description of minimum quality standard",
    "summary": "2-3 sentence summary of what makes these files share-worthy"
}}

Be specific and concrete. Look for patterns across ALL files.{text_context}"""
        ]

        for item in visual_items:
            prompt_parts.append(item)

        response = self.ai_client.generate(prompt_parts)
        return response.parse_json(fallback=None)

    def _analyze_storage_photos(
        self, storage_photos: dict[str, float]
    ) -> Optional[dict]:
        """Analyze Storage/rejected files to identify rejection patterns."""
        if not storage_photos:
            return None

        visual_items: list[Union[Path, Image.Image]] = []
        text_items: list[str] = []
        for path_str in list(storage_photos.keys()):
            p = Path(path_str)
            if not p.exists():
                continue
            item = self._prepare_file_for_analysis(p)
            if item is None:
                continue
            if isinstance(item, str):
                text_items.append(item)
            else:
                visual_items.append(item)
            if len(visual_items) + len(text_items) >= self.MAX_VISUAL_SAMPLES:
                break

        if not visual_items and not text_items:
            return None

        total = len(visual_items) + len(text_items)
        text_context = ""
        if text_items:
            text_context = (
                "\n\n**TEXT-BASED FILES (documents/code) the user REJECTED:**\n"
                + "\n---\n".join(text_items)
            )

        prompt_parts: list = [
            f"""Analyze these {total} files that the user REJECTED (Storage).

What makes these files NOT share-worthy? What patterns do you see?

Consider:
- For images: technical issues, expression problems, composition
- For documents/code: quality issues, unclear writing, poor structure
- Common rejection reasons across all files

Respond with JSON:
{{
    "rejection_patterns": [
        "list of common reasons for rejection"
    ],
    "technical_issues": [
        "technical or quality problems that lead to rejection"
    ],
    "expression_issues": [
        "for images: expression problems; for text: tone/voice issues"
    ],
    "composition_issues": [
        "structural, framing, or organization problems"
    ],
    "summary": "2-3 sentence summary of what makes these files rejectable"
}}

Be specific. Look for patterns across ALL files.{text_context}"""
        ]

        for item in visual_items:
            prompt_parts.append(item)

        response = self.ai_client.generate(prompt_parts)
        return response.parse_json(fallback=None)

    def _synthesize_profile(
        self,
        reasoning_analysis: Optional[dict],
        share_visual: Optional[dict],
        storage_visual: Optional[dict],
        existing_profile: Optional[TasteProfile],
    ) -> Optional[dict]:
        """Synthesize all analyses into a profile definition."""
        existing_context = ""
        if existing_profile:
            existing_context = f"""

**EXISTING PROFILE TO REFINE:**
{json.dumps(existing_profile.to_dict(), indent=2)}

Incorporate the new training data to REFINE this profile. Keep what works,
adjust what the training data contradicts, and add new insights."""

        prompt = f"""Synthesize these analyses into a comprehensive file sorting taste profile.

**REASONING TEXT ANALYSIS:**
{json.dumps(reasoning_analysis, indent=2) if reasoning_analysis else "Not available"}

**SHARE PHOTOS VISUAL ANALYSIS:**
{json.dumps(share_visual, indent=2) if share_visual else "Not available"}

**STORAGE PHOTOS VISUAL ANALYSIS:**
{json.dumps(storage_visual, indent=2) if storage_visual else "Not available"}
{existing_context}

Create a complete taste profile as JSON:
{{
    "description": "One sentence describing what this profile sorts",
    "media_types": ["image"],
    "categories": [
        {{"name": "Share", "description": "Photos worth sharing with family and friends"}},
        {{"name": "Storage", "description": "Keep for archive but not worth sharing"}},
        {{"name": "Review", "description": "Borderline photos needing manual review"}}
    ],
    "default_category": "Review",
    "philosophy": "2-3 sentence philosophy of what makes photos share-worthy for this user",
    "top_priorities": [
        "5-7 most important factors in order of priority",
        "Be specific and actionable"
    ],
    "positive_criteria": {{
        "must_have": ["non-negotiable requirements"],
        "highly_valued": ["strongly preferred attributes"],
        "bonus_points": ["nice-to-have qualities"]
    }},
    "negative_criteria": {{
        "deal_breakers": ["automatic rejection reasons"],
        "negative_factors": ["qualities that count against a photo"]
    }},
    "specific_guidance": [
        "specific, actionable guidance for photo evaluation",
        "e.g., 'Both parent and child should have engaged expressions'",
        "e.g., 'Reject if can't see baby's face clearly, even if parent is well-framed'"
    ]
}}

Synthesize insights across all available analyses. Be specific, concrete, and actionable."""

        return self.ai_client.generate_json(prompt=prompt, fallback=None)

    def refine_from_corrections(
        self,
        profile_name: str,
        corrections: list[dict],
    ) -> TasteProfile:
        """
        Refine an existing profile based on post-classification corrections.

        Args:
            profile_name: Name of profile to refine.
            corrections: List of {file_path, original_category, correct_category, reason}.

        Returns:
            Updated TasteProfile.
        """
        profile = self.pm.load_profile(profile_name)

        corrections_text = "\n".join(
            f"- {c.get('file_path', 'unknown')}: "
            f"was '{c.get('original_category', '?')}', "
            f"should be '{c.get('correct_category', '?')}'"
            f"{' — ' + c['reason'] if c.get('reason') else ''}"
            for c in corrections
        )

        prompt = f"""A media classification profile needs refinement based on user corrections.

**CURRENT PROFILE:**
{json.dumps(profile.to_dict(), indent=2)}

**USER CORRECTIONS (files that were misclassified):**
{corrections_text}

Analyze the correction patterns:
1. What kinds of files are being miscategorized?
2. What criteria need adjustment?
3. Are there systematic biases (e.g., too strict, too lenient)?

Generate updated profile fields as JSON:
{{
    "top_priorities": ["updated priority list incorporating corrections"],
    "positive_criteria": {{
        "must_have": ["adjusted must-have criteria"],
        "highly_valued": ["adjusted highly-valued criteria"],
        "bonus_points": ["adjusted bonus criteria"]
    }},
    "negative_criteria": {{
        "deal_breakers": ["adjusted deal-breaker criteria"],
        "negative_factors": ["adjusted negative factors"]
    }},
    "specific_guidance": ["updated guidance rules based on correction patterns"],
    "changes_made": ["list of what was changed and why"]
}}

Keep existing criteria that weren't contradicted by corrections.
Add or modify criteria to prevent the observed misclassifications.
Be specific about what changed and why."""

        result = self.ai_client.generate_json(prompt=prompt, fallback=None)
        if result is None:
            raise RuntimeError("AI failed to generate refined profile data.")

        changes_made = result.pop("changes_made", [])

        # Only update fields the AI actually returned
        update_kwargs = {}
        for field_name in [
            "top_priorities", "positive_criteria",
            "negative_criteria", "specific_guidance",
        ]:
            if field_name in result:
                update_kwargs[field_name] = result[field_name]

        if not update_kwargs:
            raise RuntimeError(
                "AI returned no updatable fields. Corrections may be insufficient."
            )

        updated = self.pm.update_profile(profile_name, **update_kwargs)
        # Attach changes metadata for the caller
        updated._refinement_changes = changes_made
        return updated
