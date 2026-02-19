"""Unified prompt builder for singletons, bursts, videos, and documents."""
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

from ..core.config import Config
from ..core.profiles import TasteProfile


class PromptBuilder:
    """
    Unified prompt builder for all media types.

    Handles loading taste profiles and building consistent prompts for:
    - Single photos
    - Photo bursts
    - Videos
    - Single documents
    - Document groups

    Supports both legacy taste_preferences.json format and the new TasteProfile system.
    """

    def __init__(
        self,
        config: Config,
        training_examples: Optional[Dict] = None,
        profile: Optional[TasteProfile] = None,
    ):
        """
        Initialize prompt builder.

        Args:
            config: Configuration object.
            training_examples: Optional training examples for fallback taste extraction.
            profile: Optional TasteProfile to use. If None, loads from JSON files (legacy).
        """
        self.config = config
        self.training_examples = training_examples or {}
        self.profile = profile
        self.taste_profile_data = self._load_taste_profiles()
        self.share_ratio = self._calculate_share_ratio()

    def _load_taste_profiles(self) -> Dict[str, Any]:
        """Load taste profiles from JSON files (legacy support)."""
        # If a TasteProfile is provided, we derive the taste data from it
        if self.profile is not None:
            return self._profile_to_taste_data(self.profile)

        # Legacy: try generated profile first, then manual
        generated_path = self.config.paths.taste_preferences_generated
        if generated_path.exists():
            try:
                with open(generated_path, "r", encoding="utf-8") as f:
                    profile = json.load(f)
                    print(f"Loaded generated taste profile from {generated_path.name}")
                    return profile
            except Exception as e:
                print(f"Warning: Failed to load generated profile: {e}")

        manual_path = self.config.paths.taste_preferences
        if manual_path.exists():
            try:
                with open(manual_path, "r", encoding="utf-8") as f:
                    profile = json.load(f)
                    print(f"Loaded manual taste profile from {manual_path.name}")
                    return profile
            except Exception as e:
                print(f"Warning: Failed to load manual profile: {e}")

        print("Warning: No taste profiles found, will use training examples")
        return {}

    def _profile_to_taste_data(self, profile: TasteProfile) -> Dict[str, Any]:
        """Convert a TasteProfile to the legacy taste_profile_data dict format."""
        return {
            "share_philosophy": profile.philosophy,
            "top_priorities": profile.top_priorities,
            "share_criteria": profile.positive_criteria,
            "reject_criteria": profile.negative_criteria,
            "specific_guidance": profile.specific_guidance,
            "contextual_preferences": {},
        }

    def _calculate_share_ratio(self) -> float:
        """Calculate share ratio from training examples."""
        if not self.training_examples:
            return self.config.classification.target_share_rate
        share_count = sum(
            1 for data in self.training_examples.values()
            if data.get("action") == "Share"
        )
        total_count = len(self.training_examples)
        if total_count > 0:
            return share_count / total_count
        return self.config.classification.target_share_rate

    # ---- Profile-aware category helpers ----

    @property
    def _category_names(self) -> List[str]:
        """Get the list of valid category names."""
        if self.profile:
            return self.profile.category_names
        return ["Share", "Storage", "Ignore"]

    @property
    def _category_descriptions(self) -> Dict[str, str]:
        """Get category name -> description mapping."""
        if self.profile:
            return {c.name: c.description for c in self.profile.categories}
        return {
            "Share": "Photos worth sharing with family",
            "Storage": "Keep but don't share",
            "Ignore": "No children or inappropriate",
        }

    @property
    def _is_photo_profile(self) -> bool:
        """Check if current profile is photo-oriented."""
        if self.profile:
            return any(t in self.profile.media_types for t in ["image", "video", "mixed"])
        return True

    @property
    def _has_children_check(self) -> bool:
        """Check if children/appropriateness checks should be included."""
        if self.profile and self.profile.photo_settings:
            return self.profile.photo_settings.contains_children_check
        # Legacy: always check for photos
        return self._is_photo_profile

    # ---- Taste section builders ----

    def _build_taste_section(self, mode: str = "singleton") -> str:
        """Build taste profile section."""
        if not self.taste_profile_data:
            return self._build_fallback_taste_section(mode)

        taste_note = ""
        top_priorities = self.taste_profile_data.get("top_priorities", [])
        share_criteria = self.taste_profile_data.get("share_criteria", {})
        reject_criteria = self.taste_profile_data.get("reject_criteria", {})

        if mode == "singleton":
            if top_priorities:
                taste_note = f"\n**Your taste priorities:** {', '.join(top_priorities[:4])}\n"
            if share_criteria.get("must_have"):
                taste_note += f"**Must have:** {', '.join(share_criteria['must_have'][:3])}\n"
            if reject_criteria.get("deal_breakers"):
                taste_note += f"**Deal breakers:** {', '.join(reject_criteria['deal_breakers'][:3])}\n"

        elif mode == "burst":
            if top_priorities:
                taste_note = f"\n**When ranking, prioritize:** {', '.join(top_priorities[:4])}\n"
            burst_philosophy = self.taste_profile_data.get("contextual_preferences", {}).get("burst_philosophy", "")
            if burst_philosophy:
                taste_note += f"**Burst approach:** {burst_philosophy}\n"

        elif mode == "video":
            taste_note = "\n**USER'S TASTE PROFILE (AI-generated from photo training):**\n"
            taste_note += f"Philosophy: {self.taste_profile_data.get('share_philosophy', '')}\n\n"
            if top_priorities:
                taste_note += "**Top Priorities (apply to videos):**\n"
                for i, p in enumerate(top_priorities[:5], 1):
                    taste_note += f"{i}. {p}\n"
                taste_note += "\n"
            if share_criteria.get("must_have"):
                taste_note += f"**Must Have:** {', '.join(share_criteria['must_have'][:3])}\n"
            if share_criteria.get("highly_valued"):
                taste_note += f"**Highly Valued:** {', '.join(share_criteria['highly_valued'][:3])}\n"
            if reject_criteria.get("deal_breakers"):
                taste_note += f"**Deal Breakers:** {', '.join(reject_criteria['deal_breakers'][:3])}\n"
            specific_guidance = self.taste_profile_data.get("specific_guidance", [])
            if specific_guidance:
                taste_note += "\n**Specific Guidance (adapted for video):**\n"
                for g in specific_guidance[:3]:
                    taste_note += f"- {g}\n"
            taste_note += "\n**VIDEO-SPECIFIC CONSIDERATIONS:**\n"
            taste_note += "- Audio matters: Joyful sounds, laughter, and engaged conversation enhance moments\n"
            taste_note += "- Motion reveals personality: Watch for natural gestures, expressions changing over time\n"
            taste_note += "- Interaction quality: Videos show the FLOW of interaction, not just a single moment\n"
            taste_note += "- Avoid: Excessive crying/screaming, very shaky footage, can't see faces clearly\n"

        elif mode == "document":
            taste_note = "\n**EVALUATION CRITERIA:**\n"
            taste_note += f"Philosophy: {self.taste_profile_data.get('share_philosophy', '')}\n\n"
            if top_priorities:
                taste_note += "**Priorities:**\n"
                for i, p in enumerate(top_priorities[:5], 1):
                    taste_note += f"{i}. {p}\n"
                taste_note += "\n"
            if share_criteria.get("must_have"):
                taste_note += f"**Must Have:** {', '.join(share_criteria['must_have'][:3])}\n"
            if share_criteria.get("highly_valued"):
                taste_note += f"**Highly Valued:** {', '.join(share_criteria['highly_valued'][:3])}\n"
            if reject_criteria.get("deal_breakers"):
                taste_note += f"**Deal Breakers:** {', '.join(reject_criteria['deal_breakers'][:3])}\n"

        return taste_note

    def _build_fallback_taste_section(self, mode: str) -> str:
        """Build taste section from training examples when no profile exists."""
        all_reasoning = " ".join([
            data.get("reasoning", "")
            for data in self.training_examples.values()
        ])
        valued_keywords = []
        if "expression" in all_reasoning.lower():
            valued_keywords.append("expressive faces" if mode == "singleton" else "best expressions")
        if "dad" in all_reasoning.lower() or "parent" in all_reasoning.lower():
            valued_keywords.append("parent-child interaction")
        if "mischief" in all_reasoning.lower() or "joy" in all_reasoning.lower():
            valued_keywords.append("joyful moments")
        if "interaction" in all_reasoning.lower():
            valued_keywords.append("genuine interaction")

        if valued_keywords:
            if mode == "burst":
                return f"\n**When ranking, prioritize:** {', '.join(valued_keywords)}\n"
            else:
                return f"\n**Your taste priorities:** {', '.join(valued_keywords)}\n"
        return ""

    def _build_appropriateness_section(self) -> str:
        """Build shared appropriateness checking section (photo-specific)."""
        if not self._has_children_check:
            return ""

        return """**STEP 1 - FIRST CHECK: Does this contain young children (babies/toddlers/preschoolers)?**
- Look carefully at the ENTIRE content
- If you see NO young children -> classify as IGNORE
- Only proceed to Share/Storage if young children ARE visible

**STEP 2 - APPROPRIATENESS CHECK (CRITICAL):**
- **IGNORE if content shows inappropriate nudity or "special parts"**
- Bath time photos showing private areas -> IGNORE
- Diaper changes showing genitals -> IGNORE
- Any content you wouldn't show to grandparents -> IGNORE
- **When in doubt about appropriateness -> IGNORE**

**EXAMPLES OF IGNORE:**
- Food ONLY (no people) -> Ignore
- Landscapes/scenery ONLY (no people) -> Ignore
- Objects/documents ONLY (toys, papers) -> Ignore
- Adults ONLY (no children present) -> Ignore
- Inappropriate nudity or private parts -> Ignore"""

    def _build_scoring_rubric(self) -> str:
        """Build scoring rubric section from profile or generic fallback."""
        if self.profile and self.profile.scoring_rubric:
            rubric = self.profile.scoring_rubric
        else:
            # Generic fallback rubric
            rubric = {
                "5": "Exceptional -- clearly outstanding, no hesitation",
                "4": "Good -- clearly above average, minor imperfections at most",
                "3": "Borderline -- genuinely torn, notable strengths and weaknesses",
                "2": "Below average -- some merit but not enough to stand out",
                "1": "Poor -- significant problems or not relevant",
            }

        lines = ["**SCORING RUBRIC (1-5 scale):**"]
        for level in ["5", "4", "3", "2", "1"]:
            if level in rubric:
                lines.append(f"  {level} = {rubric[level]}")
        lines.append("")
        lines.append("Use the FULL 1-5 range. Score 3 means genuinely borderline. Do not default to 4.")
        return "\n".join(lines)

    def _build_calibration_section(self) -> str:
        """Build calibration guidance section."""
        return f"""**CALIBRATION:**
- In training: {self.share_ratio:.1%} of content is Share-worthy
- Be selective! Not every cute photo/video is share-worthy"""

    def _build_category_format(self) -> str:
        """Build the category list portion of a prompt from the profile."""
        categories = self._category_descriptions
        lines = []
        for name, desc in categories.items():
            lines.append(f"**{name}** = {desc}")

        # Add explicit Review guidance if Review is a valid category
        if "Review" in categories:
            lines.append("")
            lines.append("**When to use Review:** Use Review for borderline cases where you're "
                         "genuinely torn between Share and Storage. If a photo has some share-worthy "
                         "qualities but also notable flaws, or if the moment is nice but you're not "
                         "confident it meets the Share bar, classify as Review. "
                         "Roughly 10-20% of photos with children should land in Review.")

        return "\n".join(lines)

    # ---- Public prompt builders ----

    def build_singleton_prompt(self, media_type: str = "image") -> str:
        """Build prompt for evaluating a single item.

        Dispatches based on media_type.
        """
        if media_type == "image":
            return self._build_image_singleton_prompt()
        elif media_type == "video":
            return self.build_video_prompt()
        elif media_type == "document":
            return self._build_document_singleton_prompt()
        return self._build_image_singleton_prompt()

    def build_group_prompt(self, group_size: int, media_type: str = "image") -> str:
        """Build prompt for comparing a group of items."""
        if media_type == "image":
            return self.build_burst_prompt(group_size)
        elif media_type == "document":
            return self._build_document_group_prompt(group_size)
        return self.build_burst_prompt(group_size)

    def _build_image_singleton_prompt(self) -> str:
        """Build prompt for evaluating a single photo."""
        taste_section = self._build_taste_section("singleton")
        appropriateness_section = self._build_appropriateness_section()
        calibration_section = self._build_calibration_section()

        # Dynamic category names
        cat_names = self._category_names
        cat_str = " or ".join([f'"{c}"' for c in cat_names])

        scoring_rubric = self._build_scoring_rubric()

        json_format = f"""{{
    "classification": {cat_str},
    "score": 1 to 5,
    "reasoning": "Brief explanation (1 sentence max)",
    "contains_children": true or false,
    "is_appropriate": true or false
}}"""

        prompt = f"""You are helping sort family photos of young children into categories: {', '.join(cat_names)}.
{taste_section}

{appropriateness_section}

**ONLY if young children ARE visible AND photo is family-appropriate, evaluate quality:**

{self._build_category_format()}

{scoring_rubric}

{calibration_section}

Analyze this photo and respond with CONCISE JSON:
{json_format}"""

        return prompt

    def build_singleton_with_burst_context(
        self,
        burst_size: int,
        sibling_classifications: List[str],
        original_position: int = 0
    ) -> str:
        """Build singleton prompt with burst context for retry scenarios."""
        base_prompt = self._build_image_singleton_prompt()
        context_lines = [
            "\n\n**BURST CONTEXT (for reference only):**",
            f"This photo was originally part of a burst of {burst_size} photos.",
            f"It was photo #{original_position + 1} in the sequence.",
        ]
        if sibling_classifications:
            class_summary = ", ".join(sibling_classifications[:5])
            if len(sibling_classifications) > 5:
                class_summary += f", ... (+{len(sibling_classifications) - 5} more)"
            context_lines.append(f"The other photos were classified as: {class_summary}.")
        context_lines.append(
            "Evaluate this photo on its own merits, but consider that it captures a similar moment."
        )
        return base_prompt + "\n".join(context_lines)

    def build_burst_prompt(self, burst_size: int) -> str:
        """Build prompt for evaluating a burst of photos."""
        taste_section = self._build_taste_section("burst")
        appropriateness_section = self._build_appropriateness_section()
        calibration_section = self._build_calibration_section()

        cat_names = self._category_names
        cat_str = " or ".join([f'"{c}"' for c in cat_names])

        scoring_rubric = self._build_scoring_rubric()

        json_example = f"""[
  {{
    "rank": 1 to {burst_size},
    "classification": {cat_str},
    "score": 1 to 5,
    "reasoning": "Why this rank? (1 sentence)",
    "contains_children": true or false,
    "is_appropriate": true or false
  }},
  ...
]"""

        prompt = f"""You are helping sort family photos of young children into categories: {', '.join(cat_names)}.

**BURST EVALUATION MODE:**
You are viewing {burst_size} photos taken in rapid succession.
These are very similar - slight variations of the same moment.
{taste_section}

{appropriateness_section}

**ONLY if young children ARE present AND photos are family-appropriate, proceed with:**

**YOUR TASK:**
1. Check appropriateness: Are photos family-friendly?
2. Compare ALL photos in the burst
3. Rank them by quality (1=best, {burst_size}=worst)
4. **CRITICAL**: Also evaluate ABSOLUTE share-worthiness (score 1-5)
   - This is INDEPENDENT of ranking
   - Even Rank 1 might not be absolutely shareable if whole burst is mediocre
   - Score should reflect: "Would I share this if it appeared alone?"
5. Be VERY selective - typically keep only 0-2 photos from a burst
6. Consider: Is this burst even worth keeping ANY photos from?

**EVALUATION CRITERIA (only if children are present AND photos are appropriate):**
- Technical quality: sharpness, lighting, eyes open
- Expression: smile, personality, natural moment
- Composition: framing, background
- **Appropriateness**: Family-friendly, no inappropriate nudity
- **Relative quality**: Rank within THIS burst (1=best)
- **Absolute quality**: Share-worthiness independent of burst (score 1-5)

{scoring_rubric}

**CRITICAL - BE SELECTIVE:**
- NOT every burst deserves a share
- If all photos are mediocre -> Share NONE (rank them but mark all Storage)
- If burst has 1 great photo -> Share that one only
- If burst has 2 exceptional photos -> Share both (rare)

{calibration_section}

Respond with JSON array (one entry per photo, SAME ORDER as shown):
{json_example}"""

        return prompt

    def build_video_prompt(self) -> str:
        """Build prompt for evaluating a video."""
        taste_section = self._build_taste_section("video")
        appropriateness_section = self._build_appropriateness_section().replace(
            "photo", "video"
        ).replace(
            "photos", "videos"
        ).replace(
            "this contain", "this video contain"
        )
        calibration_section = self._build_calibration_section()

        cat_names = self._category_names
        cat_str = " or ".join([f'"{c}"' for c in cat_names])

        scoring_rubric = self._build_scoring_rubric()

        prompt = f"""You are helping sort family VIDEOS of young children into categories: {', '.join(cat_names)}.

**VIDEO EVALUATION - Watch the entire video carefully, considering both visual AND audio content**
{taste_section}

{appropriateness_section}

**ONLY if young children ARE visible AND video is family-appropriate, evaluate quality:**

{self._build_category_format()}

{scoring_rubric}

{calibration_section}

Watch this video and respond with CONCISE JSON:
{{
    "classification": {cat_str},
    "score": 1 to 5,
    "reasoning": "Brief explanation (1-2 sentences max)",
    "contains_children": true or false,
    "is_appropriate": true or false,
    "audio_quality": "good" or "poor" or "silent",
    "highlights": "Key moments if Share-worthy (optional)"
}}"""

        return prompt

    def _build_document_singleton_prompt(self) -> str:
        """Build prompt for evaluating a single document."""
        taste_section = self._build_taste_section("document")
        cat_names = self._category_names
        cat_str = " or ".join([f'"{c}"' for c in cat_names])

        scoring_rubric = self._build_scoring_rubric()

        prompt = f"""You are helping classify documents into categories: {', '.join(cat_names)}.
{taste_section}

{self._build_category_format()}

{scoring_rubric}

Evaluate this document based on the criteria above.
Consider content quality, relevance, completeness, and presentation.

Respond with CONCISE JSON:
{{
    "classification": {cat_str},
    "score": 1 to 5,
    "reasoning": "Brief explanation (1-2 sentences max)",
    "content_summary": "2-3 sentence summary of the document content",
    "key_topics": ["topic1", "topic2", "topic3"]
}}"""

        return prompt

    def _build_document_group_prompt(self, group_size: int) -> str:
        """Build prompt for comparing a group of similar documents."""
        taste_section = self._build_taste_section("document")
        cat_names = self._category_names
        cat_str = " or ".join([f'"{c}"' for c in cat_names])

        scoring_rubric = self._build_scoring_rubric()

        prompt = f"""You are comparing {group_size} similar documents to classify them.
{taste_section}

{self._build_category_format()}

{scoring_rubric}

**YOUR TASK:**
1. Compare ALL documents in this group
2. Rank them by quality (1=best, {group_size}=worst)
3. Evaluate ABSOLUTE quality for each (score 1-5)
4. If documents are near-duplicates, keep only the best version
5. Consider: content quality, completeness, relevance, presentation

Respond with JSON array (one entry per document, SAME ORDER as provided):
[
  {{
    "rank": 1 to {group_size},
    "classification": {cat_str},
    "score": 1 to 5,
    "reasoning": "Why this rank? (1 sentence)",
    "content_summary": "2-3 sentence summary",
    "key_topics": ["topic1", "topic2"]
  }},
  ...
]"""

        return prompt
