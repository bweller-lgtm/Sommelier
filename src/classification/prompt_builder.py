"""Unified prompt builder for singletons, bursts, and videos."""
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

from ..core.config import Config


class PromptBuilder:
    """
    Unified prompt builder for all media types.

    Handles loading taste profiles and building consistent prompts for:
    - Single photos
    - Photo bursts
    - Videos
    """

    def __init__(self, config: Config, training_examples: Optional[Dict] = None):
        """
        Initialize prompt builder.

        Args:
            config: Configuration object.
            training_examples: Optional training examples for fallback taste extraction.
        """
        self.config = config
        self.training_examples = training_examples or {}
        self.taste_profile_data = self._load_taste_profiles()

        # Calculate share ratio from training examples
        self.share_ratio = self._calculate_share_ratio()

    def _load_taste_profiles(self) -> Dict[str, Any]:
        """
        Load taste profiles from both JSON files.

        Prioritizes taste_preferences_generated.json, falls back to taste_preferences.json.

        Returns:
            Dictionary with taste profile data.
        """
        # Try generated profile first
        generated_path = self.config.paths.taste_preferences_generated
        if generated_path.exists():
            try:
                with open(generated_path, "r", encoding="utf-8") as f:
                    profile = json.load(f)
                    print(f"✅ Loaded generated taste profile from {generated_path.name}")
                    return profile
            except Exception as e:
                print(f"⚠️  Failed to load generated profile: {e}")

        # Fall back to manual profile
        manual_path = self.config.paths.taste_preferences
        if manual_path.exists():
            try:
                with open(manual_path, "r", encoding="utf-8") as f:
                    profile = json.load(f)
                    print(f"✅ Loaded manual taste profile from {manual_path.name}")
                    return profile
            except Exception as e:
                print(f"⚠️  Failed to load manual profile: {e}")

        print("⚠️  No taste profiles found, will use training examples")
        return {}

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

    def _build_taste_section(self, mode: str = "singleton") -> str:
        """
        Build taste profile section.

        Args:
            mode: "singleton", "burst", or "video"

        Returns:
            Formatted taste profile string.
        """
        if not self.taste_profile_data:
            return self._build_fallback_taste_section(mode)

        taste_note = ""

        # Extract key sections
        top_priorities = self.taste_profile_data.get("top_priorities", [])
        share_criteria = self.taste_profile_data.get("share_criteria", {})
        reject_criteria = self.taste_profile_data.get("reject_criteria", {})

        if mode == "singleton":
            # Concise version for singletons
            if top_priorities:
                taste_note = f"\n**Your taste priorities:** {', '.join(top_priorities[:4])}\n"
            if share_criteria.get("must_have"):
                taste_note += f"**Must have:** {', '.join(share_criteria['must_have'][:3])}\n"
            if reject_criteria.get("deal_breakers"):
                taste_note += f"**Deal breakers:** {', '.join(reject_criteria['deal_breakers'][:3])}\n"

        elif mode == "burst":
            # Burst-specific
            if top_priorities:
                taste_note = f"\n**When ranking, prioritize:** {', '.join(top_priorities[:4])}\n"

            burst_philosophy = self.taste_profile_data.get("contextual_preferences", {}).get("burst_philosophy", "")
            if burst_philosophy:
                taste_note += f"**Burst approach:** {burst_philosophy}\n"

        elif mode == "video":
            # Comprehensive for videos
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
                    taste_note += f"• {g}\n"

            # Video-specific guidance
            taste_note += "\n**VIDEO-SPECIFIC CONSIDERATIONS:**\n"
            taste_note += "• Audio matters: Joyful sounds, laughter, and engaged conversation enhance moments\n"
            taste_note += "• Motion reveals personality: Watch for natural gestures, expressions changing over time\n"
            taste_note += "• Interaction quality: Videos show the FLOW of interaction, not just a single moment\n"
            taste_note += "• Avoid: Excessive crying/screaming, very shaky footage, can't see faces clearly\n"

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
        """Build shared appropriateness checking section."""
        return """**STEP 1 - FIRST CHECK: Does this contain young children (babies/toddlers/preschoolers)?**
- Look carefully at the ENTIRE content
- If you see NO young children → classify as IGNORE
- Only proceed to Share/Storage if young children ARE visible

**STEP 2 - APPROPRIATENESS CHECK (CRITICAL):**
- **IGNORE if content shows inappropriate nudity or "special parts"**
- Bath time photos showing private areas → IGNORE
- Diaper changes showing genitals → IGNORE
- Any content you wouldn't show to grandparents → IGNORE
- **When in doubt about appropriateness → IGNORE**

**EXAMPLES OF IGNORE:**
- Food ONLY (no people) → Ignore
- Landscapes/scenery ONLY (no people) → Ignore
- Objects/documents ONLY (toys, papers) → Ignore
- Adults ONLY (no children present) → Ignore
- Inappropriate nudity or private parts → Ignore"""

    def _build_calibration_section(self) -> str:
        """Build calibration guidance section."""
        return f"""**CALIBRATION:**
- In training: {self.share_ratio:.1%} of content is Share-worthy
- Be selective! Not every cute photo/video is share-worthy"""

    def _build_gray_zone_section(self) -> str:
        """Build gray zone detection section for improvement candidates."""
        cost = self.config.photo_improvement.cost_per_image
        return f"""
**GRAY ZONE DETECTION (IMPORTANT):**
Some photos capture irreplaceable moments but have technical issues. Identify these "improvement candidates."

**HIGH CONTEXTUAL VALUE indicators:**
- Rare family groupings (child with both parents, grandparents, siblings together)
- Parent-child interaction with genuine eye contact or emotional connection
- First-time milestones (first steps, first words reaction, first day of school)
- Candid moments showing authentic personality
- Multi-generational moments

**TECHNICAL ISSUES to detect:**
- motion_blur: Subject or camera movement causing blur
- focus_blur: Missed focus, wrong focal point
- noise: High ISO grain, especially in low light
- underexposed: Too dark, lost shadow detail
- overexposed: Too bright, blown highlights
- white_balance: Color cast (too orange, too blue)
- low_resolution: Noticeably pixelated when viewed at full size
- composition: Could benefit from cropping or straightening

**IMPROVEMENT CANDIDATE CRITERIA:**
Flag as improvement_candidate=true ONLY if:
1. Photo has HIGH contextual value (irreplaceable moment)
2. Photo has at least one significant technical issue
3. The technical issue prevents this from being Share-worthy
4. The moment captured is special enough to warrant ~${cost:.2f} improvement cost

Include in your response:
- "improvement_candidate": true or false
- "improvement_reasons": ["issue1", "issue2"] (from list above, empty if not a candidate)
- "contextual_value": "high", "medium", or "low"
- "contextual_value_reasoning": "Why this moment matters (1 sentence, empty if low)"
"""

    def build_singleton_prompt(self) -> str:
        """
        Build prompt for evaluating a single photo.

        Returns:
            Complete prompt string.
        """
        taste_section = self._build_taste_section("singleton")
        appropriateness_section = self._build_appropriateness_section()
        calibration_section = self._build_calibration_section()

        # Add gray zone section if improvement is enabled
        gray_zone_section = ""
        if self.config.photo_improvement.enabled:
            gray_zone_section = self._build_gray_zone_section()

        # Build JSON response format based on enabled features
        if self.config.photo_improvement.enabled:
            json_format = """{
    "classification": "Share" or "Storage" or "Ignore",
    "confidence": 0.0 to 1.0,
    "reasoning": "Brief explanation (1 sentence max)",
    "contains_children": true or false,
    "is_appropriate": true or false,
    "improvement_candidate": true or false,
    "improvement_reasons": ["motion_blur", "noise", ...] or [],
    "contextual_value": "high" or "medium" or "low",
    "contextual_value_reasoning": "Why this moment matters (if high/medium)"
}"""
        else:
            json_format = """{
    "classification": "Share" or "Storage" or "Ignore",
    "confidence": 0.0 to 1.0,
    "reasoning": "Brief explanation (1 sentence max)",
    "contains_children": true or false,
    "is_appropriate": true or false  // Family-friendly? No inappropriate nudity?
}"""

        prompt = f"""You are helping sort family photos of young children into three categories: SHARE, STORAGE, or IGNORE.
{taste_section}

{appropriateness_section}

**ONLY if young children ARE visible AND photo is family-appropriate, evaluate quality:**

**SHARE = Photos worth sharing with family**
- Shows child's personality, expression, or special moment
- Good technical quality (sharp, well-lit, eyes open)
- Parent/adult expressions matter too
- Genuine interaction or emotional moments

**STORAGE = Photos of children that aren't share-worthy**
- Poor quality (blurry, bad lighting, eyes closed)
- Not special or interesting
- Can't see child's face clearly
- Parent expressions are poor/vacant

{calibration_section}
{gray_zone_section}
Analyze this photo and respond with CONCISE JSON:
{json_format}"""

        return prompt

    def build_burst_prompt(self, burst_size: int) -> str:
        """
        Build prompt for evaluating a burst of photos.

        Args:
            burst_size: Number of photos in the burst.

        Returns:
            Complete prompt string.
        """
        taste_section = self._build_taste_section("burst")
        appropriateness_section = self._build_appropriateness_section()
        calibration_section = self._build_calibration_section()

        # Add gray zone section if improvement is enabled
        gray_zone_section = ""
        if self.config.photo_improvement.enabled:
            gray_zone_section = self._build_gray_zone_section()

        # Build JSON response format based on enabled features
        if self.config.photo_improvement.enabled:
            json_example = f"""[
  {{
    "rank": 1 to {burst_size},
    "classification": "Share" or "Storage" or "Ignore",
    "confidence": 0.0 to 1.0,
    "reasoning": "Why this rank? (1 sentence)",
    "contains_children": true or false,
    "is_appropriate": true or false,
    "improvement_candidate": true or false,
    "improvement_reasons": ["motion_blur", "noise", ...] or [],
    "contextual_value": "high" or "medium" or "low",
    "contextual_value_reasoning": "Why this moment matters (if high/medium)"
  }},
  ...
]"""
        else:
            json_example = f"""[
  {{
    "rank": 1 to {burst_size},
    "classification": "Share" or "Storage" or "Ignore",
    "confidence": 0.0 to 1.0,
    "reasoning": "Why this rank? (1 sentence)",
    "contains_children": true or false,
    "is_appropriate": true or false
  }},
  ...
]"""

        prompt = f"""You are helping sort family photos of young children into three categories: SHARE, STORAGE, or IGNORE.

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
4. **CRITICAL**: Also evaluate ABSOLUTE share-worthiness (0.0-1.0)
   - This is INDEPENDENT of ranking
   - Even Rank 1 might not be absolutely shareable if whole burst is mediocre
   - Confidence should reflect: "Would I share this if it appeared alone?"
5. Be VERY selective - typically keep only 0-2 photos from a burst
6. Consider: Is this burst even worth keeping ANY photos from?

**EVALUATION CRITERIA (only if children are present AND photos are appropriate):**
- Technical quality: sharpness, lighting, eyes open
- Expression: smile, personality, natural moment
- Composition: framing, background
- **Appropriateness**: Family-friendly, no inappropriate nudity
- **Relative quality**: Rank within THIS burst (1=best)
- **Absolute quality**: Share-worthiness independent of burst (confidence score)

**CRITICAL - BE SELECTIVE:**
- NOT every burst deserves a share
- If all photos are mediocre → Share NONE (rank them but mark all Storage)
- If burst has 1 great photo → Share that one only
- If burst has 2 exceptional photos → Share both (rare)

{calibration_section}
{gray_zone_section}
Respond with JSON array (one entry per photo, SAME ORDER as shown):
{json_example}"""

        return prompt

    def build_video_prompt(self) -> str:
        """
        Build prompt for evaluating a video.

        Returns:
            Complete prompt string.
        """
        taste_section = self._build_taste_section("video")
        appropriateness_section = self._build_appropriateness_section().replace(
            "photo", "video"
        ).replace(
            "photos", "videos"
        ).replace(
            "this contain", "this video contain"
        )
        calibration_section = self._build_calibration_section()

        prompt = f"""You are helping sort family VIDEOS of young children into three categories: SHARE, STORAGE, or IGNORE.

**VIDEO EVALUATION - Watch the entire video carefully, considering both visual AND audio content**
{taste_section}

{appropriateness_section}

**ONLY if young children ARE visible AND video is family-appropriate, evaluate quality:**

**SHARE = Videos worth sharing with family**
- Captures personality, expressions, special moments
- Good quality (stable, clear, good audio)
- Shows interaction, emotion, or development
- Joyful sounds (laughter, talking, playing)

**STORAGE = Videos of children that aren't share-worthy**
- Poor quality (very shaky, dark, muffled audio)
- Not special or interesting
- Excessive crying or screaming
- Can't see faces clearly
- Nothing meaningful happening

{calibration_section}

Watch this video and respond with CONCISE JSON:
{{
    "classification": "Share" or "Storage" or "Ignore",
    "confidence": 0.0 to 1.0,
    "reasoning": "Brief explanation (1-2 sentences max)",
    "contains_children": true or false,
    "is_appropriate": true or false,
    "audio_quality": "good" or "poor" or "silent",
    "highlights": "Key moments if Share-worthy (optional)"
}}"""

        return prompt
