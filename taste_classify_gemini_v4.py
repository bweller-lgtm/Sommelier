# taste_classify_gemini_v4_burst_grouped.py
# Grouped burst evaluation with fixes for:
# 1. Model version (2.5-flash)
# 2. Confidence interpretation (rank-based routing instead)
# 3. Quality threshold (don't auto-share from terrible bursts)
# 4. Better messaging about when files move

import os
import sys
import json
import pickle
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm
import google.generativeai as genai
import open_clip
import torch
import torch.nn.functional as F
import time
from datetime import timedelta

# Import from existing code
sys.path.insert(0, '/mnt/project')
sys.path.insert(0, '/home/claude')

from taste_sort_win import (
    list_images, DRY_RUN, get_cache_key,
    copy_file, CACHE_ROOT, load_model, embed_paths
)

# These will be set via command-line arguments
UNLABELED = None
OUT_BASE = None

# Import burst detection
try:
    from burst_detector import detect_bursts_temporal_visual
    from burst_features import compute_burst_features
    USE_BURST_DETECTION = True
    print("✅ Burst detection enabled")
except ImportError as e:
    USE_BURST_DETECTION = False
    print(f"⚠️  Burst detection disabled: {e}")

# ========================== IMAGE UTILITIES ==========================
def fix_image_orientation(image_path, max_size=1024):
    """Load image, fix orientation, and resize for faster loading."""
    try:
        img = Image.open(image_path)
        img = ImageOps.exif_transpose(img)
        
        width, height = img.size
        if max(width, height) > max_size:
            if width > height:
                new_width = max_size
                new_height = int(height * (max_size / width))
            else:
                new_height = max_size
                new_width = int(width * (max_size / height))
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        return img
    except Exception:
        try:
            return Image.open(image_path)
        except:
            return None

# ========================== VIDEO HANDLING ==========================
VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.m4v', '.3gp', '.wmv', '.flv', '.webm'}

def is_video_file(path):
    """Check if file is a video."""
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS

# ========================== BURST DIVERSITY CHECK ==========================
def check_diversity_between_photos(photo_a_path, photo_b_path, cache):
    """
    Check if two photos from same burst are meaningfully different.
    Returns (is_diverse, confidence, reasoning).
    Low barrier: just need different expressions or composition, not major differences.
    """
    cache_key = f"diversity_{get_cache_key(Path(photo_a_path))[:8]}_{get_cache_key(Path(photo_b_path))[:8]}"

    if cache_key in cache:
        result = cache[cache_key]
        return result.get("is_diverse", True), result.get("confidence", 0.5), result.get("reasoning", "")

    if not GEMINI_API_KEY:
        return True, 0.5, "No API"

    try:
        img_a = fix_image_orientation(photo_a_path, max_size=800)
        img_b = fix_image_orientation(photo_b_path, max_size=800)

        if img_a is None or img_b is None:
            return True, 0.5, "Could not load images"

        model = genai.GenerativeModel('gemini-2.5-flash')

        prompt = """Compare these two photos from the same burst. Are they meaningfully DIFFERENT enough to both be worth keeping?

**Context:** These are consecutive photos from the same moment. We want to keep multiple photos from a burst ONLY if they capture genuinely different moments or expressions.

**"Meaningfully different" means (keep BOTH):**
- Clearly different facial expressions (smiling → laughing, eyes closed → open, neutral → big smile)
- Significantly different composition (different crop, different angle, different framing)
- Different action/moment (before → after movement, different interaction between people)
- Different pose (not just micro-adjustments)

**NOT meaningfully different (keep ONLY the best one):**
- Nearly identical - just camera shake or tiny position shifts (1-2cm)
- Same general pose, same general expression (minor micro-expression changes don't count)
- Same composition with fractional timing difference
- One is just a worse/blurrier version of the other
- Baby lying in same position on blanket with no real expression change

**Decision rule:** If a regular person looking at both photos would say "these look the same to me", mark as NOT diverse. Only keep both if there's a clear, human-noticeable difference. When uncertain, be slightly conservative (lean toward NOT diverse to avoid too many duplicates).

Respond with JSON:
{
    "is_diverse": true or false,
    "confidence": 0.0 to 1.0,
    "reasoning": "Brief explanation (10 words max)",
    "key_difference": "What's different? (5 words max)"
}

Examples:
- Baby on blanket, same pose, same expression, 2 seconds apart → is_diverse=false
- Identical pose, identical smile, no visible difference → is_diverse=false
- Same pose but eyes closed vs open → is_diverse=true
- Noticeable expression change (neutral → big smile) → is_diverse=true
- Significantly different crop/angle → is_diverse=true
- One sharp, one blurry, otherwise identical → is_diverse=false
- Just camera shake or tiny shift, no real difference → is_diverse=false
- Minor head position change but same expression → is_diverse=false
"""

        import time
        time.sleep(0.2)

        response = model.generate_content(
            [prompt, img_a, img_b],
            generation_config={'max_output_tokens': 512}
        )

        # Validate response before accessing text
        if not response.candidates:
            print(f"\n⚠️  Diversity check blocked by safety filters")
            return True, 0.5, "Safety filter - assuming diverse"

        candidate = response.candidates[0]
        if candidate.finish_reason != 1:  # 1 = STOP (normal completion)
            print(f"\n⚠️  Diversity check finish reason: {candidate.finish_reason}")
            return True, 0.5, "Incomplete response - assuming diverse"

        text = response.text.strip()

        # Parse JSON
        import re
        text_cleaned = re.sub(r'^```json\s*\n', '', text, flags=re.MULTILINE)
        text_cleaned = re.sub(r'^```\s*\n', '', text_cleaned, flags=re.MULTILINE)
        text_cleaned = re.sub(r'\n```\s*$', '', text_cleaned, flags=re.MULTILINE)
        text_cleaned = text_cleaned.strip()

        json_match = re.search(r'\{[^}]+\}', text_cleaned, re.DOTALL)

        if json_match:
            result = json.loads(json_match.group(0))
            is_diverse = result.get("is_diverse", True)
            confidence = result.get("confidence", 0.5)
            reasoning = result.get("reasoning", "")

            cache[cache_key] = result
            return is_diverse, confidence, reasoning

        # Fallback: assume diverse
        cache[cache_key] = {"is_diverse": True, "confidence": 0.5, "reasoning": "Parse error"}
        return True, 0.5, "Parse error - assuming diverse"

    except Exception as e:
        print(f"\n⚠️  Diversity check failed: {e}")
        return True, 0.5, f"Error: {str(e)[:50]}"

# ========================== CONFIG ==========================
LABELING_CACHE = CACHE_ROOT / "labeling_pairwise"
LABELING_CACHE.mkdir(parents=True, exist_ok=True)

LABELS_FILE = LABELING_CACHE / "pairwise_labels.json"

def get_folder_identifier(folder_path):
    """Generate a safe identifier from folder path for cache/checkpoint files."""
    import re
    # Use folder name, sanitize for filename
    folder_name = Path(folder_path).name
    # Remove special characters, keep alphanumeric and underscore
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', folder_name)
    # Limit length to avoid overly long filenames
    return sanitized[:50]

def get_gemini_cache_file(folder_path):
    """Get folder-specific Gemini cache file path."""
    folder_id = get_folder_identifier(folder_path)
    return LABELING_CACHE / f"gemini_cache_v4_grouped_{folder_id}.json"

def get_checkpoint_file(folder_path):
    """Get folder-specific checkpoint file path."""
    folder_id = get_folder_identifier(folder_path)
    return LABELING_CACHE / f"classification_checkpoint_v4_grouped_{folder_id}.json"

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Burst handling
BURST_MODE = "grouped"
MAX_BURST_SIZE_FOR_GROUPING = 50
MAX_BURST_SIZE_PER_API_CALL = 7  # Split bursts larger than this to avoid truncation

# FIX #3: Hybrid burst routing
# Step 1: Filter by rank (best in burst)
# Step 2: Apply absolute share-worthiness threshold (like singletons)
# This prevents sharing "best of a mediocre burst"
BURST_RANK_CONSIDER_THRESHOLD = 2  # Only consider rank 1-2 for potential sharing (was 3)
BURST_RANK_REVIEW_THRESHOLD = 4     # Rank 3-4 go to Review, 5+ to Storage (was 5)

# Absolute share-worthiness thresholds (applied to ALL photos including top-ranked bursts)
# Lowered to hit 25-30% share rate (was 0.70, which gave ~20%)
SHARE_THRESHOLD = 0.60   # Must be >= 60% share-worthy (lowered from 0.70)
REVIEW_THRESHOLD = 0.35  # 35-59% → Review (lowered from 0.40)
# Below 35% → Storage

# Cost estimation - UPDATED WITH ACTUAL GEMINI 2.5 FLASH PRICING
# Source: https://ai.google.dev/gemini-api/docs/pricing (as of 2025)
COST_PER_1M_INPUT_TOKENS = 0.30   # $0.30 per 1M input tokens (text/image/video)
COST_PER_1M_OUTPUT_TOKENS = 2.50  # $2.50 per 1M output tokens (was missing!)

# Photo costs (rough estimates including prompt + output)
COST_PER_PHOTO = 0.0008  # ~$0.0008 per photo (input + output)
COST_PER_BURST_PHOTO = 0.00026  # ~$0.00026 per photo in burst (shared prompt cost)

# Video classification costs (Gemini processes ~300 tokens per second at 1 FPS)
TOKENS_PER_VIDEO_SECOND = 300  # Visual tokens per second of video
PROMPT_TOKENS_VIDEO = 1500  # Rich taste profile prompt
OUTPUT_TOKENS_VIDEO = 300  # JSON response with reasoning

# ========================== LABEL CONVERSION ==========================
def convert_pairwise_labels_to_training_examples(labels):
    """Convert pairwise labels to individual photo training examples."""
    training_examples = {}
    
    # Process pairwise comparisons
    for comp in labels.get("pairwise", []):
        choice = comp.get("choice")
        reason = comp.get("reason", "")
        comparison_type = comp.get("comparison_type", "unknown")
        
        photo_a = comp.get("photo_a")
        photo_b = comp.get("photo_b")
        
        if not choice or not photo_a or not photo_b:
            continue
        
        if choice == "both":
            training_examples[photo_a] = {
                "action": "Share",
                "confidence": 0.9,
                "reasoning": f"User chose both photos. {reason}".strip(),
                "source": "pairwise_both",
                "comparison_type": comparison_type
            }
            training_examples[photo_b] = {
                "action": "Share",
                "confidence": 0.9,
                "reasoning": f"User chose both photos. {reason}".strip(),
                "source": "pairwise_both",
                "comparison_type": comparison_type
            }
        
        elif choice == "left":
            training_examples[photo_a] = {
                "action": "Share",
                "confidence": 0.95,
                "reasoning": f"User preferred this over another photo. {reason}".strip(),
                "source": "pairwise_winner",
                "comparison_type": comparison_type
            }
            training_examples[photo_b] = {
                "action": "Storage",
                "confidence": 0.85,
                "reasoning": f"User preferred another photo over this. {reason}".strip(),
                "source": "pairwise_loser",
                "comparison_type": comparison_type
            }
        
        elif choice == "right":
            training_examples[photo_b] = {
                "action": "Share",
                "confidence": 0.95,
                "reasoning": f"User preferred this over another photo. {reason}".strip(),
                "source": "pairwise_winner",
                "comparison_type": comparison_type
            }
            training_examples[photo_a] = {
                "action": "Storage",
                "confidence": 0.85,
                "reasoning": f"User preferred another photo over this. {reason}".strip(),
                "source": "pairwise_loser",
                "comparison_type": comparison_type
            }
        
        elif choice == "neither":
            training_examples[photo_a] = {
                "action": "Storage",
                "confidence": 0.9,
                "reasoning": f"User rejected both photos. {reason}".strip(),
                "source": "pairwise_neither",
                "comparison_type": comparison_type
            }
            training_examples[photo_b] = {
                "action": "Storage",
                "confidence": 0.9,
                "reasoning": f"User rejected both photos. {reason}".strip(),
                "source": "pairwise_neither",
                "comparison_type": comparison_type
            }
    
    # Process single photo labels
    for photo_path, label in labels.get("single", {}).items():
        action = label.get("action")
        confidence = label.get("confidence", 0.5)
        reason = label.get("reason", "")
        
        training_examples[photo_path] = {
            "action": action,
            "confidence": confidence,
            "reasoning": reason,
            "source": "single",
            "comparison_type": "explicit"
        }
    
    # Process gallery selections
    for gallery in labels.get("gallery", []):
        photos = gallery.get("photos", [])
        selected_indices = gallery.get("selected_indices", [])
        reason = gallery.get("reason", "")
        
        for i, photo in enumerate(photos):
            if i in selected_indices:
                training_examples[photo] = {
                    "action": "Share",
                    "confidence": 0.95,
                    "reasoning": f"Selected in gallery review (burst of {len(photos)}). {reason}".strip(),
                    "source": "gallery_selected",
                    "comparison_type": "within"
                }
            else:
                training_examples[photo] = {
                    "action": "Storage",
                    "confidence": 0.85,
                    "reasoning": f"Not selected in gallery review (burst of {len(photos)}). {reason}".strip(),
                    "source": "gallery_rejected",
                    "comparison_type": "within"
                }
    
    return training_examples

# ========================== TASTE PROFILE LOADING ==========================
def load_generated_taste_profile():
    """Load generated taste profile if available."""
    profile_path = Path("taste_preferences_generated.json")
    if profile_path.exists():
        try:
            with open(profile_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️  Could not load taste profile: {e}")
    return None

# ========================== GEMINI PROMPT BUILDING ==========================
def build_single_photo_prompt(training_examples, share_ratio, max_examples=15):
    """Build prompt for evaluating INDIVIDUAL photos."""
    # Try to load generated taste profile first
    taste_profile_data = load_generated_taste_profile()

    taste_note = ""
    if taste_profile_data:
        # Use concise taste profile (like burst prompt)
        print("   ✅ Using generated taste profile")

        top_priorities = taste_profile_data.get("top_priorities", [])
        if top_priorities:
            taste_note = f"\n**Your taste priorities:** {', '.join(top_priorities[:4])}\n"

        share_criteria = taste_profile_data.get("share_criteria", {})
        if share_criteria.get("must_have"):
            taste_note += f"**Must have:** {', '.join(share_criteria['must_have'][:3])}\n"
        if share_criteria.get("deal_breakers"):
            reject_criteria = taste_profile_data.get("reject_criteria", {})
            taste_note += f"**Deal breakers:** {', '.join(reject_criteria.get('deal_breakers', [])[:3])}\n"

    else:
        # Fallback: Extract key themes from user's reasoning (concise)
        all_reasoning = " ".join([data.get("reasoning", "") for data in training_examples.values()])
        valued_keywords = []

        if "expression" in all_reasoning.lower():
            valued_keywords.append("expressive faces")
        if "dad" in all_reasoning.lower() or "parent" in all_reasoning.lower():
            valued_keywords.append("parent-child interaction")
        if "mischief" in all_reasoning.lower() or "joy" in all_reasoning.lower():
            valued_keywords.append("joyful moments")
        if "interaction" in all_reasoning.lower():
            valued_keywords.append("genuine interaction")

        if valued_keywords:
            taste_note = f"\n**Your taste priorities:** {', '.join(valued_keywords)}\n"

    prompt = f"""You are helping sort family photos of young children into three categories: SHARE, STORAGE, or IGNORE.
{taste_note}

**STEP 1 - FIRST CHECK: Does this photo contain young children (babies/toddlers/preschoolers)?**
- Look carefully at the ENTIRE photo
- If you see NO young children → classify as IGNORE
- Only proceed to Share/Storage if young children ARE visible

**STEP 2 - APPROPRIATENESS CHECK (CRITICAL):**
- **IGNORE if photo shows inappropriate nudity or "special parts"**
- Bath time photos showing private areas → IGNORE
- Diaper changes showing genitals → IGNORE
- Any photo you wouldn't show to grandparents → IGNORE
- **When in doubt about appropriateness → IGNORE**

**EXAMPLES OF IGNORE:**
- Food ONLY (no people) → Ignore
- Landscapes/scenery ONLY (no people) → Ignore
- Objects/documents ONLY (toys, papers) → Ignore
- Adults ONLY (no children present) → Ignore
- Inappropriate nudity or private parts → Ignore

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

**CALIBRATION:**
- In training: {share_ratio:.1%} of photos are Share-worthy
- Be selective! Not every cute photo is share-worthy

Analyze this photo and respond with CONCISE JSON:
{{
    "classification": "Share" or "Storage" or "Ignore",
    "confidence": 0.0 to 1.0,
    "reasoning": "Brief explanation (1 sentence max)",
    "contains_children": true or false,
    "is_appropriate": true or false  // Family-friendly? No inappropriate nudity?
}}
"""
    
    return prompt

def build_burst_prompt(training_examples, share_ratio, burst_size, max_examples=50):
    """Build prompt for evaluating a BURST of photos together."""
    gallery_examples = [
        (data.get("reasoning", ""), data.get("source", ""))
        for data in training_examples.values()
        if data.get("source", "").startswith("gallery")
    ]

    burst_reasonings = list(set([g[0] for g in gallery_examples if g[0]]))[:5]

    # Try to load generated taste profile first
    taste_profile_data = load_generated_taste_profile()

    taste_note = ""
    if taste_profile_data:
        # Use comprehensive profile for burst ranking
        top_priorities = taste_profile_data.get("top_priorities", [])
        if top_priorities:
            taste_note = f"\n**When ranking, prioritize:** {', '.join(top_priorities[:4])}\n"

        burst_philosophy = taste_profile_data.get("contextual_preferences", {}).get("burst_philosophy", "")
        if burst_philosophy:
            taste_note += f"**Burst approach:** {burst_philosophy}\n"
    else:
        # Fallback: Extract user taste from ALL examples
        all_reasoning = " ".join([data.get("reasoning", "") for data in training_examples.values()])
        valued_keywords = []

        if "expression" in all_reasoning.lower():
            valued_keywords.append("best expressions")
        if "dad" in all_reasoning.lower() or "parent" in all_reasoning.lower():
            valued_keywords.append("parent-child interaction quality")
        if "mischief" in all_reasoning.lower() or "joy" in all_reasoning.lower():
            valued_keywords.append("emotional quality (joy, mischief)")
        if "interaction" in all_reasoning.lower() or "together" in all_reasoning.lower():
            valued_keywords.append("subjects engaged together")

        if valued_keywords:
            taste_note = f"\n**When ranking, prioritize:** {', '.join(valued_keywords)}\n"

    prompt = f"""You are helping sort family photos of young children into three categories: SHARE, STORAGE, or IGNORE.

**BURST EVALUATION MODE:**
You are viewing {burst_size} photos taken in rapid succession.
These are very similar - slight variations of the same moment.
{taste_note}

**STEP 1 - CRITICAL FIRST CHECK:**
Do ANY of these photos contain young children (babies/toddlers/preschoolers)?
- Scan all photos carefully
- If NONE contain young children → classify ALL as IGNORE
- Examples that should be IGNORE:
  * Burst of food photos (no people)
  * Burst of landscape photos (no people)
  * Burst of object/document photos
  * Burst of adults only (no children)

**STEP 2 - APPROPRIATENESS CHECK:**
- **Flag as IGNORE if photos show inappropriate nudity or "special parts"**
- Bath time showing private areas → IGNORE
- Diaper changes showing genitals → IGNORE
- Any photo you wouldn't show to grandparents → IGNORE
- **When in doubt about appropriateness → IGNORE**

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
- Duplicates → Storage (keep only the best)

**CALIBRATION:**
- From training: {share_ratio:.1%} of ALL photos are Share-worthy
- In bursts: This means ~{int(share_ratio * 10)}-{int(share_ratio * 20)} out of 10 photos across ALL bursts
- Don't feel pressured to share from every burst!

"""
    
    if burst_reasonings:
        prompt += "\n**Gallery selection examples from training:**\n\n"
        for reasoning in burst_reasonings:
            if reasoning:
                prompt += f"- {reasoning}\n"
        prompt += "\n"
    
    prompt += f"""
You will see {burst_size} photos. Respond with CONCISE JSON (keep descriptions brief!):

{{
    "burst_analysis": "Brief overall assessment (1 sentence max)",
    "all_photos_mediocre": true or false,
    "contains_children": true or false,  // Do ANY photos contain young children?
    "is_appropriate": true or false,  // Family-friendly? No inappropriate nudity?
    "photos": [
        {{
            "index": 0,
            "classification": "Share" or "Storage" or "Ignore",
            "confidence": 0.0 to 1.0,  // ABSOLUTE share-worthiness (not ranking certainty!)
            "rank": 1 to {burst_size},  // Relative rank within THIS burst
            "reasoning": "Brief reason (10 words max)",
            "technical_quality": "3-5 words max",
            "expression_quality": "3-5 words max",
            "contains_children": true or false,
            "is_appropriate": true or false
        }},
        ... (one entry for EACH photo, in order)
    ],
    "recommended_keep_count": 0 to {burst_size},
    "burst_summary": "Brief summary (1 sentence)"
}}

**CRITICAL DECISION FLOW:**
1. Scan all photos for young children (babies/toddlers/preschoolers)
2. If NO children in ANY photo → ALL photos get classification="Ignore", contains_children=false
3. Check for inappropriate nudity → If found, classification="Ignore", is_appropriate=false
4. Only evaluate quality (Share vs Storage) if children ARE present AND photos are appropriate

**Examples:**
- Burst of 5 food photos (no people) → ALL 5 marked as Ignore
- Burst of 3 landscape photos → ALL 3 marked as Ignore
- Burst of bath time with private parts showing → ALL marked as Ignore (inappropriate)
- Burst with 4 appropriate child photos → Evaluate quality, rank them

**CRITICAL - CONFIDENCE INTERPRETATION:**
- Confidence = ABSOLUTE share-worthiness (0.0-1.0)
- Ask: "Would I share this photo if it appeared alone?"
- NOT: "How confident am I in the ranking?"
- Examples:
  * Rank 1, confidence 0.4 = "Best of burst, but burst is mediocre"
  * Rank 1, confidence 0.95 = "Best of burst AND absolutely share-worthy"

**REMEMBER:**
- KEEP DESCRIPTIONS BRIEF to avoid truncation!
- It's OK to recommend 0 photos if burst is entirely mediocre
- Rank honestly - don't give multiple photos the same rank
- Be selective - typically only 0-2 photos per burst
"""
    
    return prompt

# ========================== GEMINI CLASSIFICATION ==========================
def classify_single_photo_with_gemini(image_path, prompt, cache):
    """Classify a SINGLE photo."""
    cache_key = f"single_{get_cache_key(Path(image_path))}"
    
    if cache_key in cache:
        return cache[cache_key]
    
    if not GEMINI_API_KEY:
        return {"error": "No API key"}
    
    try:
        img = fix_image_orientation(image_path, max_size=1024)
        if img is None:
            raise ValueError("Could not load image")
        
        # FIX #2: Use correct Gemini version
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        query = prompt + "\n\nNow analyze this photo:"
        
        import time
        time.sleep(0.15)

        response = model.generate_content(
            [query, img],
            generation_config={'max_output_tokens': 4096}
        )

        # Validate response before accessing text
        if not response.candidates:
            print(f"\n⚠️  No response candidates (likely blocked by safety filters)")
            if hasattr(response, 'prompt_feedback'):
                print(f"   Prompt feedback: {response.prompt_feedback}")
            return {
                "classification": "Review",
                "confidence": 0.3,
                "reasoning": "Response blocked by safety filters",
                "contains_children": True,
                "is_appropriate": False
            }

        candidate = response.candidates[0]
        if candidate.finish_reason != 1:  # 1 = STOP (normal completion)
            print(f"\n⚠️  Unusual finish reason: {candidate.finish_reason}")
            if hasattr(candidate, 'safety_ratings'):
                print(f"   Safety ratings: {candidate.safety_ratings}")
            return {
                "classification": "Review",
                "confidence": 0.3,
                "reasoning": f"Finish reason: {candidate.finish_reason}",
                "contains_children": True,
                "is_appropriate": False
            }

        text = response.text.strip()

        # Parse JSON - handle markdown code fences
        import re

        # Remove markdown code fences (both ``` and ''')
        text_cleaned = re.sub(r'^```json\s*\n', '', text, flags=re.MULTILINE)
        text_cleaned = re.sub(r'^```\s*\n', '', text_cleaned, flags=re.MULTILINE)
        text_cleaned = re.sub(r'\n```\s*$', '', text_cleaned, flags=re.MULTILINE)
        text_cleaned = re.sub(r"^'''json\s*\n", '', text_cleaned, flags=re.MULTILINE)
        text_cleaned = re.sub(r"^'''\s*\n", '', text_cleaned, flags=re.MULTILINE)
        text_cleaned = re.sub(r"\n'''\s*$", '', text_cleaned, flags=re.MULTILINE)
        text_cleaned = text_cleaned.strip()  # Remove all whitespace

        json_match = re.search(r'\{[^}]+\}', text_cleaned, re.DOTALL)

        if json_match:
            try:
                result = json.loads(json_match.group(0))
                result["raw_response"] = text

                # Normalize classification
                cls = result.get("classification", "").strip()
                if cls.upper() == "SHARE":
                    result["classification"] = "Share"
                elif cls.upper() == "STORAGE":
                    result["classification"] = "Storage"
                elif cls.upper() == "IGNORE":
                    result["classification"] = "Ignore"
                else:
                    result["classification"] = "Review"
                    result["confidence"] = 0.3
                    result["reasoning"] = f"Could not parse: {cls}"

                cache[cache_key] = result
                return result
            except json.JSONDecodeError as e:
                print(f"\n⚠️  JSON decode error for {Path(image_path).name}: {e}")
                print(f"   Response (first 300 chars): {text[:300]}")
        
        # Fallback
        classification = "Storage"
        confidence = 0.5
        if "share" in text.lower():
            classification = "Share"
            confidence = 0.7
        
        result = {
            "classification": classification,
            "confidence": confidence,
            "reasoning": text[:200],
            "raw_response": text
        }
        
        cache[cache_key] = result
        return result
    
    except Exception as e:
        print(f"\n⚠️  ERROR {Path(image_path).name}: {str(e)[:100]}")
        return {
            "error": str(e),
            "classification": "Storage",
            "confidence": 0.3,
            "reasoning": f"Error: {str(e)[:100]}"
        }

# ========================== VIDEO CLASSIFICATION ==========================
def get_video_duration(video_path):
    """Get video duration in seconds (rough estimate from file size)."""
    try:
        # For more accurate duration, would need to parse video metadata
        # For now, use rough estimate: ~1-2 MB per second for typical home videos
        file_size_mb = video_path.stat().st_size / (1024 * 1024)
        estimated_seconds = file_size_mb / 1.5  # Conservative estimate
        return max(1, int(estimated_seconds))
    except:
        return 30  # Default to 30 seconds if can't determine

def build_video_prompt(training_examples, share_ratio):
    """Build prompt for evaluating VIDEO content with full taste profile integration."""
    # Load taste profile
    taste_profile_data = load_generated_taste_profile()

    taste_profile = ""
    if taste_profile_data:
        # Use FULL generated taste profile (like photo prompt)
        print("   ✅ Using generated taste profile for videos")

        taste_profile = "\n**USER'S TASTE PROFILE (AI-generated from photo training):**\n"
        taste_profile += f"Philosophy: {taste_profile_data.get('share_philosophy', '')}\n\n"

        top_priorities = taste_profile_data.get("top_priorities", [])
        if top_priorities:
            taste_profile += f"**Top Priorities (apply to videos):**\n"
            for i, p in enumerate(top_priorities[:5], 1):
                taste_profile += f"{i}. {p}\n"
            taste_profile += "\n"

        share_criteria = taste_profile_data.get("share_criteria", {})
        if share_criteria.get("must_have"):
            taste_profile += f"**Must Have:** {', '.join(share_criteria['must_have'][:3])}\n"
        if share_criteria.get("highly_valued"):
            taste_profile += f"**Highly Valued:** {', '.join(share_criteria['highly_valued'][:3])}\n"

        reject_criteria = taste_profile_data.get("reject_criteria", {})
        if reject_criteria.get("deal_breakers"):
            taste_profile += f"**Deal Breakers:** {', '.join(reject_criteria['deal_breakers'][:3])}\n"

        specific_guidance = taste_profile_data.get("specific_guidance", [])
        if specific_guidance:
            taste_profile += f"\n**Specific Guidance (adapted for video):**\n"
            for g in specific_guidance[:3]:
                taste_profile += f"• {g}\n"

        # Add video-specific context
        taste_profile += f"\n**VIDEO-SPECIFIC CONSIDERATIONS:**\n"
        taste_profile += f"• Audio matters: Joyful sounds, laughter, and engaged conversation enhance moments\n"
        taste_profile += f"• Motion reveals personality: Watch for natural gestures, expressions changing over time\n"
        taste_profile += f"• Interaction quality: Videos show the FLOW of interaction, not just a single moment\n"
        taste_profile += f"• Avoid: Excessive crying/screaming, very shaky footage, can't see faces clearly\n"

    else:
        # Fallback: Extract key themes from ALL training examples
        all_share_reasoning = " ".join([data.get("reasoning", "") for _, data in training_examples.items()
                                       if data.get("action") == "Share"])
        all_storage_reasoning = " ".join([data.get("reasoning", "") for _, data in training_examples.items()
                                         if data.get("action") == "Storage"])

        # Analyze patterns
        valued_keywords = []
        reject_keywords = []

        if "expression" in all_share_reasoning.lower():
            valued_keywords.append("expressive, engaged faces and reactions")
        if "dad" in all_share_reasoning.lower() or "parent" in all_share_reasoning.lower() or "mom" in all_share_reasoning.lower():
            valued_keywords.append("quality parent-child interaction")
        if "mischief" in all_share_reasoning.lower() or "joy" in all_share_reasoning.lower():
            valued_keywords.append("joyful, mischievous, genuine emotional moments")
        if "interaction" in all_share_reasoning.lower() or "together" in all_share_reasoning.lower():
            valued_keywords.append("subjects engaged together")

        if "can't see" in all_storage_reasoning.lower() or "can't tell" in all_storage_reasoning.lower():
            reject_keywords.append("unclear subject or can't see faces")
        if "vacant" in all_storage_reasoning.lower() or "lame" in all_storage_reasoning.lower():
            reject_keywords.append("poor expressions from parents")
        if "not special" in all_storage_reasoning.lower() or "not great" in all_storage_reasoning.lower():
            reject_keywords.append("ordinary moments without special quality")

        if valued_keywords or reject_keywords:
            taste_profile = "\n**USER'S SPECIFIC TASTE (learned from photo examples):**\n"
            if valued_keywords:
                taste_profile += f"✓ Values in videos: {', '.join(valued_keywords)}\n"
            if reject_keywords:
                taste_profile += f"✗ Rejects in videos: {', '.join(reject_keywords)}\n"

            taste_profile += f"\n**VIDEO-SPECIFIC:**\n"
            taste_profile += f"• Audio quality matters: joyful sounds vs. excessive crying\n"
            taste_profile += f"• Video quality matters: stable and clear vs. very shaky and dark\n"

    prompt = f"""You are helping sort family VIDEOS of young children into three categories: SHARE, STORAGE, or IGNORE.

**VIDEO EVALUATION - Watch the entire video carefully, considering both visual AND audio content**

**STEP 1 - FIRST CHECK: Does this video contain young children (babies/toddlers/preschoolers)?**
- Watch the ENTIRE video from start to finish
- If you see NO young children throughout → classify as IGNORE
- Only proceed to Share/Storage if young children ARE visible

**STEP 2 - APPROPRIATENESS CHECK (CRITICAL):**
- **IGNORE if video shows inappropriate nudity or "special parts"**
- Bath time showing private areas → IGNORE
- Diaper changes showing genitals → IGNORE
- Any video you wouldn't show to grandparents → IGNORE
- **When in doubt about appropriateness → IGNORE**

**EXAMPLES OF IGNORE:**
- Food prep videos (no people) → Ignore
- Scenery/landscape videos (no people) → Ignore
- Adults only videos (no children) → Ignore
- Screen recordings, work-related content → Ignore
- Inappropriate nudity or private parts → Ignore

**ONLY if young children ARE visible AND video is family-appropriate, evaluate share-worthiness:**

{taste_profile}

**SHARE = Videos worth sharing with family**
- Shows child's personality, genuine expression, or special moment UNFOLDING
- IMPORTANT: Parent/adult expressions and engagement matter throughout video
- Good audio: Joyful sounds, laughter, engaged conversation (not just constant crying/screaming)
- Decent video quality: Stable enough to see faces clearly, adequate lighting
- Captures genuine interaction or emotional moments OVER TIME
- Natural, candid moments rather than posed/staged
- Something memorable is actually HAPPENING (not just child sitting there)

**STORAGE = Videos of children that aren't share-worthy**
- Poor video quality: Too shaky to watch, too dark to see faces, out of focus
- Poor audio: Excessive crying/screaming/tantrum throughout entire video
- Boring/uneventful: Nothing interesting happening, just child sitting/standing
- Too long with no clear focus or special moment
- Parent expressions are poor/vacant/disengaged throughout
- Can't see child's face or reactions clearly
- Still contains children, just not worth sharing

**EVALUATE ACROSS THE FULL VIDEO:**
Unlike photos (single frozen moment), videos show moments UNFOLDING:
- Does something interesting HAPPEN? (first steps, funny reaction, discovery moment)
- Do expressions and interactions EVOLVE? (building up to laughter, surprise reaction)
- Is there a STORY or moment captured, or just random footage?
- Are audio and visual elements TOGETHER creating something shareable?

**CALIBRATION:**
- In training data: {share_ratio:.1%} of photos are Share-worthy
- Apply similar selectivity to videos - not every cute video is share-worthy
- Videos showing genuine personality/moments > videos of child just existing
- Parent engagement visible in video > parent filming passively

Analyze this video and respond with CONCISE JSON:
{{
    "classification": "Share" or "Storage" or "Ignore",
    "confidence": 0.0 to 1.0,
    "reasoning": "Brief explanation focusing on WHY (1-2 sentences)",
    "contains_children": true or false,
    "is_appropriate": true or false,  // Family-friendly? No inappropriate nudity?
    "audio_quality": "good" or "poor" or "n/a",  // Joyful/engaging vs. excessive crying?
    "video_quality": "good" or "poor"  // Stable/clear vs. shaky/dark?
}}
"""

    return prompt

def classify_videos_parallel(videos, video_prompt, gemini_cache, folder_path, max_workers=10):
    """Classify videos in parallel using ThreadPoolExecutor for 10-20x speedup."""
    import concurrent.futures
    from threading import Lock

    cache_lock = Lock()
    video_results = []
    results_lock = Lock()
    completed_count = [0]  # Use list to allow modification in nested function

    def process_single_video(video_info):
        i, video_path = video_info

        # Classify video (cache check happens inside this function)
        result = classify_video_with_gemini(video_path, video_prompt, gemini_cache, cache_lock)

        result["path"] = str(video_path)
        result["burst_index"] = -1
        result["burst_size"] = 1
        result["is_video"] = True

        bucket, routing_reason = route_photo_burst_aware(result, is_burst=False)
        result["bucket"] = bucket
        result["routing_reason"] = routing_reason

        # Prepare output
        cls = result.get("classification", "Unknown")
        conf = result.get("confidence", 0)
        audio = result.get("audio_quality", "n/a")
        video_q = result.get("video_quality", "unknown")
        reasoning = result.get("reasoning", "No reasoning provided")
        conf_str = f"{conf:.2f}" if conf is not None else "N/A"

        # Thread-safe updates
        with results_lock:
            video_results.append(result)
            completed_count[0] += 1
            count = completed_count[0]

        print(f"\n   Video {count}/{len(videos)}: {video_path.name}")
        print(f"      {cls:7s} → {bucket:7s} ({conf_str}) | Audio: {audio}, Video: {video_q}")
        print(f"      Reasoning: {reasoning}")

        # Save cache periodically
        if count % 5 == 0:
            with cache_lock:
                save_gemini_cache(gemini_cache, folder_path)

        return result

    # Create list of (index, video_path) tuples
    video_list = [(i+1, v) for i, v in enumerate(videos)]

    # Process in parallel
    print(f"\n🚀 Processing {len(videos)} videos with {max_workers} parallel workers...")
    print(f"   Expected speedup: {max_workers}x faster than sequential processing")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks and process as they complete
        futures = [executor.submit(process_single_video, video_info) for video_info in video_list]

        # Wait for all to complete
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()  # Raise any exceptions that occurred
            except Exception as e:
                print(f"\n⚠️  Error processing video: {e}")

    # Final save
    with cache_lock:
        save_gemini_cache(gemini_cache, folder_path)

    print(f"\n   ✅ {len(videos)} videos classified")

    return video_results

def classify_video_with_gemini(video_path, prompt, cache, cache_lock=None):
    """Classify a single VIDEO using Gemini Files API."""
    cache_key = f"video_{get_cache_key(Path(video_path))}"

    # Thread-safe cache check
    if cache_lock:
        with cache_lock:
            if cache_key in cache:
                return cache[cache_key]
    else:
        if cache_key in cache:
            return cache[cache_key]

    if not GEMINI_API_KEY:
        return {"error": "No API key"}

    video_file = None
    try:
        print(f"      Uploading video to Gemini...")

        # Upload video via Files API
        video_file = genai.upload_file(path=str(video_path))

        # Wait for processing to complete
        max_wait = 60  # Max 60 seconds
        start_time = time.time()
        while video_file.state.name == "PROCESSING":
            if time.time() - start_time > max_wait:
                raise Exception("Video processing timeout")
            time.sleep(2)
            video_file = genai.get_file(video_file.name)

        if video_file.state.name == "FAILED":
            raise Exception("Video processing failed")

        print(f"      Video processed, analyzing...")

        # FIX #2: Use correct Gemini version
        model = genai.GenerativeModel('gemini-2.5-flash')

        query = prompt + "\n\nNow analyze this video:"

        time.sleep(0.3)

        response = model.generate_content(
            [query, video_file],
            generation_config={'max_output_tokens': 4096}
        )

        # Validate response
        if not response.candidates:
            print(f"\n⚠️  No response candidates (likely blocked by safety filters)")
            result = {
                "classification": "Review",
                "confidence": 0.3,
                "reasoning": "Response blocked by safety filters",
                "contains_children": True,
                "is_appropriate": False
            }
            if cache_lock:
                with cache_lock:
                    cache[cache_key] = result
            else:
                cache[cache_key] = result
            return result

        candidate = response.candidates[0]
        if candidate.finish_reason != 1:
            print(f"\n⚠️  Unusual finish reason: {candidate.finish_reason}")
            result = {
                "classification": "Review",
                "confidence": 0.3,
                "reasoning": f"Finish reason: {candidate.finish_reason}",
                "contains_children": True,
                "is_appropriate": False
            }
            if cache_lock:
                with cache_lock:
                    cache[cache_key] = result
            else:
                cache[cache_key] = result
            return result

        text = response.text.strip()

        # Parse JSON
        import re
        text_cleaned = re.sub(r'^```json\s*\n', '', text, flags=re.MULTILINE)
        text_cleaned = re.sub(r'^```\s*\n', '', text_cleaned, flags=re.MULTILINE)
        text_cleaned = re.sub(r'\n```\s*$', '', text_cleaned, flags=re.MULTILINE)
        text_cleaned = text_cleaned.strip()

        json_match = re.search(r'\{[^}]+\}', text_cleaned, re.DOTALL)

        if json_match:
            try:
                result = json.loads(json_match.group(0))

                # Validate and fix classification
                cls = result.get("classification", "Storage")
                if cls not in ["Share", "Storage", "Ignore", "Review"]:
                    if "ignore" in cls.lower():
                        result["classification"] = "Ignore"
                    else:
                        result["classification"] = "Review"
                        result["confidence"] = 0.3
                        result["reasoning"] = f"Could not parse: {cls}"

                if cache_lock:
                    with cache_lock:
                        cache[cache_key] = result
                else:
                    cache[cache_key] = result
                return result
            except json.JSONDecodeError as e:
                print(f"\n⚠️  JSON decode error for {Path(video_path).name}: {e}")
                print(f"   Response (first 300 chars): {text[:300]}")

        # Fallback
        classification = "Storage"
        confidence = 0.5
        if "share" in text.lower():
            classification = "Share"
            confidence = 0.7

        result = {
            "classification": classification,
            "confidence": confidence,
            "reasoning": text[:200],
            "raw_response": text
        }

        if cache_lock:
            with cache_lock:
                cache[cache_key] = result
        else:
            cache[cache_key] = result
        return result

    except Exception as e:
        print(f"\n⚠️  ERROR processing video {Path(video_path).name}: {str(e)[:100]}")
        return {
            "error": str(e),
            "classification": "Storage",
            "confidence": 0.3,
            "reasoning": f"Error: {str(e)[:100]}"
        }
    finally:
        # Cleanup: Delete uploaded file from Gemini
        if video_file:
            try:
                genai.delete_file(video_file.name)
            except:
                pass  # Ignore cleanup errors

def classify_burst_with_gemini_chunked(burst_paths, prompt, cache):
    """
    Classify a burst, splitting into chunks if too large to avoid truncation.
    """
    # If burst is small enough, process normally
    if len(burst_paths) <= MAX_BURST_SIZE_PER_API_CALL:
        return classify_burst_with_gemini(burst_paths, prompt, cache)

    # Split large burst into chunks
    print(f"      ⚠️  Large burst ({len(burst_paths)} photos) - splitting into chunks of {MAX_BURST_SIZE_PER_API_CALL}")

    chunk_size = MAX_BURST_SIZE_PER_API_CALL
    chunks = [burst_paths[i:i + chunk_size] for i in range(0, len(burst_paths), chunk_size)]

    all_photos_data = []

    # Process each chunk
    for chunk_idx, chunk in enumerate(chunks):
        print(f"         Processing chunk {chunk_idx + 1}/{len(chunks)} ({len(chunk)} photos)...")

        # Build prompt for this chunk
        chunk_prompt = build_burst_prompt(
            {},  # No training examples needed here, already in outer prompt
            0.3,  # Share ratio - doesn't matter for chunk
            len(chunk)
        )

        chunk_result = classify_burst_with_gemini(chunk, chunk_prompt, cache)
        chunk_photos = chunk_result.get("photos", [])

        # Adjust indices to match position in full burst
        offset = chunk_idx * chunk_size
        for photo_data in chunk_photos:
            photo_data["index"] = photo_data.get("index", 0) + offset
            photo_data["rank"] = photo_data.get("rank", 0) + offset  # Temporary rank
            all_photos_data.append(photo_data)

    # Re-rank across all photos based on confidence
    all_photos_data.sort(key=lambda x: x.get("confidence", 0), reverse=True)
    for new_rank, photo_data in enumerate(all_photos_data, start=1):
        photo_data["rank"] = new_rank

    # Sort back by index to maintain order
    all_photos_data.sort(key=lambda x: x.get("index", 0))

    # Combine into final result
    result = {
        "burst_analysis": f"Large burst ({len(burst_paths)} photos) processed in {len(chunks)} chunks",
        "all_photos_mediocre": all(p.get("confidence", 0) < 0.6 for p in all_photos_data),
        "contains_children": any(p.get("contains_children", True) for p in all_photos_data),
        "photos": all_photos_data,
        "recommended_keep_count": sum(1 for p in all_photos_data if p.get("confidence", 0) >= 0.7),
        "burst_summary": f"Processed in chunks, re-ranked globally"
    }

    # Cache the combined result
    burst_cache_key = f"burst_{'_'.join([get_cache_key(p)[:8] for p in burst_paths])}"
    cache[burst_cache_key] = result

    return result

def classify_burst_with_gemini(burst_paths, prompt, cache):
    """Classify an ENTIRE BURST together (internal - use classify_burst_with_gemini_chunked for public API)."""
    burst_cache_key = f"burst_{'_'.join([get_cache_key(p)[:8] for p in burst_paths])}"

    if burst_cache_key in cache:
        return cache[burst_cache_key]

    if not GEMINI_API_KEY:
        return {"error": "No API key"}

    try:
        # Load all images
        images = []
        for path in burst_paths:
            img = fix_image_orientation(path, max_size=800)
            if img is None:
                raise ValueError(f"Could not load {path.name}")
            images.append(img)

        # FIX #2: Use correct Gemini version
        model = genai.GenerativeModel('gemini-2.5-flash')

        query = prompt + f"\n\nAnalyze these {len(burst_paths)} photos:\n\n"
        for i, path in enumerate(burst_paths):
            query += f"Photo {i}: {path.name}\n"
        query += "\nProvide your analysis in JSON format."

        content = [query] + images

        import time
        time.sleep(0.3)

        response = model.generate_content(
            content,
            generation_config={'max_output_tokens': 4096}
        )

        # Validate response before accessing text
        if not response.candidates:
            print(f"\n⚠️  No response candidates (likely blocked by safety filters)")
            if hasattr(response, 'prompt_feedback'):
                print(f"   Prompt feedback: {response.prompt_feedback}")
            # Return fallback for entire burst
            result = {
                "burst_analysis": "Response blocked by safety filters",
                "all_photos_mediocre": False,
                "contains_children": True,
                "is_appropriate": False,
                "photos": [
                    {
                        "index": i,
                        "classification": "Review",
                        "confidence": 0.3,
                        "rank": i + 1,
                        "reasoning": "Safety filter blocked",
                        "technical_quality": "Unknown",
                        "expression_quality": "Unknown",
                        "contains_children": True,
                        "is_appropriate": False
                    }
                    for i in range(len(burst_paths))
                ],
                "recommended_keep_count": 0,
                "burst_summary": "Response blocked by safety filters"
            }
            cache[burst_cache_key] = result
            return result

        candidate = response.candidates[0]
        if candidate.finish_reason != 1:  # 1 = STOP (normal completion)
            print(f"\n⚠️  Unusual finish reason: {candidate.finish_reason}")
            if hasattr(candidate, 'safety_ratings'):
                print(f"   Safety ratings: {candidate.safety_ratings}")
            # Return fallback for entire burst
            result = {
                "burst_analysis": f"Unusual finish reason: {candidate.finish_reason}",
                "all_photos_mediocre": False,
                "contains_children": True,
                "is_appropriate": False,
                "photos": [
                    {
                        "index": i,
                        "classification": "Review",
                        "confidence": 0.3,
                        "rank": i + 1,
                        "reasoning": f"Finish reason: {candidate.finish_reason}",
                        "technical_quality": "Unknown",
                        "expression_quality": "Unknown",
                        "contains_children": True,
                        "is_appropriate": False
                    }
                    for i in range(len(burst_paths))
                ],
                "recommended_keep_count": 0,
                "burst_summary": f"Response issue: finish_reason={candidate.finish_reason}"
            }
            cache[burst_cache_key] = result
            return result

        text = response.text.strip()

        # FIX: Better JSON extraction - handle markdown code fences FIRST
        import re

        # Remove markdown code fences (both ``` and ''')
        # Handle: ```json ... ```, '''json ... ''', or plain JSON
        text_cleaned = re.sub(r'^```json\s*\n', '', text, flags=re.MULTILINE)
        text_cleaned = re.sub(r'^```\s*\n', '', text_cleaned, flags=re.MULTILINE)
        text_cleaned = re.sub(r'\n```\s*$', '', text_cleaned, flags=re.MULTILINE)
        text_cleaned = re.sub(r"^'''json\s*\n", '', text_cleaned, flags=re.MULTILINE)
        text_cleaned = re.sub(r"^'''\s*\n", '', text_cleaned, flags=re.MULTILINE)
        text_cleaned = re.sub(r"\n'''\s*$", '', text_cleaned, flags=re.MULTILINE)
        text_cleaned = text_cleaned.strip()  # Remove all leading/trailing whitespace

        # Check for response truncation AFTER cleaning
        is_truncated = False
        if text_cleaned and not text_cleaned.endswith('}'):
            is_truncated = True
            print(f"\n⚠️  Response truncated (doesn't end with '}}'):")
            print(f"   Last 100 chars: ...{text_cleaned[-100:]}")
            # Try to close the JSON by finding the last complete photo entry
            # Look for pattern: }] which closes the photos array
            photos_array_end = text_cleaned.rfind('}]')
            if photos_array_end > 0:
                # Close after the photos array
                text_cleaned = text_cleaned[:photos_array_end+2] + ', "recommended_keep_count": 0, "burst_summary": "Response truncated"}'
                print(f"   Attempted to recover JSON by closing at photos array")
            else:
                print(f"   Could not recover - will use fallback")

        # Try to extract JSON object
        json_match = re.search(r'\{.*\}', text_cleaned, re.DOTALL)

        if json_match:
            json_str = json_match.group(0)

            # Try multiple JSON parsing strategies
            result = None
            parse_attempts = []

            # Attempt 1: Parse as-is
            try:
                result = json.loads(json_str)
                parse_attempts.append("direct parse")
            except json.JSONDecodeError as e:
                parse_attempts.append(f"direct parse failed: {str(e)[:50]}")

                # Attempt 2: Fix common issues
                try:
                    # Remove trailing commas before closing braces/brackets
                    json_fixed = re.sub(r',(\s*[}\]])', r'\1', json_str)
                    # Remove comments (// or #)
                    json_fixed = re.sub(r'//.*?\n', '\n', json_fixed)
                    json_fixed = re.sub(r'#.*?\n', '\n', json_fixed)
                    result = json.loads(json_fixed)
                    parse_attempts.append("fixed trailing commas")
                except json.JSONDecodeError as e2:
                    parse_attempts.append(f"trailing comma fix failed: {str(e2)[:50]}")

                    # Attempt 3: Try to find and parse just the complete parts
                    try:
                        # Find the last complete photo entry
                        last_complete = json_str.rfind('"reasoning"')
                        if last_complete > 0:
                            # Find the closing brace for this photo entry
                            next_brace = json_str.find('}', last_complete)
                            if next_brace > 0:
                                # Reconstruct JSON with complete photos only
                                reconstructed = json_str[:next_brace+1] + '], "recommended_keep_count": 0, "burst_summary": "Partial parse"}'
                                result = json.loads(reconstructed)
                                parse_attempts.append("reconstructed from partial")
                    except Exception as e3:
                        parse_attempts.append(f"reconstruction failed: {str(e3)[:50]}")

            if result:
                result["raw_response"] = text
                result["parse_attempts"] = parse_attempts

                photos_data = result.get("photos", [])

                # Mark if response was truncated
                if is_truncated:
                    result["response_truncated"] = True
                    print(f"   ⚠️  Response was truncated, got {len(photos_data)}/{len(burst_paths)} photos")

                # FIX: Better error reporting for mismatched photo counts
                if len(photos_data) != len(burst_paths):
                    print(f"\n⚠️  Mismatch: Expected {len(burst_paths)} entries, got {len(photos_data)}")
                    print(f"   Burst files: {[p.name for p in burst_paths]}")
                    print(f"   Gemini returned data for indices: {[p.get('index', '?') for p in photos_data]}")
                    print(f"   Creating placeholders for missing photos...")

                    # FIX: Ensure we have data for ALL photos
                    # Create a mapping of index -> photo_data
                    index_to_data = {p.get("index", i): p for i, p in enumerate(photos_data)}

                    # Build complete photos list with placeholders
                    complete_photos = []
                    for i in range(len(burst_paths)):
                        if i in index_to_data:
                            complete_photos.append(index_to_data[i])
                        else:
                            # Placeholder for missing photo
                            print(f"      Missing data for photo {i}: {burst_paths[i].name}")
                            complete_photos.append({
                                "index": i,
                                "classification": "Review",  # Changed from Storage to Review
                                "confidence": 0.5,
                                "rank": len(burst_paths),
                                "reasoning": "Gemini did not return data for this photo",
                                "technical_quality": "Unknown",
                                "expression_quality": "Unknown"
                            })

                    result["photos"] = complete_photos
                    photos_data = complete_photos

                # Normalize classifications
                for photo_data in photos_data:
                    cls = photo_data.get("classification", "").strip()
                    if cls.upper() == "SHARE":
                        photo_data["classification"] = "Share"
                    elif cls.upper() == "STORAGE":
                        photo_data["classification"] = "Storage"
                    elif cls.upper() == "IGNORE":
                        photo_data["classification"] = "Ignore"
                    elif cls.upper() == "REVIEW":
                        photo_data["classification"] = "Review"
                    else:
                        photo_data["classification"] = "Storage"
                        photo_data["confidence"] = 0.3

                cache[burst_cache_key] = result
                return result
            else:
                # All parse attempts failed
                print(f"\n⚠️  Parse error: All {len(parse_attempts)} parse attempts failed")
                for i, attempt in enumerate(parse_attempts, 1):
                    print(f"   Attempt {i}: {attempt}")
                print(f"   Response (first 500 chars): {text[:500]}")
                print(f"   Response (last 200 chars): ...{text[-200:]}")
                if json_match:
                    print(f"   Extracted JSON (first 300 chars): {json_match.group(0)[:300]}")
        else:
            print(f"\n⚠️  Could not find JSON in response")
            print(f"   Response: {text[:500]}")

        # Fallback
        print(f"   Creating fallback response for {len(burst_paths)} photos...")
        result = {
            "burst_analysis": "Parse error - could not extract valid JSON",
            "all_photos_mediocre": False,
            "photos": [
                {
                    "index": i,
                    "classification": "Review",  # Changed from Storage to Review for safety
                    "confidence": 0.3,
                    "rank": i + 1,
                    "reasoning": "Parse error",
                    "technical_quality": "Unknown",
                    "expression_quality": "Unknown"
                }
                for i in range(len(burst_paths))
            ],
            "recommended_keep_count": 0,
            "burst_summary": f"Error: Could not parse response",
            "raw_response": text
        }

        cache[burst_cache_key] = result
        return result

    except Exception as e:
        print(f"\n⚠️  ERROR: {str(e)[:100]}")
        import traceback
        print(f"   Traceback: {traceback.format_exc()[:300]}")
        return {
            "error": str(e),
            "burst_analysis": "Error",
            "all_photos_mediocre": False,
            "photos": [
                {
                    "index": i,
                    "classification": "Review",  # Changed from Storage to Review for safety
                    "confidence": 0.3,
                    "rank": i + 1,
                    "reasoning": f"Error: {str(e)[:50]}",
                    "technical_quality": "Unknown",
                    "expression_quality": "Unknown"
                }
                for i in range(len(burst_paths))
            ],
            "recommended_keep_count": 0,
            "burst_summary": f"Error: {str(e)[:100]}"
        }

# ========================== UTILITIES ==========================
def load_labels():
    """Load pairwise training labels."""
    if not LABELS_FILE.exists():
        print("❌ No training labels found!")
        return None
    
    with open(LABELS_FILE, 'r') as f:
        return json.load(f)

def load_gemini_cache(folder_path):
    """Load folder-specific Gemini cache."""
    cache_file = get_gemini_cache_file(folder_path)
    if cache_file.exists():
        with open(cache_file, 'r') as f:
            return json.load(f)
    return {}

def save_gemini_cache(cache, folder_path):
    """Save folder-specific Gemini cache."""
    cache_file = get_gemini_cache_file(folder_path)
    with open(cache_file, 'w') as f:
        json.dump(cache, f, indent=2)

def load_checkpoint(folder_path):
    """Load folder-specific classification checkpoint."""
    checkpoint_file = get_checkpoint_file(folder_path)
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r') as f:
            return json.load(f)
    return {"processed_singles": [], "processed_bursts": [], "results": []}

def save_checkpoint(checkpoint, folder_path):
    """Save folder-specific classification checkpoint."""
    checkpoint_file = get_checkpoint_file(folder_path)
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint, f, indent=2)

# ========================== ROUTING ==========================
def route_photo_burst_aware(photo_data, is_burst=False):
    """
    HYBRID routing for bursts: rank filter + absolute threshold.

    For bursts:
    - Step 0: If Ignore → route to Ignore folder
    - Step 1: Filter by rank (only top 1-3 considered for sharing)
    - Step 2: Apply ABSOLUTE share-worthiness threshold (same as singletons)
    - This prevents "best of a mediocre burst" from being shared

    For singletons: Direct absolute threshold-based routing.
    """
    cls = photo_data["classification"]
    conf = photo_data.get("confidence", 0.5)
    rank = photo_data.get("rank", 999)

    # Handle Ignore classification
    if cls == "Ignore":
        return "Ignore", "No children/babies in photo"

    if is_burst:
        # STEP 1: Rank filter - only consider top photos
        if rank <= BURST_RANK_CONSIDER_THRESHOLD:
            # Top 1-2 photos: Apply ABSOLUTE share-worthiness threshold
            # Must meet BOTH criteria: (1) top-ranked AND (2) absolutely share-worthy

            if cls == "Share":
                # Gemini says Share - now check absolute confidence
                if conf >= SHARE_THRESHOLD:
                    return "Share", f"Rank {rank} + absolutely share-worthy ({conf:.0%})"
                elif conf >= REVIEW_THRESHOLD:
                    return "Review", f"Rank {rank} but uncertain share-worthiness ({conf:.0%})"
                else:
                    return "Storage", f"Rank {rank} but not absolutely shareable ({conf:.0%})"
            else:
                # Gemini says Storage despite being top-ranked
                # Could be "best of a bad burst"
                if rank == 1 and conf < REVIEW_THRESHOLD:
                    # Rank 1 but classified Storage with low confidence - review to be safe
                    return "Review", f"Rank {rank} but classified Storage, uncertain ({conf:.0%})"
                else:
                    # Trust Gemini's Storage classification
                    return "Storage", f"Rank {rank} but classified Storage ({conf:.0%})"

        elif rank <= BURST_RANK_REVIEW_THRESHOLD:
            # Rank 3-4: Middle of burst, send to Review only if classified Share
            if cls == "Share" and conf >= REVIEW_THRESHOLD:
                return "Review", f"Rank {rank} (middle tier, potential share)"
            else:
                return "Storage", f"Rank {rank} (middle tier)"
        else:
            # Rank 5+: Lower quality in burst
            return "Storage", f"Rank {rank} (lower tier in burst)"

    else:
        # SINGLETONS: Apply absolute threshold directly
        # Confidence here means "absolute share-worthiness"
        if cls == "Share":
            if conf >= SHARE_THRESHOLD:
                return "Share", f"Absolutely share-worthy ({conf:.0%})"
            elif conf >= REVIEW_THRESHOLD:
                return "Review", f"Share but uncertain ({conf:.0%})"
            else:
                return "Storage", f"Share classification but low confidence ({conf:.0%})"
        elif cls == "Storage":
            if conf >= SHARE_THRESHOLD:
                return "Storage", f"High-confidence Storage ({conf:.0%})"
            else:
                return "Review", f"Uncertain classification ({conf:.0%})"
        else:  # cls == "Ignore" (shouldn't reach here for bursts, but handle it)
            return "Ignore", "No children/babies in photo"

# ========================== BURST DETECTION & GROUPING ==========================
def organize_photos_into_bursts(photos, embeddings):
    """Organize photos into bursts and singletons."""
    if not USE_BURST_DETECTION:
        print("   Burst detection disabled")
        return [], photos
    
    try:
        print("   Detecting bursts...")
        clusters = detect_bursts_temporal_visual(
            photos,
            embeddings,
            time_window_seconds=10,
            embedding_similarity_threshold=0.85,
            min_burst_size=2
        )
        
        bursts = [c for c in clusters if len(c) >= 2]
        singletons = [c[0] for c in clusters if len(c) == 1]
        
        print(f"   ✅ Found {len(clusters)} groups")
        print(f"      Bursts: {len(bursts)}")
        print(f"      Singletons: {len(singletons)}")
        
        if bursts:
            sizes = [len(b) for b in bursts]
            print(f"      Burst sizes: min={min(sizes)}, max={max(sizes)}, avg={np.mean(sizes):.1f}")
        
        return bursts, singletons
    
    except Exception as e:
        print(f"   ⚠️  Burst detection failed: {e}")
        return [], photos

# ========================== MAIN CLASSIFICATION ==========================
def classify_and_route_burst_aware(photos, single_prompt, training_examples, share_ratio,
                                    gemini_cache, bursts, singletons, folder_path, checkpoint=None):
    """Main classification with burst awareness."""
    if checkpoint is None:
        checkpoint = {"processed_singles": [], "processed_bursts": [], "results": []}
    
    processed_singles = set(checkpoint["processed_singles"])
    processed_bursts = set(checkpoint["processed_bursts"])
    
    remaining_singletons = [p for p in singletons if str(p) not in processed_singles]
    
    remaining_bursts = []
    for burst in bursts:
        burst_key = "_".join(sorted([str(p) for p in burst]))
        if burst_key not in processed_bursts:
            remaining_bursts.append(burst)
    
    singleton_cost = len(remaining_singletons) * COST_PER_PHOTO
    burst_photos = sum(len(b) for b in remaining_bursts)
    burst_cost = burst_photos * COST_PER_BURST_PHOTO
    
    print(f"\n📊 Classification plan:")
    print(f"   Singletons: {len(remaining_singletons)}")
    print(f"   Bursts: {len(remaining_bursts)} ({burst_photos} photos)")
    print(f"   Already processed: {len(processed_singles)} singles, {len(processed_bursts)} bursts")
    print(f"   Estimated cost: ${singleton_cost + burst_cost:.3f}")
    
    # FIX #5: Clarify when files move
    print(f"\n⚠️  NOTE: Files will be moved at the END, not during classification")
    print(f"   Classification happens first, then all files move at once")
    
    if not remaining_singletons and not remaining_bursts:
        print("✅ All photos already classified!")
        return checkpoint["results"]
    
    input("\nPress Enter to start (Ctrl+C to cancel)...")
    
    results = checkpoint["results"]
    responses_shown = 0
    
    # Process bursts
    print(f"\n🎯 Processing {len(remaining_bursts)} bursts...")
    for burst_idx, burst in enumerate(remaining_bursts, 1):
        print(f"\n   Burst {burst_idx}/{len(remaining_bursts)}: {len(burst)} photos")

        # FIX: Don't truncate bursts - process all photos
        # If burst is too large, we'll handle it in chunks but still process all
        original_burst_size = len(burst)
        if len(burst) > MAX_BURST_SIZE_FOR_GROUPING:
            print(f"      ⚠️  WARNING: Burst has {len(burst)} photos (max recommended: {MAX_BURST_SIZE_FOR_GROUPING})")
            print(f"      ⚠️  This may result in reduced quality responses from Gemini")
            print(f"      ⚠️  Consider processing in chunks or reviewing manually")
            # Still process all photos, don't truncate
            # burst = burst[:MAX_BURST_SIZE_FOR_GROUPING]  # REMOVED - don't truncate!
        
        burst_prompt = build_burst_prompt(training_examples, share_ratio, len(burst))
        
        if responses_shown < 2:
            print(f"      🔍 Showing response...")
            responses_shown += 1

        burst_result = classify_burst_with_gemini_chunked(burst, burst_prompt, gemini_cache)
        
        photos_data = burst_result.get("photos", [])
        
        print(f"      Analysis: {burst_result.get('burst_analysis', 'N/A')[:80]}...")
        print(f"      Recommended: {burst_result.get('recommended_keep_count', 0)}/{len(burst)}")
        
        # Check if Gemini flagged as mediocre
        all_mediocre = burst_result.get("all_photos_mediocre", False)
        if all_mediocre:
            print(f"      ⚠️  Gemini flagged: entire burst is mediocre")

        # First pass: route all photos
        burst_photos_data = []
        for i, photo in enumerate(burst):
            if i < len(photos_data):
                photo_data = photos_data[i]
            else:
                photo_data = {
                    "index": i,
                    "classification": "Storage",
                    "confidence": 0.3,
                    "rank": i + 1,
                    "reasoning": "No data",
                    "technical_quality": "Unknown",
                    "expression_quality": "Unknown"
                }

            photo_data["path"] = str(photo)
            photo_data["burst_index"] = i
            photo_data["burst_size"] = len(burst)
            photo_data["burst_analysis"] = burst_result.get("burst_analysis", "")
            photo_data["burst_flagged_mediocre"] = all_mediocre

            # FIX #4: Use rank-based routing for bursts
            bucket, routing_reason = route_photo_burst_aware(photo_data, is_burst=True)
            photo_data["bucket"] = bucket
            photo_data["routing_reason"] = routing_reason

            burst_photos_data.append(photo_data)

        # Second pass: Check diversity among Share candidates
        share_candidates = [(idx, pd) for idx, pd in enumerate(burst_photos_data) if pd["bucket"] == "Share"]

        if len(share_candidates) > 1:
            print(f"      🔍 Checking diversity among {len(share_candidates)} Share candidates...")

            # Sort by rank (best first)
            share_candidates.sort(key=lambda x: x[1].get("rank", 999))

            # Keep first photo (best ranked), check diversity for others
            kept_photos = [share_candidates[0]]

            for idx, candidate_data in share_candidates[1:]:
                candidate_path = Path(candidate_data["path"])

                # Check diversity against all kept photos
                is_diverse = True
                for kept_idx, kept_data in kept_photos:
                    kept_path = Path(kept_data["path"])

                    diverse, conf, reason = check_diversity_between_photos(
                        kept_path, candidate_path, gemini_cache
                    )

                    if not diverse:
                        is_diverse = False
                        print(f"         ⚠️  Photo {idx} too similar to photo {kept_idx}: {reason}")

                        # Downgrade to Review
                        candidate_data["bucket"] = "Review"
                        candidate_data["routing_reason"] = f"Rank {candidate_data.get('rank')} but too similar to better photo"
                        candidate_data["diversity_check"] = "failed"
                        candidate_data["diversity_reason"] = reason
                        break

                if is_diverse:
                    kept_photos.append((idx, candidate_data))
                    candidate_data["diversity_check"] = "passed"

            print(f"      ✅ Diversity check: {len(kept_photos)}/{len(share_candidates)} photos kept for Share")

        # Add all photos to results
        for photo_data in burst_photos_data:
            results.append(photo_data)

            cls = photo_data.get("classification", "Unknown")
            conf = photo_data.get("confidence", 0)
            rank = photo_data.get("rank", 0)
            bucket = photo_data.get("bucket", "Unknown")
            photo_path = Path(photo_data["path"])
            # Handle None confidence values
            conf_str = f"{conf:.2f}" if conf is not None else "N/A"
            rank_str = f"{rank:2d}" if rank is not None else "??"
            print(f"         [{photo_data['burst_index']}] {photo_path.name[:30]:30s} | Rank {rank_str} | {cls:7s} → {bucket:7s} ({conf_str})")
        
        burst_key = "_".join(sorted([str(p) for p in burst]))
        checkpoint["processed_bursts"].append(burst_key)

        save_checkpoint(checkpoint, folder_path)
        save_gemini_cache(gemini_cache, folder_path)
    
    # Process singletons
    if remaining_singletons:
        print(f"\n📸 Processing {len(remaining_singletons)} singletons...")

        for i, photo in enumerate(remaining_singletons, 1):
            result = classify_single_photo_with_gemini(photo, single_prompt, gemini_cache)

            result["path"] = str(photo)
            result["burst_index"] = -1
            result["burst_size"] = 1

            bucket, routing_reason = route_photo_burst_aware(result, is_burst=False)
            result["bucket"] = bucket
            result["routing_reason"] = routing_reason

            results.append(result)
            checkpoint["processed_singles"].append(str(photo))

            # Print every singleton (like bursts)
            cls = result.get("classification", "Unknown")
            conf = result.get("confidence", 0)
            conf_str = f"{conf:.2f}" if conf is not None else "N/A"
            print(f"   [{i}/{len(remaining_singletons)}] {photo.name[:40]:40s} | {cls:7s} → {bucket:7s} ({conf_str})")

            if i % 10 == 0:
                save_checkpoint(checkpoint, folder_path)
                save_gemini_cache(gemini_cache, folder_path)

    save_checkpoint(checkpoint, folder_path)
    save_gemini_cache(gemini_cache, folder_path)

    return results

def move_photos_to_buckets(results):
    """
    FIX #5: Move photos to buckets.
    This happens at the END, after all classification is complete.
    """
    stats = defaultdict(int)
    log_rows = []
    
    print("\n📦 Moving photos to destination folders...")
    print("   (This happens at the end, after all classification)")
    
    for result in tqdm(results, desc="Moving files"):
        photo_path = Path(result["path"])
        bucket = result["bucket"]
        
        dst_dir = OUT_BASE / bucket
        dst_dir.mkdir(parents=True, exist_ok=True)
        
        written = copy_file(photo_path, dst_dir)
        
        if written:
            stats[bucket] += 1
            log_rows.append({
                "source": str(photo_path),
                "destination": written,
                "bucket": bucket,
                "classification": result.get("classification", "Unknown"),
                "confidence": result.get("confidence", 0),
                "burst_size": result.get("burst_size", 1),
                "burst_index": result.get("burst_index", -1),
                "rank": result.get("rank", 0),
                "reasoning": result.get("reasoning", ""),
                "routing_reason": result.get("routing_reason", ""),
                "burst_flagged_mediocre": result.get("burst_flagged_mediocre", False)
            })
    
    rep_dir = OUT_BASE / "Reports"
    rep_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(log_rows)
    df.to_csv(rep_dir / "gemini_v4_grouped_log.csv", index=False)
    
    summary_df = pd.DataFrame([
        {"bucket": k, "count": v, "percentage": v/len(results)*100}
        for k, v in stats.items()
    ])
    summary_df.to_csv(rep_dir / "gemini_v4_grouped_summary.csv", index=False)
    
    # Burst analysis
    burst_photos = df[df["burst_size"] > 1]
    if len(burst_photos) > 0:
        burst_summary = burst_photos.groupby(["burst_size", "bucket"]).size().reset_index(name="count")
        burst_summary.to_csv(rep_dir / "gemini_v4_grouped_burst_analysis.csv", index=False)
        
        print(f"\n   📊 Burst routing:")
        for size in sorted(burst_photos["burst_size"].unique()):
            size_data = burst_photos[burst_photos["burst_size"] == size]
            share_pct = (size_data["bucket"] == "Share").sum() / len(size_data) * 100
            print(f"      Size {size}: {share_pct:.1f}% to Share")
    
    print(f"\n   ✅ Reports saved to {rep_dir}/")
    
    return stats

# ========================== ANALYSIS ==========================
def analyze_results(results, training_examples):
    """Analyze classification results."""
    print("\n" + "="*60)
    print("CLASSIFICATION ANALYSIS")
    print("="*60)
    
    burst_results = [r for r in results if r.get("burst_size", 1) > 1]
    singleton_results = [r for r in results if r.get("burst_size", 1) == 1]
    
    print(f"\n📊 Dataset:")
    print(f"   Total: {len(results)}")
    print(f"   Bursts: {len(burst_results)} ({len(burst_results)/len(results)*100:.1f}%)")
    print(f"   Singletons: {len(singleton_results)}")
    
    print(f"\n📊 Routing:")
    buckets = defaultdict(int)
    for r in results:
        buckets[r["bucket"]] += 1
    for bucket, count in sorted(buckets.items()):
        print(f"   {bucket}: {count} ({count/len(results)*100:.1f}%)")
    
    # Burst analysis
    if burst_results:
        burst_share = sum(1 for r in burst_results if r["bucket"] == "Share")
        burst_share_pct = burst_share / len(burst_results) * 100
        
        singleton_share = sum(1 for r in singleton_results if r["bucket"] == "Share")
        singleton_share_pct = singleton_share / len(singleton_results) * 100 if singleton_results else 0
        
        print(f"\n📊 Burst vs Singleton:")
        print(f"   Burst Share: {burst_share_pct:.1f}%")
        print(f"   Singleton Share: {singleton_share_pct:.1f}%")
        
        if burst_share_pct < singleton_share_pct * 0.7:
            print("      ✅ Burst filtering working")
        
        # Bursts with 0 shares
        burst_groups = defaultdict(list)
        for r in burst_results:
            burst_key = f"{r.get('burst_size')}_{r.get('burst_index')//10}"
            burst_groups[burst_key].append(r)
        
        zero_share_bursts = sum(1 for photos in burst_groups.values()
                                if all(p["bucket"] != "Share" for p in photos))
        
        print(f"\n📊 Burst selectivity:")
        print(f"   Bursts with 0 shares: {zero_share_bursts}/{len(burst_groups)}")
        print(f"   ({zero_share_bursts/len(burst_groups)*100:.1f}%)")

# ========================== MAIN ==========================
def main():
    global UNLABELED, OUT_BASE

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Classify photos using Gemini AI')
    parser.add_argument('input_folder', type=str,
                       help='Path to the folder containing photos to classify')
    parser.add_argument('--dry-run', action='store_true',
                       help='Run without actually moving files')
    parser.add_argument('--classify-videos', action='store_true',
                       help='Classify videos using Gemini (costs more tokens). Default: just copy to Videos/ folder')
    parser.add_argument('--parallel-videos', type=int, default=10, metavar='N',
                       help='Number of parallel workers for video classification (default: 10, use 1 for sequential)')

    args = parser.parse_args()

    # Set input and output paths
    UNLABELED = Path(args.input_folder)
    if not UNLABELED.exists():
        print(f"\n❌ Input folder does not exist: {UNLABELED}")
        return

    # Create output folder name: input_folder_sorted
    if UNLABELED.name.endswith('_sorted'):
        # Avoid double-suffixing
        OUT_BASE = UNLABELED
    else:
        OUT_BASE = UNLABELED.parent / f"{UNLABELED.name}_sorted"

    start_time = time.time()

    print("="*60)
    print("GEMINI PHOTO CLASSIFIER V4 (GROUPED BURST EVALUATION)")
    print("="*60)
    print(f"\n📂 Input folder:  {UNLABELED}")
    print(f"📂 Output folder: {OUT_BASE}")

    if not GEMINI_API_KEY:
        print("\n❌ GEMINI_API_KEY not set!")
        return

    print(f"\n✅ API key configured")
    
    print("\n📦 Loading training data...")
    labels = load_labels()
    if labels is None:
        return
    
    training_examples = convert_pairwise_labels_to_training_examples(labels)
    
    if len(training_examples) < 5:
        print(f"\n❌ Not enough training examples")
        return
    
    share_count = sum(1 for d in training_examples.values() if d["action"] == "Share")
    share_ratio = share_count / len(training_examples)
    
    print(f"   ✅ {len(training_examples)} examples")
    print(f"   Share ratio: {share_ratio:.1%}")

    gemini_cache = load_gemini_cache(UNLABELED)
    print(f"   ✅ {len(gemini_cache)} cached responses for folder: {get_folder_identifier(UNLABELED)}")

    print("\n🤖 Building prompts...")
    single_prompt = build_single_photo_prompt(training_examples, share_ratio)
    
    print("\n📂 Loading media files...")
    # Get ALL files (images + videos) - list_images only returns images
    all_media_files = []
    if UNLABELED.exists():
        for p in UNLABELED.rglob("*"):
            if p.is_file():
                ext = p.suffix.lower()
                # Check if image or video
                if ext in {".jpg",".jpeg",".png",".webp",".heic",".tif",".tiff",".bmp"} or is_video_file(p):
                    all_media_files.append(p)

    print(f"   Found {len(all_media_files)} media files total")

    # Separate videos from photos
    videos = [f for f in all_media_files if is_video_file(f)]
    photos = [f for f in all_media_files if not is_video_file(f)]

    print(f"   Images: {len(photos)}, Videos: {len(videos)}")

    # Initialize video_results (will be populated if videos are classified)
    video_results = []

    if videos and args.classify_videos:
        print(f"\n📹 Found {len(videos)} videos - CLASSIFICATION MODE ENABLED")

        # Estimate cost with ACCURATE pricing
        total_video_seconds = sum(get_video_duration(v) for v in videos)
        total_size_gb = sum(v.stat().st_size for v in videos) / (1024**3)

        # Input tokens: video frames + prompt for each video
        input_tokens_per_video = TOKENS_PER_VIDEO_SECOND * (total_video_seconds / len(videos)) + PROMPT_TOKENS_VIDEO
        total_input_tokens = input_tokens_per_video * len(videos)

        # Output tokens: JSON response for each video
        total_output_tokens = OUTPUT_TOKENS_VIDEO * len(videos)

        # Calculate cost
        input_cost = (total_input_tokens / 1_000_000) * COST_PER_1M_INPUT_TOKENS
        output_cost = (total_output_tokens / 1_000_000) * COST_PER_1M_OUTPUT_TOKENS
        estimated_cost = input_cost + output_cost

        print(f"   Video stats:")
        print(f"      Count: {len(videos)} videos")
        print(f"      Total size: {total_size_gb:.1f} GB")
        print(f"      Estimated duration: ~{total_video_seconds//60} minutes ({total_video_seconds//len(videos)} sec/video avg)")
        print(f"\n   Cost breakdown:")
        print(f"      Input tokens: ~{total_input_tokens:,.0f} (video frames + prompts)")
        print(f"      Output tokens: ~{total_output_tokens:,.0f} (JSON responses)")
        print(f"      Input cost: ${input_cost:.2f}")
        print(f"      Output cost: ${output_cost:.2f}")
        print(f"      TOTAL COST: ${estimated_cost:.2f}")
        print(f"\n   ⚠️  This is {len(videos)} videos × ${estimated_cost/len(videos):.4f}/video")
        print(f"   ⚠️  Estimate assumes ~{total_video_seconds//len(videos)} sec/video based on file size")
        print(f"   ⚠️  Actual cost may be 2-3x higher if videos are longer than estimated")
        print(f"\n   💡 RECOMMENDATION: Test with 10-20 videos first to verify accuracy")

        # Build video prompt
        video_prompt = build_video_prompt(training_examples, share_ratio)

        # Process videos - use parallel or sequential based on flag
        if args.parallel_videos > 1:
            # Parallel processing (10x faster!)
            video_results = classify_videos_parallel(
                videos, video_prompt, gemini_cache, UNLABELED,
                max_workers=args.parallel_videos
            )
        else:
            # Sequential processing (original behavior)
            print(f"\n   Processing videos sequentially (use --parallel-videos to speed up)")
            video_results = []

            for i, video_path in enumerate(videos, 1):
                print(f"\n   Video {i}/{len(videos)}: {video_path.name}")

                result = classify_video_with_gemini(video_path, video_prompt, gemini_cache)

                result["path"] = str(video_path)
                result["burst_index"] = -1
                result["burst_size"] = 1
                result["is_video"] = True

                bucket, routing_reason = route_photo_burst_aware(result, is_burst=False)
                result["bucket"] = bucket
                result["routing_reason"] = routing_reason

                video_results.append(result)

                cls = result.get("classification", "Unknown")
                conf = result.get("confidence", 0)
                audio = result.get("audio_quality", "n/a")
                video_q = result.get("video_quality", "unknown")
                reasoning = result.get("reasoning", "No reasoning provided")
                conf_str = f"{conf:.2f}" if conf is not None else "N/A"

                print(f"      {cls:7s} → {bucket:7s} ({conf_str}) | Audio: {audio}, Video: {video_q}")
                print(f"      Reasoning: {reasoning}")

                if i % 5 == 0:
                    save_gemini_cache(gemini_cache, UNLABELED)

            save_gemini_cache(gemini_cache, UNLABELED)

            print(f"\n   ✅ {len(videos)} videos classified")

        # Videos will be routed to Share/Storage/Ignore by move_photos_to_buckets()
        # Add video results to main results list (will be handled at the end)

    elif videos:
        print(f"\n📹 Found {len(videos)} videos - copying to Videos/ folder")
        print(f"   (Use --classify-videos to classify videos with Gemini)")

        videos_folder = OUT_BASE / "Videos"
        videos_folder.mkdir(exist_ok=True, parents=True)

        copied_count = 0
        skipped_count = 0
        for video_path in videos:
            dest_path = videos_folder / video_path.name

            # Skip if already exists
            if dest_path.exists():
                skipped_count += 1
                continue

            try:
                # copy_file(src, dst_dir) - pass directory, not full path
                copy_file(video_path, videos_folder)
                copied_count += 1
                if not DRY_RUN:
                    print(f"   Copied: {video_path.name}")
            except Exception as e:
                print(f"   ⚠️  Could not copy {video_path.name}: {e}")

        print(f"   ✅ Videos: {copied_count} copied, {skipped_count} already exist")

    print(f"   ✅ {len(photos)} photos to classify")

    # Initialize results
    results = []

    # Process photos if any exist
    if len(photos) > 0:
        # Load CLIP model first (needed for baby gate and burst detection)
        print("\n🔍 Loading CLIP model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            print(f"   🎮 GPU detected: {torch.cuda.get_device_name(0)}")
        else:
            print(f"   💻 No GPU detected, using CPU")
        model, preprocess, _ = load_model(device)
        print(f"   ✅ Model loaded on {device}")

        print("\n🔍 Computing embeddings and organizing into bursts...")
        embeddings, good_paths = embed_paths(photos, model, preprocess, device, batch_size=128)

        if len(good_paths) == 0:
            print("   ❌ No photos processed")
        else:
            bursts, singletons = organize_photos_into_bursts(good_paths, embeddings)

            checkpoint = load_checkpoint(UNLABELED)

            if checkpoint["processed_singles"] or checkpoint["processed_bursts"]:
                print(f"\n📍 Found checkpoint for folder: {get_folder_identifier(UNLABELED)}")
                print(f"   Processed: {len(checkpoint['processed_singles'])} singles, {len(checkpoint['processed_bursts'])} bursts")
                print(f"   Results: {len(checkpoint['results'])}")
                print("   ℹ️  Continuing from checkpoint")

            try:
                results = classify_and_route_burst_aware(
                    good_paths, single_prompt, training_examples, share_ratio,
                    gemini_cache, bursts, singletons, UNLABELED, checkpoint
                )

            except KeyboardInterrupt:
                elapsed_time = time.time() - start_time
                hours, remainder = divmod(int(elapsed_time), 3600)
                minutes, seconds = divmod(remainder, 60)

                print("\n\n⚠️  Interrupted")
                print(f"   ⏱️  Runtime so far: {hours:02d}:{minutes:02d}:{seconds:02d}")
                print("   ✅ Progress saved")
                print(f"   📍 {get_checkpoint_file(UNLABELED)}")
                print(f"   📊 {len(checkpoint['results'])} photos classified so far")
                print("   ℹ️  Run again to continue")
                return
    else:
        print("   No photos to classify - skipping photo processing")

    # Add video results if videos were classified
    if video_results:
        print(f"\n   Adding {len(video_results)} video classification results")
        results.extend(video_results)

    # If we have no results at all (no photos, no videos), exit
    if len(results) == 0:
        print("\n❌ No photos or videos to process!")
        return

    analyze_results(results, training_examples)
    
    # FIX #5: Clarify this is when files move
    if not DRY_RUN:
        print("\n" + "="*60)
        print("MOVING FILES TO DESTINATION FOLDERS")
        print("="*60)
        stats = move_photos_to_buckets(results)
        
        print("\n" + "="*60)
        print("FINAL SUMMARY")
        print("="*60)
        for bucket, count in sorted(stats.items()):
            print(f"{bucket:20s}: {count:5d} ({count/len(results)*100:.1f}%)")
    else:
        print("\n⚠️  DRY RUN - Files not moved")

    checkpoint_file = get_checkpoint_file(UNLABELED)
    if checkpoint_file.exists():
        checkpoint_file.unlink()
        print(f"\n✅ Checkpoint cleaned up: {checkpoint_file.name}")

    # Display elapsed time
    elapsed_time = time.time() - start_time
    elapsed_td = timedelta(seconds=int(elapsed_time))
    hours, remainder = divmod(int(elapsed_time), 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"\n⏱️  Total runtime: {hours:02d}:{minutes:02d}:{seconds:02d}")
    print(f"📊 Reports: {OUT_BASE / 'Reports'}/")
    print("\n✅ Complete!")

if __name__ == "__main__":
    main()