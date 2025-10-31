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
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm
import google.generativeai as genai

# Import from existing code  
sys.path.insert(0, '..')
sys.path.insert(0, '..')

from taste_sort_win import (
    list_images, UNLABELED, OUT_BASE, DRY_RUN, get_cache_key,
    copy_file, CACHE_ROOT, load_model, embed_paths
)

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

# ========================== CONFIG ==========================
LABELING_CACHE = CACHE_ROOT / "labeling_pairwise"
LABELING_CACHE.mkdir(parents=True, exist_ok=True)

LABELS_FILE = LABELING_CACHE / "pairwise_labels.json"
GEMINI_CACHE_FILE = LABELING_CACHE / "gemini_cache_v4_grouped.json"
CHECKPOINT_FILE = LABELING_CACHE / "classification_checkpoint_v4_grouped.json"

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

# FIX #3: More conservative burst routing
# Only share top photos from bursts, and only if they meet quality bar
BURST_RANK_SHARE_THRESHOLD = 2  # Only share rank 1-2
BURST_MIN_QUALITY_TO_SHARE = 0.90  # Gemini confidence must be >= 0.90 to share from burst

# Routing thresholds for singletons
SHARE_THRESHOLD = 0.50
REVIEW_THRESHOLD = 0.35

# Cost estimation
COST_PER_1K_IMAGES = 0.075
COST_PER_BURST_MULTIPLIER = 1.5

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

# ========================== GEMINI PROMPT BUILDING ==========================
def build_single_photo_prompt(training_examples, share_ratio, max_examples=50):
    """Build prompt for evaluating INDIVIDUAL photos."""
    share_examples = [(path, data) for path, data in training_examples.items()
                     if data["action"] == "Share" and data["confidence"] >= 0.7]
    storage_examples = [(path, data) for path, data in training_examples.items()
                       if data["action"] == "Storage" and data["confidence"] >= 0.7]
    
    n_share = min(len(share_examples), max_examples // 2)
    n_storage = min(len(storage_examples), max_examples // 2)
    
    selected_share = share_examples[:n_share]
    selected_storage = storage_examples[:n_storage]
    
    prompt = f"""You are helping sort baby photos into two categories: SHARE and STORAGE.

**CRITICAL CONTEXT:**
- These are photos of BABIES/TODDLERS for a family photo album
- User ONLY shares photos of their children
- User does NOT share: food, documents, scenery, objects, or other people

**SHARE = Photos worth sharing with family**
- Shows baby's personality, expression, or special moment
- Good technical quality (sharp, well-lit, eyes open)

**STORAGE = Everything else**
- Not a baby photo
- Poor quality (blurry, bad lighting, eyes closed)
- Not special or interesting

**CALIBRATION:**
- In training data, {share_ratio:.1%} of photos are Share-worthy
- Be selective!

Here are examples:

=== SHARE EXAMPLES ===

"""
    
    for i, (path, data) in enumerate(selected_share, 1):
        prompt += f"Example {i}: SHARE\n"
        if data.get("reasoning"):
            prompt += f"Reasoning: {data['reasoning']}\n"
        prompt += "\n"
    
    prompt += "\n=== STORAGE EXAMPLES ===\n\n"
    for i, (path, data) in enumerate(selected_storage, 1):
        prompt += f"Example {i}: STORAGE\n"
        if data.get("reasoning"):
            prompt += f"Reasoning: {data['reasoning']}\n"
        prompt += "\n"
    
    prompt += """
Analyze this photo and respond with JSON:
{
    "classification": "Share" or "Storage",
    "confidence": 0.0 to 1.0,
    "reasoning": "Brief explanation",
    "subject_check": "Is this a baby photo? Yes/No"
}
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
    
    prompt = f"""You are helping sort baby photos into two categories: SHARE and STORAGE.

**BURST EVALUATION MODE:**
You are viewing {burst_size} photos taken in rapid succession.
These are very similar - slight variations of the same moment.

**YOUR TASK:**
1. Compare ALL photos in the burst
2. Rank them by quality (1=best, {burst_size}=worst)
3. Be VERY selective - typically keep only 0-2 photos from a burst
4. Consider: Is this burst even worth keeping ANY photos from?

**EVALUATION CRITERIA:**
- Technical quality: sharpness, lighting, eyes open
- Expression: smile, personality, natural moment
- Composition: framing, background
- Relative quality within THIS burst

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
You will see {burst_size} photos. Respond with JSON:

{{
    "burst_analysis": "Overall assessment of this burst's quality",
    "all_photos_mediocre": true or false,
    "photos": [
        {{
            "index": 0,
            "classification": "Share" or "Storage",
            "confidence": 0.0 to 1.0,
            "rank": 1 to {burst_size},
            "reasoning": "Why this rank? Compare to others in burst",
            "technical_quality": "Sharpness, lighting, eyes",
            "expression_quality": "Personality, moment captured"
        }},
        ... (one entry for EACH photo, in order)
    ],
    "recommended_keep_count": 0 to {burst_size},
    "burst_summary": "Which specific photos (by index) to share and why"
}}

**REMEMBER:**
- It's OK to recommend 0 photos if burst is entirely mediocre
- Rank honestly - don't give multiple photos the same rank
- Be selective - keeping too many defeats the purpose
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
        
        response = model.generate_content([query, img])
        text = response.text.strip()
        
        # Parse JSON
        import re
        json_match = re.search(r'\{[^}]+\}', text, re.DOTALL)
        
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
                else:
                    result["classification"] = "Storage"
                    result["confidence"] = 0.3
                    result["reasoning"] = f"Could not parse: {cls}"
                
                cache[cache_key] = result
                return result
            except json.JSONDecodeError:
                pass
        
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

def classify_burst_with_gemini(burst_paths, prompt, cache):
    """Classify an ENTIRE BURST together."""
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
        
        response = model.generate_content(content)
        text = response.text.strip()
        
        # Parse JSON
        import re
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        
        if json_match:
            try:
                result = json.loads(json_match.group(0))
                result["raw_response"] = text
                
                photos_data = result.get("photos", [])
                
                if len(photos_data) != len(burst_paths):
                    raise ValueError(f"Expected {len(burst_paths)} entries, got {len(photos_data)}")
                
                # Normalize classifications
                for photo_data in photos_data:
                    cls = photo_data.get("classification", "").strip()
                    if cls.upper() == "SHARE":
                        photo_data["classification"] = "Share"
                    elif cls.upper() == "STORAGE":
                        photo_data["classification"] = "Storage"
                    else:
                        photo_data["classification"] = "Storage"
                        photo_data["confidence"] = 0.3
                
                cache[burst_cache_key] = result
                return result
            
            except (json.JSONDecodeError, ValueError) as e:
                print(f"\n⚠️  Parse error: {e}")
                print(f"   Response: {text[:300]}")
        
        # Fallback
        result = {
            "burst_analysis": "Parse error",
            "all_photos_mediocre": False,
            "photos": [
                {
                    "index": i,
                    "classification": "Storage",
                    "confidence": 0.3,
                    "rank": i + 1,
                    "reasoning": "Parse error",
                    "technical_quality": "Unknown",
                    "expression_quality": "Unknown"
                }
                for i in range(len(burst_paths))
            ],
            "recommended_keep_count": 0,
            "burst_summary": f"Error: {text[:200]}",
            "raw_response": text
        }
        
        cache[burst_cache_key] = result
        return result
    
    except Exception as e:
        print(f"\n⚠️  ERROR: {str(e)[:100]}")
        return {
            "error": str(e),
            "burst_analysis": "Error",
            "all_photos_mediocre": False,
            "photos": [
                {
                    "index": i,
                    "classification": "Storage",
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

def load_gemini_cache():
    """Load Gemini cache."""
    if GEMINI_CACHE_FILE.exists():
        with open(GEMINI_CACHE_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_gemini_cache(cache):
    """Save Gemini cache."""
    with open(GEMINI_CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)

def load_checkpoint():
    """Load classification checkpoint."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {"processed_singles": [], "processed_bursts": [], "results": []}

def save_checkpoint(checkpoint):
    """Save classification checkpoint."""
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f, indent=2)

# ========================== ROUTING ==========================
def route_photo_burst_aware(photo_data, is_burst=False):
    """
    FIX #4: Rank-based routing for bursts instead of confidence-based.
    
    For bursts: Gemini's confidence just means "I'm sure about the ranking"
    not "This photo is X% likely to be Share-worthy".
    
    So we use RANK + quality threshold instead.
    """
    cls = photo_data["classification"]
    conf = photo_data.get("confidence", 0.5)
    rank = photo_data.get("rank", 999)
    
    if is_burst:
        # For burst photos, use rank-based routing with quality gate
        if cls == "Share":
            # Must meet BOTH criteria:
            # 1. Gemini said "Share" 
            # 2. Good enough rank
            # 3. High enough confidence (quality bar)
            if rank <= BURST_RANK_SHARE_THRESHOLD and conf >= BURST_MIN_QUALITY_TO_SHARE:
                return "Share", f"Rank {rank}, high quality ({conf:.2f})"
            elif rank <= BURST_RANK_SHARE_THRESHOLD:
                return "Review", f"Rank {rank} but quality concern ({conf:.2f})"
            else:
                return "Storage", f"Rank {rank} too low for sharing"
        else:
            # Gemini said Storage
            return "Storage", f"Rank {rank}, Gemini classified as Storage"
    else:
        # For singletons, use confidence-based routing (works fine)
        if cls == "Share":
            if conf >= SHARE_THRESHOLD:
                return "Share", f"High confidence Share ({conf:.2f})"
            elif conf >= REVIEW_THRESHOLD:
                return "Review", f"Medium confidence Share ({conf:.2f})"
            else:
                return "Storage", f"Low confidence Share ({conf:.2f})"
        else:
            if conf >= SHARE_THRESHOLD:
                return "Storage", f"High confidence Storage ({conf:.2f})"
            else:
                return "Review", f"Uncertain ({conf:.2f})"

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
                                    gemini_cache, bursts, singletons, checkpoint=None):
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
    
    singleton_cost = len(remaining_singletons) * COST_PER_1K_IMAGES / 1000
    burst_photos = sum(len(b) for b in remaining_bursts)
    burst_cost = burst_photos * COST_PER_1K_IMAGES * COST_PER_BURST_MULTIPLIER / 1000
    
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
        
        if len(burst) > MAX_BURST_SIZE_FOR_GROUPING:
            print(f"      ⚠️  Truncating to {MAX_BURST_SIZE_FOR_GROUPING}")
            burst = burst[:MAX_BURST_SIZE_FOR_GROUPING]
        
        burst_prompt = build_burst_prompt(training_examples, share_ratio, len(burst))
        
        if responses_shown < 2:
            print(f"      🔍 Showing response...")
            responses_shown += 1
        
        burst_result = classify_burst_with_gemini(burst, burst_prompt, gemini_cache)
        
        photos_data = burst_result.get("photos", [])
        
        print(f"      Analysis: {burst_result.get('burst_analysis', 'N/A')[:80]}...")
        print(f"      Recommended: {burst_result.get('recommended_keep_count', 0)}/{len(burst)}")
        
        # Check if Gemini flagged as mediocre
        all_mediocre = burst_result.get("all_photos_mediocre", False)
        if all_mediocre:
            print(f"      ⚠️  Gemini flagged: entire burst is mediocre")
        
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
            
            results.append(photo_data)
            
            cls = photo_data.get("classification", "Unknown")
            conf = photo_data.get("confidence", 0)
            rank = photo_data.get("rank", 0)
            print(f"         [{i}] {photo.name[:30]:30s} | Rank {rank:2d} | {cls:7s} → {bucket:7s} ({conf:.2f})")
        
        burst_key = "_".join(sorted([str(p) for p in burst]))
        checkpoint["processed_bursts"].append(burst_key)
        
        save_checkpoint(checkpoint)
        save_gemini_cache(gemini_cache)
    
    # Process singletons
    if remaining_singletons:
        print(f"\n📸 Processing {len(remaining_singletons)} singletons...")
        
        for i, photo in enumerate(remaining_singletons, 1):
            if i % 10 == 1 or i == len(remaining_singletons):
                print(f"   [{i}/{len(remaining_singletons)}] {photo.name}...", end='')
            
            result = classify_single_photo_with_gemini(photo, single_prompt, gemini_cache)
            
            result["path"] = str(photo)
            result["burst_index"] = -1
            result["burst_size"] = 1
            
            bucket, routing_reason = route_photo_burst_aware(result, is_burst=False)
            result["bucket"] = bucket
            result["routing_reason"] = routing_reason
            
            results.append(result)
            checkpoint["processed_singles"].append(str(photo))
            
            cls = result.get("classification", "Unknown")
            conf = result.get("confidence", 0)
            if i % 10 == 0 or i == len(remaining_singletons):
                print(f" [{cls}] → {bucket} ({conf:.2f})")
            
            if i % 10 == 0:
                save_checkpoint(checkpoint)
                save_gemini_cache(gemini_cache)
    
    save_checkpoint(checkpoint)
    save_gemini_cache(gemini_cache)
    
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
    print("="*60)
    print("GEMINI PHOTO CLASSIFIER V4 (GROUPED BURST EVALUATION)")
    print("="*60)
    
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
    
    gemini_cache = load_gemini_cache()
    print(f"   ✅ {len(gemini_cache)} cached responses")
    
    print("\n🤖 Building prompts...")
    single_prompt = build_single_photo_prompt(training_examples, share_ratio)
    
    print("\n📂 Loading photos...")
    photos = list_images(UNLABELED)
    print(f"   ✅ {len(photos)} photos")
    
    if len(photos) == 0:
        return
    
    print("\n🔍 Organizing into bursts...")
    device = "cuda" if os.path.exists("/dev/nvidia0") else "cpu"
    model, preprocess, _ = load_model(device)
    embeddings, good_paths = embed_paths(photos, model, preprocess, device, batch_size=128)
    
    if len(good_paths) == 0:
        print("   ❌ No photos processed")
        return
    
    bursts, singletons = organize_photos_into_bursts(good_paths, embeddings)
    
    checkpoint = load_checkpoint()
    
    if checkpoint["processed_singles"] or checkpoint["processed_bursts"]:
        print(f"\n📍 Found checkpoint:")
        print(f"   Processed: {len(checkpoint['processed_singles'])} singles, {len(checkpoint['processed_bursts'])} bursts")
        print(f"   Results: {len(checkpoint['results'])}")
        print("   ℹ️  Continuing from checkpoint")
    
    try:
        results = classify_and_route_burst_aware(
            good_paths, single_prompt, training_examples, share_ratio,
            gemini_cache, bursts, singletons, checkpoint
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted")
        print("   ✅ Progress saved")
        print(f"   📍 {CHECKPOINT_FILE}")
        print(f"   📊 {len(checkpoint['results'])} photos classified so far")
        print("   ℹ️  Run again to continue")
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
    
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        print("\n✅ Checkpoint cleaned up")
    
    print(f"\n📊 Reports: {OUT_BASE / 'Reports'}/")
    print("\n✅ Complete!")

if __name__ == "__main__":
    main()