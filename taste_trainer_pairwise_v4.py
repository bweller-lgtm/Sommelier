# taste_trainer_pairwise_v4_FIXED.py
# Complete fixes: Gallery UI + Weighted Training + Comparison Type Tracking

import os
import sys
import json
import random
import pickle
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import gradio as gr
from tqdm import tqdm

# Import from existing code
sys.path.insert(0, '..')
from taste_sort_win import (
    list_images, embed_paths, cluster_near_dups, get_quality_scores_batch,
    load_model, UNLABELED, FAMILY_DIR, PRIVATE_DIR, RANDOM_SEED,
    CACHE_ROOT, get_cache_key
)

# Import new burst detector
try:
    sys.path.insert(0, '..')
    from burst_detector import detect_bursts_temporal_visual
    USE_NEW_BURST_DETECTION = True
except:
    USE_NEW_BURST_DETECTION = False
    print("[!] New burst detection not available, using original method")

import google.generativeai as genai

# ========================== CONFIG ==========================
LABELING_CACHE = CACHE_ROOT / "labeling_pairwise"
LABELING_CACHE.mkdir(parents=True, exist_ok=True)

LABELS_FILE = LABELING_CACHE / "pairwise_labels.json"
SCENE_CLUSTERS_FILE = LABELING_CACHE / "scene_clusters.pkl"
GEMINI_CACHE_FILE = LABELING_CACHE / "gemini_cache.json"

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Sampling strategy
MIN_LABELS_BEFORE_TRAINING = 15  # Lower since pairwise gives more info

# Sampling mix (should sum to ~1.0)
WITHIN_CLUSTER_RATIO = 0.5   # 50% within-cluster (similar photos)
BETWEEN_CLUSTER_RATIO = 0.3  # 30% between-cluster (different photos)
GALLERY_RATIO = 0.2          # 20% gallery mode (large bursts)

# Large burst handling
LARGE_BURST_THRESHOLD = 5  # Bursts with 5+ photos get special "gallery" mode
ENABLE_GALLERY_MODE = True  # Enable gallery mode for large bursts

# Weighted training (NEW)
WITHIN_BURST_WEIGHT = 2.0   # Higher weight for quality distinctions
BETWEEN_BURST_WEIGHT = 1.0  # Normal weight for preference learning
GALLERY_WEIGHT = 1.5        # Medium weight for multi-selection

# Confidence levels for single photos
CONFIDENCE_LEVELS = {
    "Very High": 1.0,
    "High": 0.8,
    "Medium": 0.6,
    "Low": 0.4,
    "Uncertain": 0.2
}

# ========================== UTILITIES ==========================
def fix_image_orientation(image_path, max_size=1024):
    """
    Load image, fix orientation, and resize for faster loading.
    max_size: Maximum dimension (width or height) in pixels
    """
    try:
        img = Image.open(image_path)
        # Fix orientation based on EXIF
        img = ImageOps.exif_transpose(img)
        
        # Resize if too large (for faster UI loading)
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

# ========================== DATA STRUCTURES ==========================
class PairwiseLabelingSession:
    def __init__(self):
        self.labels = self.load_labels()
        self.scene_clusters = None
        self.gemini_cache = self.load_gemini_cache()
        
    def load_labels(self):
        """Load existing pairwise labels from disk."""
        if LABELS_FILE.exists():
            with open(LABELS_FILE, 'r') as f:
                return json.load(f)
        return {
            "pairwise": [],  # List of pairwise comparisons
            "single": {},    # Single photo labels
            "gallery": []    # Gallery/burst selections
        }
    
    def save_labels(self):
        """Save labels to disk."""
        with open(LABELS_FILE, 'w') as f:
            json.dump(self.labels, f, indent=2)
    
    def load_gemini_cache(self):
        """Load Gemini API response cache."""
        if GEMINI_CACHE_FILE.exists():
            with open(GEMINI_CACHE_FILE, 'r') as f:
                return json.load(f)
        return {}
    
    def save_gemini_cache(self):
        """Save Gemini cache."""
        with open(GEMINI_CACHE_FILE, 'w') as f:
            json.dump(self.gemini_cache, f, indent=2)
    
    def add_pairwise_label(self, photo_a, photo_b, choice, reason="", comparison_type=None):
        """
        Add a pairwise comparison.
        choice: "both", "left", "right", "neither"
        comparison_type: "within" or "between" to indicate burst context
        """
        self.labels["pairwise"].append({
            "photo_a": str(photo_a),
            "photo_b": str(photo_b),
            "choice": choice,
            "reason": reason,
            "comparison_type": comparison_type,  # NEW: track context
            "timestamp": datetime.now().isoformat()
        })
        self.save_labels()
    
    def add_gallery_label(self, photos, selected_indices, reason=""):
        """
        Add a gallery/burst selection.
        photos: List of all photos in burst
        selected_indices: List of indices that were selected as keepers
        """
        self.labels["gallery"].append({
            "photos": [str(p) for p in photos],
            "selected_indices": selected_indices,
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        })
        
        # Also mark each photo individually for easier tracking
        for i, photo in enumerate(photos):
            action = "Share" if i in selected_indices else "Storage"
            self.add_single_label(photo, action, "High", f"Gallery selection: {reason}")
    
    def add_single_label(self, photo_path, action, confidence, reason=""):
        """Add a single photo label (for photos not in bursts)."""
        photo_path = str(photo_path)
        self.labels["single"][photo_path] = {
            "action": action,
            "confidence": CONFIDENCE_LEVELS[confidence],
            "confidence_label": confidence,
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        }
        self.save_labels()
    
    def get_label_stats(self):
        """Get statistics about labeled data."""
        n_pairwise = len(self.labels["pairwise"])
        n_gallery = len(self.labels.get("gallery", []))
        n_single = len(self.labels["single"])
        total = n_pairwise + n_gallery + n_single
        
        if total == 0:
            return "No labels yet"
        
        # Count choices in pairwise (skip gallery entries that don't have 'choice')
        both_count = sum(1 for p in self.labels["pairwise"] if p.get("choice") == "both")
        left_count = sum(1 for p in self.labels["pairwise"] if p.get("choice") == "left")
        right_count = sum(1 for p in self.labels["pairwise"] if p.get("choice") == "right")
        neither_count = sum(1 for p in self.labels["pairwise"] if p.get("choice") == "neither")
        
        # Count within vs between (only for pairwise comparisons, not gallery)
        within_count = sum(1 for p in self.labels["pairwise"] if p.get("type") != "gallery" and p.get("comparison_type") == "within")
        between_count = sum(1 for p in self.labels["pairwise"] if p.get("type") != "gallery" and p.get("comparison_type") == "between")
        
        # Count single photos
        share_single = sum(1 for l in self.labels["single"].values() if l["action"] == "Share")
        storage_single = len(self.labels["single"]) - share_single
        
        return f"""[STATS] Labeling Progress:

[PAIR] Pairwise Comparisons: {n_pairwise}
   - Within-burst (quality): {within_count}
   - Between-scene (preference): {between_count}
   - Share Both: {both_count}
   - Share Left: {left_count}
   - Share Right: {right_count}
   - Share Neither: {neither_count}

[GALLERY] Gallery Reviews: {n_gallery}

[PHOTO] Single Photos: {n_single}
   - Share: {share_single}
   - Storage: {storage_single}

[PROGRESS] Total: {total} labels
   Ready to train: {'[OK] Yes' if total >= MIN_LABELS_BEFORE_TRAINING else f'[NO] No (need {MIN_LABELS_BEFORE_TRAINING - total} more)'}
"""
    
    def get_training_weights(self):
        """
        Get sample weights for training based on comparison type.
        Returns dict mapping label index to weight.
        """
        weights = {}
        idx = 0
        
        # Pairwise comparisons
        for comp in self.labels["pairwise"]:
            comp_type = comp.get("comparison_type", "unknown")
            
            if comp_type == "within":
                weight = WITHIN_BURST_WEIGHT
            elif comp_type == "between":
                weight = BETWEEN_BURST_WEIGHT
            else:
                weight = 1.0
            
            # Each comparison generates 2 training examples (A and B)
            weights[idx] = weight
            weights[idx + 1] = weight
            idx += 2
        
        # Gallery selections (from single labels marked as gallery)
        for photo_path, label in self.labels["single"].items():
            if "Gallery selection" in label.get("reason", ""):
                weights[idx] = GALLERY_WEIGHT
            else:
                weights[idx] = 1.0
            idx += 1
        
        return weights

# ========================== SCENE CLUSTERING ==========================
def build_scene_clusters(photos, device="cpu"):
    """Build scene clusters from photos using improved burst detection."""
    global USE_NEW_BURST_DETECTION
    
    print("\n[SEARCH] Building scene clusters...")
    
    if SCENE_CLUSTERS_FILE.exists():
        print("   Loading cached clusters...")
        with open(SCENE_CLUSTERS_FILE, 'rb') as f:
            return pickle.load(f)
    
    # Load model and compute embeddings (needed for both methods)
    model, preprocess, _ = load_model(device)
    
    print("   Computing embeddings...")
    embeddings, good_paths = embed_paths(photos, model, preprocess, device, batch_size=128)
    
    if len(good_paths) == 0:
        print("   [NO] No photos could be embedded!")
        return []
    
    if USE_NEW_BURST_DETECTION:
        print("   Using NEW burst detection (temporal + visual)...")
        try:
            bursts = detect_bursts_temporal_visual(
                good_paths,
                embeddings,
                time_window_seconds=5,
                embedding_similarity_threshold=0.85,
                min_burst_size=2,
                max_burst_size=50
            )
            
            print(f"      Using relaxed thresholds to detect burst SEQUENCES")
            print(f"      Time window: 5s, Similarity: 0.85")
            
            clusters = bursts
            
            print(f"   [OK] Found {len(clusters)} scenes (NEW method)")
            multi = sum(1 for c in clusters if len(c) > 1)
            print(f"      Burst scenes (2+ photos): {multi}")
            print(f"      Single photos: {len(clusters) - multi}")
            
            burst_sizes = [len(c) for c in clusters if len(c) > 1]
            if burst_sizes:
                print(f"      Burst size: min={min(burst_sizes)}, max={max(burst_sizes)}, avg={np.mean(burst_sizes):.1f}")
                large_bursts = sum(1 for s in burst_sizes if s >= 10)
                if large_bursts > 0:
                    print(f"      Large bursts (10+ photos): {large_bursts}")
            
        except Exception as e:
            print(f"   [!] New burst detection failed: {e}")
            import traceback
            traceback.print_exc()
            print("   Falling back to original method...")
            USE_NEW_BURST_DETECTION = False
    
    if not USE_NEW_BURST_DETECTION:
        print("   Using ORIGINAL burst detection (phash only)...")
        
        print("   Clustering near-duplicates...")
        clusters = cluster_near_dups(good_paths, n_workers=4)
        
        # Add singletons
        path_to_cluster = {}
        for cluster_id, cluster in enumerate(clusters):
            for p in cluster:
                path_to_cluster[str(p)] = cluster_id
        
        max_cluster_id = len(clusters)
        for p in good_paths:
            if str(p) not in path_to_cluster:
                clusters.append([p])
                path_to_cluster[str(p)] = max_cluster_id
                max_cluster_id += 1
        
        print(f"   [OK] Found {len(clusters)} scenes (ORIGINAL method)")
        multi = sum(1 for c in clusters if len(c) > 1)
        print(f"      Burst scenes (2+ photos): {multi}")
        print(f"      Single photos: {len(clusters) - multi}")
    
    with open(SCENE_CLUSTERS_FILE, 'wb') as f:
        pickle.dump(clusters, f)
    
    return clusters

# ========================== SAMPLING STRATEGY ==========================
class PairwiseSampler:
    """Sample photo pairs and single photos for labeling."""
    
    def __init__(self, scene_clusters, labels):
        self.scene_clusters = scene_clusters
        self.labels = labels
        self.rng = random.Random(RANDOM_SEED)
        
        # Separate burst and single scenes
        self.burst_scenes = [c for c in scene_clusters if len(c) >= 2]
        self.single_scenes = [c for c in scene_clusters if len(c) == 1]
    
    def get_labeled_photos(self):
        """Get set of all photos that have been labeled."""
        labeled = set()
        
        # From pairwise comparisons
        for comp in self.labels["pairwise"]:
            labeled.add(comp["photo_a"])
            labeled.add(comp["photo_b"])
        
        # From single labels
        labeled.update(self.labels["single"].keys())
        
        # From gallery selections
        for gallery in self.labels.get("gallery", []):
            labeled.update(gallery["photos"])
        
        return labeled
    
    def get_next_samples(self, n=10):
        """Get next n samples (mix of within-cluster, between-cluster, and gallery)."""
        samples = []
        labeled_photos = self.get_labeled_photos()
        
        # Separate large and small bursts
        large_bursts = [c for c in self.burst_scenes if len(c) >= LARGE_BURST_THRESHOLD]
        small_bursts = [c for c in self.burst_scenes if 2 <= len(c) < LARGE_BURST_THRESHOLD]
        
        # Calculate how many of each type
        n_gallery = int(n * GALLERY_RATIO) if ENABLE_GALLERY_MODE and large_bursts else 0
        n_within = int(n * WITHIN_CLUSTER_RATIO)
        n_between = n - n_gallery - n_within
        
        print(f"\n[SAMPLING] Batch of {n}: {n_within} within-cluster, {n_between} between-cluster, {n_gallery} gallery")
        
        # 1. GALLERY MODE: Large bursts (5+ photos) - show all at once
        if n_gallery > 0:
            unlabeled_large_bursts = []
            for cluster in large_bursts:
                unlabeled_in_burst = [p for p in cluster if str(p) not in labeled_photos]
                if len(unlabeled_in_burst) >= 3:
                    unlabeled_large_bursts.append(unlabeled_in_burst)
            
            if unlabeled_large_bursts:
                selected = self.rng.sample(
                    unlabeled_large_bursts,
                    min(n_gallery, len(unlabeled_large_bursts))
                )
                
                for burst in selected:
                    samples.append({
                        "type": "gallery",
                        "photos": burst[:20],  # Show max 20 at once
                        "burst_size": len(burst)
                    })
        
        # 2. WITHIN-CLUSTER: Pairwise from same burst
        if n_within > 0:
            unlabeled_bursts = []
            for cluster in small_bursts:
                unlabeled_in_burst = [p for p in cluster if str(p) not in labeled_photos]
                if len(unlabeled_in_burst) >= 2:
                    unlabeled_bursts.append(unlabeled_in_burst)
            
            if unlabeled_bursts:
                selected_bursts = self.rng.sample(
                    unlabeled_bursts,
                    min(n_within, len(unlabeled_bursts))
                )
                
                for burst in selected_bursts:
                    pair = self.rng.sample(burst, 2)
                    samples.append({
                        "type": "pairwise_within",
                        "photo_a": pair[0],
                        "photo_b": pair[1],
                        "burst_size": len(burst),
                        "comparison_type": "within"
                    })
        
        # 3. BETWEEN-CLUSTER: Pairwise from different bursts
        if n_between > 0:
            all_bursts_with_unlabeled = []
            for cluster in (small_bursts + large_bursts):
                unlabeled_in_burst = [p for p in cluster if str(p) not in labeled_photos]
                if len(unlabeled_in_burst) >= 1:
                    all_bursts_with_unlabeled.append(unlabeled_in_burst)
            
            if len(all_bursts_with_unlabeled) >= 2:
                for _ in range(min(n_between, len(all_bursts_with_unlabeled) // 2)):
                    burst_a, burst_b = self.rng.sample(all_bursts_with_unlabeled, 2)
                    photo_a = self.rng.choice(burst_a)
                    photo_b = self.rng.choice(burst_b)
                    
                    samples.append({
                        "type": "pairwise_between",
                        "photo_a": photo_a,
                        "photo_b": photo_b,
                        "comparison_type": "between"
                    })
        
        # Shuffle to mix types
        self.rng.shuffle(samples)
        
        return samples[:n]

# ========================== GEMINI INTEGRATION ==========================
def get_gemini_comparison(photo_a, photo_b, cache):
    """Get pairwise comparison description from Gemini."""
    cache_key = f"{get_cache_key(Path(photo_a))}_{get_cache_key(Path(photo_b))}"
    
    if cache_key in cache:
        return cache[cache_key]
    
    if not GEMINI_API_KEY:
        return "[!] Gemini API key not set"
    
    try:
        img_a = fix_image_orientation(photo_a, max_size=800)
        img_b = fix_image_orientation(photo_b, max_size=800)
        
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = """Compare these two photos:

Photo A: [Brief description - technical quality and content]
Photo B: [Brief description - technical quality and content]

Key Differences: [What makes them different]

Be concise and specific."""
        
        response = model.generate_content([prompt, img_a, img_b])
        description = response.text
        
        cache[cache_key] = description
        return description
    
    except Exception as e:
        error_msg = f"[NO] Gemini error: {str(e)}"
        cache[cache_key] = error_msg
        return error_msg

# ========================== GRADIO UI ==========================
def create_labeling_ui(session):
    """Create Gradio interface for pairwise labeling."""
    
    sampler = PairwiseSampler(session.scene_clusters, session.labels)
    current_samples = []
    current_idx = [0]
    
    def load_next_batch():
        """Load next batch of samples."""
        nonlocal current_samples
        current_samples = sampler.get_next_samples(n=20)
        current_idx[0] = 0
        return show_current_sample()
    
    def show_current_sample():
        """Show current sample (pairwise, gallery, or single)."""
        if current_idx[0] >= len(current_samples):
            return (
                gr.update(visible=False),  # Hide pairwise row
                gr.update(visible=False),  # Hide gallery
                gr.update(visible=False, choices=[]),  # Hide checkboxes
                gr.update(visible=False),  # Hide submit btn
                None, None,  # Clear images
                "[OK] Batch complete! Load next batch.",  # Context
                "",  # Gemini
                session.get_label_stats(),  # Stats
                ""  # Reason
            )
        
        sample = current_samples[current_idx[0]]
        
        if sample["type"] == "gallery":
            # GALLERY MODE
            photos = sample["photos"]
            burst_size = sample.get("burst_size", len(photos))
            
            context_info = f"""[GALLERY] Gallery Mode {current_idx[0] + 1}/{len(current_samples)}

[PHOTO] Reviewing burst of {len(photos)} photos (total burst: {burst_size})
[!] Select ALL keepers (photos worth sharing)
[!] Look for: best expressions, focus, composition

Instructions:
1. Review photos in gallery
2. Check boxes for photos you'd share
3. Click Submit Selection
"""
            
            # Prepare gallery images
            gallery_images = []
            for photo in photos:
                try:
                    img = fix_image_orientation(photo, max_size=600)
                    gallery_images.append((img, f"{photo.name}"))
                except:
                    pass
            
            gemini_desc = "[!] Look for: best expressions, sharpness, composition, eyes open"
            
            return (
                gr.update(visible=False),  # Hide pairwise
                gr.update(visible=True, value=gallery_images),  # Show gallery
                gr.update(visible=True, choices=[f"Photo #{i+1}: {p.name[:30]}" for i, p in enumerate(photos)], value=[]),  # Show checkboxes
                gr.update(visible=True),  # Show submit button
                None, None,  # Hide main images
                context_info,
                gemini_desc,
                session.get_label_stats(),
                ""  # Clear reason
            )
        
        elif sample["type"] in ["pairwise_within", "pairwise_between"]:
            # PAIRWISE COMPARISON
            photo_a = sample["photo_a"]
            photo_b = sample["photo_b"]
            comparison_type = sample.get("comparison_type", "within")
            
            if comparison_type == "within":
                context_info = f"""[PAIR] Within-Burst Comparison {current_idx[0] + 1}/{len(current_samples)}

[PHOTO] Comparing 2 photos from SAME burst (burst size: {sample.get('burst_size', 2)})
[!] Fine-grained quality: Which is technically better?
[!] Look for: sharpness, expression, composition
"""
            else:
                context_info = f"""[PAIR] Between-Scene Comparison {current_idx[0] + 1}/{len(current_samples)}

[PHOTO] Comparing 2 photos from DIFFERENT scenes
[!] Coarse-grained preference: Which type do you prefer?
[!] This helps learn your overall taste/style
"""
            
            # Get Gemini comparison
            gemini_desc = get_gemini_comparison(photo_a, photo_b, session.gemini_cache)
            
            try:
                img_a = fix_image_orientation(photo_a)
                img_b = fix_image_orientation(photo_b)
                
                return (
                    gr.update(visible=True),   # Show pairwise buttons
                    gr.update(visible=False),  # Hide gallery
                    gr.update(visible=False, choices=[]),  # Hide checkboxes
                    gr.update(visible=False),  # Hide submit btn
                    img_a, img_b,
                    context_info,
                    f"[AI] Gemini Analysis:\n{gemini_desc}",
                    session.get_label_stats(),
                    ""  # Clear reason
                )
            except Exception as e:
                return (
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False, choices=[]),
                    gr.update(visible=False),
                    None, None,
                    f"[NO] Error loading photos: {e}",
                    "",
                    session.get_label_stats(),
                    ""
                )
        
        else:
            return (
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False, choices=[]),
                gr.update(visible=False),
                None, None,
                f"[NO] Unknown sample type: {sample.get('type')}",
                "",
                session.get_label_stats(),
                ""
            )
    
    def label_gallery(selected_photos, reason):
        """Label a gallery/burst selection."""
        if current_idx[0] >= len(current_samples):
            return show_current_sample()
        
        sample = current_samples[current_idx[0]]
        if sample["type"] != "gallery":
            return show_current_sample()
        
        # Parse selected indices from checkbox values
        selected_indices = []
        for selection in selected_photos:
            try:
                # "Photo #1: filename" -> 0
                idx = int(selection.split("#")[1].split(":")[0]) - 1
                selected_indices.append(idx)
            except:
                pass
        
        print(f"[GALLERY] Selected {len(selected_indices)} of {len(sample['photos'])} photos")
        
        session.add_gallery_label(
            sample["photos"],
            selected_indices,
            reason
        )
        
        current_idx[0] += 1
        
        if current_idx[0] % 5 == 0:
            session.save_gemini_cache()
        
        return show_current_sample()
    
    def label_pairwise(choice, reason):
        """Label a pairwise comparison."""
        if current_idx[0] >= len(current_samples):
            return show_current_sample()
        
        sample = current_samples[current_idx[0]]
        if sample["type"] not in ["pairwise_within", "pairwise_between"]:
            return show_current_sample()
        
        # Extract comparison type from sample
        comparison_type = sample.get("comparison_type", None)
        
        session.add_pairwise_label(
            sample["photo_a"],
            sample["photo_b"],
            choice,
            reason,
            comparison_type=comparison_type  # Pass context for weighted training
        )
        
        current_idx[0] += 1
        
        if current_idx[0] % 5 == 0:
            session.save_gemini_cache()
        
        return show_current_sample()
    
    def skip_sample():
        """Skip current sample."""
        current_idx[0] += 1
        return show_current_sample()
    
    # Create UI
    with gr.Blocks(title="Photo Taste Trainer - Burst-Aware v4", css="""
        .side-by-side {display: flex; gap: 10px;}
        .photo-container {flex: 1;}
    """) as demo:
        gr.Markdown("# [PHOTO] Photo Taste Trainer - Burst-Aware v4")
        gr.Markdown("Compare photos to teach Claude your taste (within bursts + between scenes + weighted training)")
        
        with gr.Row():
            with gr.Column(scale=3):
                # Pairwise comparison buttons (top)
                pairwise_row = gr.Row(visible=False)
                with pairwise_row:
                    both_btn = gr.Button("[OK] Share Both", variant="primary", size="lg")
                    left_btn = gr.Button("[<] Share Left", variant="secondary", size="lg")
                    right_btn = gr.Button("[>] Share Right", variant="secondary", size="lg")
                    neither_btn = gr.Button("[X] Share Neither", variant="stop", size="lg")
                
                # Side-by-side images (for pairwise)
                with gr.Row(visible=True) as images_row:
                    image_a = gr.Image(label="Photo A (Left)", type="pil", height=400)
                    image_b = gr.Image(label="Photo B (Right)", type="pil", height=400)
                
                # Gallery view (for large bursts)
                gallery_display = gr.Gallery(
                    label="Burst Photos - Click to enlarge",
                    show_label=True,
                    columns=4,
                    rows=5,
                    height="auto",
                    object_fit="contain",
                    visible=False
                )
                
                # Gallery controls
                gallery_checkboxes = gr.CheckboxGroup(
                    label="[!] Select ALL keepers (check all photos you'd share)",
                    choices=[],
                    visible=False
                )
                
                gallery_submit_btn = gr.Button(
                    "[OK] Submit Selection",
                    variant="primary",
                    size="lg",
                    visible=False
                )
                
                # Common controls
                with gr.Row():
                    skip_btn = gr.Button("[ ] Skip", size="sm")
                
                reason_input = gr.Textbox(
                    label="Why? (optional but helpful)",
                    placeholder="e.g., 'Left has better lighting' or 'Best expressions: 1,3,5'",
                    lines=2
                )
            
            with gr.Column(scale=1):
                stats_display = gr.Markdown(session.get_label_stats())
                context_display = gr.Markdown("Load a batch to start")
                gemini_display = gr.Markdown("")
                
                gr.Markdown("---")
                
                load_batch_btn = gr.Button("[SEARCH] Load Next Batch", variant="primary")
                
                gr.Markdown("""
### [!] Tips:

**[GALLERY] Gallery Mode** (5+ photos):
- Review all photos in burst
- CHECK ALL keepers you'd share
- Look for: expressions, focus, composition

**[PAIR] Within-Burst**:
- Fine-grained quality
- Which is sharper/better?
- Weighted 2x in training

**[PAIR] Between-Scene**:
- Coarse preference
- Which type do you prefer?
- Weighted 1x in training

**Buttons**:
- **Both**: Both share-worthy
- **Left/Right**: One is better
- **Neither**: Neither is good
""")
        
        # Define outputs list
        all_outputs = [
            pairwise_row,                # Pairwise buttons visibility
            gallery_display,             # Gallery component
            gallery_checkboxes,          # Gallery checkboxes
            gallery_submit_btn,          # Gallery submit button
            image_a, image_b,            # Main images
            context_display,             # Context text
            gemini_display,              # Gemini text
            stats_display,               # Stats
            reason_input                 # Reason textbox
        ]
        
        # Event handlers - Gallery
        gallery_submit_btn.click(
            label_gallery,
            inputs=[gallery_checkboxes, reason_input],
            outputs=all_outputs
        )
        
        # Event handlers - Pairwise
        both_btn.click(
            lambda r: label_pairwise("both", r),
            inputs=[reason_input],
            outputs=all_outputs
        )
        
        left_btn.click(
            lambda r: label_pairwise("left", r),
            inputs=[reason_input],
            outputs=all_outputs
        )
        
        right_btn.click(
            lambda r: label_pairwise("right", r),
            inputs=[reason_input],
            outputs=all_outputs
        )
        
        neither_btn.click(
            lambda r: label_pairwise("neither", r),
            inputs=[reason_input],
            outputs=all_outputs
        )
        
        # Common controls
        skip_btn.click(
            skip_sample,
            outputs=all_outputs
        )
        
        load_batch_btn.click(
            load_next_batch,
            outputs=all_outputs
        )
    
    return demo

# ========================== MAIN ==========================
def main():
    print("="*60)
    print("PHOTO TASTE TRAINER - BURST-AWARE v4 (FIXED)")
    print("="*60)
    print("\n[OK] Features:")
    print("  - Gallery mode with multi-select")
    print("  - Within-burst vs between-scene comparisons")
    print("  - Weighted training (2x for within, 1x for between)")
    print("  - Comparison type tracking\n")
    
    if not GEMINI_API_KEY:
        print("\n[!] WARNING: GEMINI_API_KEY not set!")
        print("   Gemini features will be disabled")
    else:
        print(f"\n[OK] Gemini API key configured")
    
    print("\n[SEARCH] Loading session...")
    session = PairwiseLabelingSession()
    
    print("\n[SEARCH] Loading photos from Holding Cell...")
    photos = list_images(UNLABELED)
    print(f"   Found {len(photos)} photos")
    
    if len(photos) == 0:
        print("[NO] No photos found in Holding Cell!")
        return
    
    session.scene_clusters = build_scene_clusters(photos)
    
    print(f"\n{session.get_label_stats()}")
    
    print("\n[OK] Launching burst-aware labeling interface...")
    print("   Open the URL below in your browser")
    
    demo = create_labeling_ui(session)
    demo.launch(share=False, inbrowser=True)

if __name__ == "__main__":
    main()
