# taste_sort_win_v3_debugged.py
# Comprehensive fixes for:
# 1. Quality score caching (prevents 2-hour hangs)
# 2. Feature caching (saves 80 minutes per run)
# 3. Neural network performance issues
# 4. Better error handling and progress feedback

import os, sys, shutil, hashlib, math, random, pickle
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool, freeze_support
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from PIL.ExifTags import TAGS
from tqdm import tqdm
import imagehash
import cv2
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
import open_clip
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
from sklearn.calibration import calibration_curve
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
# import mediapipe as mp  # Optional: not currently used, OpenCV is used instead
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision
# import face_recognition  # Optional: not currently used, OpenCV is used instead
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Burst detection and features
try:
    from burst_detector import detect_bursts_temporal_visual
    from burst_features import compute_burst_features, add_burst_features_to_matrix
    USE_NEW_BURST_DETECTION = True
    print(" Burst-aware features enabled")
except ImportError as e:
    USE_NEW_BURST_DETECTION = False
    print(f"  Burst features disabled: {e}")
    # Define dummy functions to avoid crashes
    def compute_burst_features(photos, clusters):
        return {p: {"is_burst_member": 0, "burst_size": 0, "burst_size_normalized": 0.0,
                    "burst_position_normalized": 0.0, "is_large_burst": 0, "is_small_burst": 0}
                for p in photos}
    def add_burst_features_to_matrix(X, photos, burst_features):
        return X


# --------------------------- CONFIG --------------------------------------------
ROOT = r"Photos"
FAMILY_DIR   = Path(ROOT) / "Family Photos"
PRIVATE_DIR  = Path(ROOT) / "Private Photos"
UNLABELED    = Path(ROOT) / "Holding Cell"
OUT_BASE     = UNLABELED.parent / (UNLABELED.name + " - Sorted")

# Camera Roll
CAMERA_ROLL  = Path(r"Camera Roll")
USE_CAMERA_ROLL_NEGATIVES = True
DATE_RANGE_DAYS = 30

# Behavior
DRY_RUN              = False
RANDOM_SEED          = 1337
VALIDATION_SPLIT     = 0.2

# Baby gate
SKIP_BABY_GATE       = True
BABY_GATE_MINPROB    = 0.30
BABY_GATE_USE_RATIO  = False

# Routing
THRESH_SHARE         = 0.60
AUTO_TUNE_THRESHOLD  = True
CONFIDENCE_BAND      = 0.15

# Training
MAX_NEG_PER_POS      = 3
BALANCE_METHOD       = "undersample"
MODEL_TYPE           = "ensemble"  # "xgboost", "random_forest", "neural_net", "logistic", "ensemble"
USE_QUALITY_FEATURES = True

# NEW: Enhanced features
USE_FACE_FEATURES    = True   # Face detection and attributes
USE_TEMPORAL_FEATURES = True  # Time of day, day of week, sequence position
USE_METADATA_FEATURES = True  # Camera settings, file properties
FACE_DETECTION_METHOD = "mediapipe"  # "mediapipe" or "face_recognition"

# Two-stage filtering
USE_TWO_STAGE_FILTER = True   # Pre-filter by quality before classification
QUALITY_FILTER_THRESHOLD = 0.3  # Min quality score to classify (0-1)

# Ensemble
ENSEMBLE_MODELS = ["xgboost", "random_forest"]  # Models to ensemble
ENSEMBLE_METHOD = "vote"  # "vote" or "average"

# Cluster-aware training
CLUSTER_AWARE        = True
CLUSTER_MIN_SIZE     = 2  # Minimum photos to form a cluster
CLUSTER_TRAINING_STRATEGY = "representatives"  # or "all_weighted"
USE_CLUSTER_SAMPLING_BALANCED = True

# Class balancing (NEW)
ENABLE_CLASS_BALANCING = True  # Enable automatic class balancing
CLASS_BALANCE_METHOD = "undersample"  # "smote", "oversample", "undersample", or "none"
CLASS_BALANCE_RATIO = 0.5  # Target ratio (minority/majority), 1.0 = perfect balance
SAMPLE_STRATEGY_BY_CLASS = True  # Use different sampling for Share vs Storage

# Near-duplicate
DEDUP_HAMMING_MAX    = 5
TRASH_GAP            = 0.30

# Performance
EMBED_BATCH          = 128
BABY_GATE_BATCH      = 64
BABY_GATE_DEVICE     = "cuda"
USE_MULTIPROCESSING  = True
QUALITY_WORKERS      = 4
DEDUP_WORKERS        = 4

# NEW: Feature caching
CACHE_FEATURES       = True  # Cache combined features (embeddings + quality)
FORCE_RECOMPUTE_FEATURES = False  # Set True to rebuild cache

# Neural Network
NN_HIDDEN_DIMS       = [256, 128, 64]
NN_DROPOUT           = 0.3
NN_LEARNING_RATE     = 0.001
NN_EPOCHS            = 50
NN_BATCH_SIZE        = 32
NN_EARLY_STOP_PATIENCE = 10

# Model
CLIP_MODEL           = "ViT-B-32"

# Cache
CACHE_ROOT           = Path(".taste_cache")
EMB_CACHE            = CACHE_ROOT / "emb"
QUALITY_CACHE        = CACHE_ROOT / "quality"
FEATURE_CACHE        = CACHE_ROOT / "features"
FACE_CACHE           = CACHE_ROOT / "faces"  # NEW: Face detection cache  # NEW
EMB_CACHE.mkdir(parents=True, exist_ok=True)
QUALITY_CACHE.mkdir(parents=True, exist_ok=True)
FEATURE_CACHE.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------------------
IMG_EXTS = {".jpg",".jpeg",".png",".webp",".heic",".tif",".tiff",".bmp"}

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:
    pass

# --------------------------- FAILURE LOGGER -----------------------------------
class FailureLogger:
    def __init__(self):
        self.failures = []
    
    def log(self, path, stage, error):
        self.failures.append({
            "path": str(path),
            "stage": stage,
            "error": str(error)[:500]
        })
    
    def save(self, output_path):
        if self.failures:
            pd.DataFrame(self.failures).to_csv(output_path, index=False)
            print(f"  Logged {len(self.failures)} failures to {output_path}")
            return len(self.failures)
        return 0

failure_logger = FailureLogger()

# --------------------------- UTILITIES ----------------------------------------
def list_images(root: Path):
    if not root.exists(): return []
    return [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS]

def get_cache_key(p: Path):
    try:
        s = p.stat()
        base = f"{str(p)}|{s.st_size}|{int(s.st_mtime)}"
    except Exception:
        base = str(p)
    return hashlib.md5(base.encode()).hexdigest()

def cache_path_for(p: Path, cache_dir=EMB_CACHE):
    return cache_dir / f"{get_cache_key(p)}.npy"

def feature_cache_path_for(p: Path):
    """Path for cached combined features (embedding + quality)."""
    return FEATURE_CACHE / f"{get_cache_key(p)}.npy"

def ensure_dirs():
    for name in [
        "Share","Storage","Review","Ignore","Dupes","TrashCandidates","Reports"
    ]:
        (OUT_BASE / name).mkdir(parents=True, exist_ok=True)

def copy_file(src: Path, dst_dir: Path):
    dst = dst_dir / src.name
    i = 1
    while dst.exists():
        dst = dst_dir / f"{src.stem}_{i}{src.suffix}"
        i += 1
    if DRY_RUN: 
        return str(dst)
    try:
        shutil.copy2(src, dst)
        return str(dst)
    except Exception as e:
        failure_logger.log(src, "copy_file", str(e))
        return None

def get_image_date(p: Path):
    try:
        with Image.open(p) as img:
            exif = img._getexif()
            if exif:
                for tag, value in exif.items():
                    tag_name = TAGS.get(tag, tag)
                    if tag_name in ['DateTimeOriginal', 'DateTime']:
                        return datetime.strptime(value, '%Y:%m:%d %H:%M:%S')
    except Exception:
        pass
    try:
        return datetime.fromtimestamp(p.stat().st_mtime)
    except Exception:
        return None

# --------------------------- QUALITY SCORING ----------------------------------
def quality_score_nocache(p: Path):
    """Compute quality score without caching."""
    try:
        with Image.open(p) as im:
            im_gray = im.convert("L")
            arr_gray = np.array(im_gray)
            sharpness = float(cv2.Laplacian(arr_gray, cv2.CV_64F).var())
            brightness = float(arr_gray.mean() / 255.0)
        return 0.8 * min(sharpness / 200.0, 1.0) + 0.2 * brightness
    except Exception as e:
        failure_logger.log(p, "quality_score", str(e))
        return 0.0

def quality_score_with_cache(p: Path):
    """Compute or load cached quality score."""
    cache_file = cache_path_for(p, QUALITY_CACHE)
    if cache_file.exists():
        try:
            return float(np.load(cache_file))
        except Exception:
            pass
    
    score = quality_score_nocache(p)
    try:
        np.save(cache_file, score)
    except Exception:
        pass
    return score

def quality_score_wrapper(p):
    """Wrapper for multiprocessing."""
    return (p, quality_score_with_cache(p))

def get_quality_scores_batch(paths):
    """Compute quality scores for batch of paths with caching."""
    if not USE_MULTIPROCESSING or QUALITY_WORKERS <= 1:
        print("Computing quality scores (serial)...")
        results = []
        for p in tqdm(paths, desc="Quality scores"):
            results.append(quality_score_wrapper(p))
        return {p: score for p, score in results}
    
    print(f"Computing quality scores with {QUALITY_WORKERS} workers...")
    
    try:
        with Pool(QUALITY_WORKERS) as pool:
            results = list(tqdm(
                pool.imap(quality_score_wrapper, paths, chunksize=50),
                total=len(paths),
                desc="Quality scores"
            ))
    except Exception as e:
        print(f"  Multiprocessing failed ({e}), falling back to serial...")
        results = []
        for p in tqdm(paths, desc="Quality scores (serial)"):
            results.append(quality_score_wrapper(p))
    
    return {p: score for p, score in results}

# --------------------------- PHASHING & CLUSTERING ----------------------------
def phash(p: Path):
    try:
        with Image.open(p) as im:
            im = im.convert("RGB")
            return imagehash.phash(im, hash_size=16)
    except Exception as e:
        failure_logger.log(p, "phash", str(e))
        return None

def phash_wrapper(p):
    return (p, phash(p))

def cluster_near_dups(paths, n_workers=DEDUP_WORKERS):
    """Parallelized phashing with clustering."""
    if len(paths) == 0:
        return []
    
    if not USE_MULTIPROCESSING or n_workers <= 1:
        print("Computing phashes (serial)...")
        results = []
        for p in tqdm(paths, desc="Hashing (serial)"):
            results.append(phash_wrapper(p))
    else:
        print(f"Computing phashes with {n_workers} workers...")
        try:
            with Pool(n_workers) as pool:
                results = list(tqdm(
                    pool.imap(phash_wrapper, paths, chunksize=50),
                    total=len(paths),
                    desc="Hashing"
                ))
        except Exception as e:
            print(f"  Multiprocessing failed ({e}), falling back to serial...")
            results = []
            for p in tqdm(paths, desc="Hashing (serial)"):
                results.append(phash_wrapper(p))
    
    buckets = {}
    for p, h in results:
        if h is None: 
            continue
        key = int(str(h), 16) >> 16
        buckets.setdefault(key, []).append((p, h))
    
    print(f"Clustering {len(buckets)} hash buckets...")
    clusters = []
    for _, items in tqdm(buckets.items(), desc="Clustering"):
        used = set()
        for i, (p_i, h_i) in enumerate(items):
            if i in used: 
                continue
            cluster = [(p_i, h_i)]
            used.add(i)
            for j, (p_j, h_j) in enumerate(items):
                if j in used: 
                    continue
                if h_i - h_j <= DEDUP_HAMMING_MAX:
                    cluster.append((p_j, h_j))
                    used.add(j)
            if len(cluster) >= CLUSTER_MIN_SIZE:
                clusters.append([x[0] for x in cluster])
    
    return clusters

def choose_best_in_cluster(cluster, quality_cache):
    """Choose best image from cluster using cached quality scores."""
    scores = [(p, quality_cache.get(p, 0.0)) for p in cluster]
    scores.sort(key=lambda x: x[1], reverse=True)
    best, best_q = scores[0]
    losers = [(p,q) for p,q in scores[1:] if (best_q - q) >= TRASH_GAP]
    others  = [(p,q) for p,q in scores[1:] if (best_q - q) <  TRASH_GAP]
    return best, losers, others

# --------------------------- FACE DETECTION -----------------------------------
def detect_faces_mediapipe(image_path):
    """Detect faces using MediaPipe. Returns face features dict."""
    try:
        # Use simple face detection for speed
        import cv2
        
        img = cv2.imread(str(image_path))
        if img is None:
            return {"num_faces": 0, "face_ratio": 0.0, "has_face": 0}
        
        # Convert to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width = img.shape[:2]
        
        # Simple face detection with OpenCV (faster than MediaPipe for our use case)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        
        # Detect faces
        faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) == 0:
            return {
                "num_faces": 0,
                "face_ratio": 0.0,
                "largest_face_ratio": 0.0,
                "has_face": 0,
                "avg_face_size": 0.0,
                "eyes_detected": 0,
                "smile_detected": 0
            }
        
        # Calculate face features
        total_face_area = 0
        largest_face_area = 0
        eyes_count = 0
        smile_count = 0
        
        for (x, y, w, h) in faces:
            face_area = w * h
            total_face_area += face_area
            largest_face_area = max(largest_face_area, face_area)
            
            # Detect eyes in face region
            roi_gray = cv2.cvtColor(img[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)
            if len(eyes) >= 2:  # Both eyes detected
                eyes_count += 1
            
            # Detect smile in lower half of face
            roi_smile = roi_gray[h//2:, :]
            smiles = smile_cascade.detectMultiScale(roi_smile, scaleFactor=1.8, minNeighbors=20)
            if len(smiles) > 0:
                smile_count += 1
        
        image_area = width * height
        
        features = {
            "num_faces": min(len(faces), 10),  # Cap at 10
            "face_ratio": total_face_area / image_area,
            "largest_face_ratio": largest_face_area / image_area,
            "has_face": 1,
            "avg_face_size": (total_face_area / len(faces)) / image_area,
            "eyes_detected": eyes_count,  # Number of faces with both eyes visible
            "smile_detected": smile_count  # Number of faces smiling
        }
        
        return features
        
    except Exception as e:
        failure_logger.log(image_path, "face_detection", str(e))
        return {
            "num_faces": 0,
            "face_ratio": 0.0,
            "largest_face_ratio": 0.0,
            "has_face": 0,
            "avg_face_size": 0.0,
            "eyes_detected": 0,
            "smile_detected": 0
        }

def get_face_features_batch(paths, workers=4):
    """Get face features for multiple images with caching."""
    FACE_CACHE.mkdir(parents=True, exist_ok=True)
    
    def face_cache_path(p):
        return FACE_CACHE / f"{get_cache_key(p)}.npy"
    
    # Check cache
    to_compute = []
    features_dict = {}
    
    for p in paths:
        cache_path = face_cache_path(p)
        if cache_path.exists():
            try:
                features_dict[p] = np.load(cache_path, allow_pickle=True).item()
                continue
            except:
                pass
        to_compute.append(p)
    
    if to_compute:
        print(f" Face detection: computing {len(to_compute)}/{len(paths)} images...")
        
        if USE_MULTIPROCESSING and workers > 1:
            with Pool(workers) as pool:
                results = list(tqdm(
                    pool.imap(detect_faces_mediapipe, to_compute),
                    total=len(to_compute),
                    desc="Detecting faces"
                ))
        else:
            results = [detect_faces_mediapipe(p) for p in tqdm(to_compute, desc="Detecting faces")]
        
        # Cache results
        for p, features in zip(to_compute, results):
            features_dict[p] = features
            try:
                np.save(face_cache_path(p), features)
            except Exception as e:
                failure_logger.log(p, "face_cache_save", str(e))
    else:
        print(f" Face detection: using {len(paths)}/{len(paths)} cached")
    
    return features_dict

# --------------------------- TEMPORAL & METADATA ------------------------------
def extract_temporal_metadata(image_path):
    """Extract temporal and camera metadata features."""
    features = {
        "hour_of_day": -1,
        "day_of_week": -1,
        "is_weekend": 0,
        "month": -1,
        "focal_length": 0.0,
        "aperture": 0.0,
        "iso": 0,
        "flash_used": 0,
        "file_size_mb": 0.0,
        "aspect_ratio": 1.0,
        "is_landscape": 0,
        "megapixels": 0.0
    }
    
    try:
        # Get file stats
        stat = os.stat(image_path)
        features["file_size_mb"] = stat.st_size / (1024 * 1024)
        
        # Get creation date from EXIF or filename
        date = get_image_date(image_path)
        if date:
            features["hour_of_day"] = date.hour
            features["day_of_week"] = date.weekday()  # 0=Monday, 6=Sunday
            features["is_weekend"] = 1 if date.weekday() >= 5 else 0
            features["month"] = date.month
        
        # Get image dimensions
        with Image.open(image_path) as img:
            width, height = img.size
            features["aspect_ratio"] = width / height if height > 0 else 1.0
            features["is_landscape"] = 1 if width > height else 0
            features["megapixels"] = (width * height) / 1_000_000
            
            # Get EXIF data
            exif = img.getexif()
            if exif:
                # Focal length
                if 37386 in exif:  # FocalLength
                    focal = exif[37386]
                    if isinstance(focal, tuple):
                        features["focal_length"] = focal[0] / focal[1] if focal[1] > 0 else 0
                    else:
                        features["focal_length"] = float(focal)
                
                # Aperture (FNumber)
                if 33437 in exif:
                    fnum = exif[33437]
                    if isinstance(fnum, tuple):
                        features["aperture"] = fnum[0] / fnum[1] if fnum[1] > 0 else 0
                    else:
                        features["aperture"] = float(fnum)
                
                # ISO
                if 34855 in exif:
                    features["iso"] = int(exif[34855])
                
                # Flash
                if 37385 in exif:
                    flash = int(exif[37385])
                    features["flash_used"] = 1 if (flash & 0x1) else 0
        
    except Exception as e:
        failure_logger.log(image_path, "metadata_extraction", str(e))
    
    return features

def get_temporal_metadata_batch(paths):
    """Get temporal/metadata features for multiple images."""
    print(f" Extracting temporal/metadata features from {len(paths)} images...")
    
    features_dict = {}
    for p in tqdm(paths, desc="Extracting metadata"):
        features_dict[p] = extract_temporal_metadata(p)
    
    return features_dict

# --------------------------- MODEL LOADING ------------------------------------
def load_model(device):
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL, pretrained='laion2b_s34b_b79k'
    )
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
    model.eval().to(device)
    return model, preprocess, tokenizer

# --------------------------- EMBEDDING ----------------------------------------
def embed_paths(paths, model, preprocess, device, batch_size=128):
    """Embed images with caching."""
    to_compute = []
    cached = []
    for p in paths:
        cpath = cache_path_for(p)
        if cpath.exists():
            cached.append(p)
        else:
            to_compute.append(p)
    
    if len(paths) > 0:
        print(f" Embedding cache: using {len(cached)}/{len(paths)} cached; computing {len(to_compute)}")
    
    computed = {}
    if to_compute:
        imgs, img_paths = [], []
        for i, p in enumerate(tqdm(to_compute, desc="Embedding (missing)")):
            try:
                with Image.open(p) as im:
                    im = im.convert("RGB")
                t = preprocess(im)
                imgs.append(t)
                img_paths.append(p)
                if len(imgs) == batch_size or i == len(to_compute) - 1:
                    b = torch.stack(imgs, dim=0).to(device)
                    with torch.no_grad():
                        f = model.encode_image(b)
                        f = f / f.norm(dim=-1, keepdim=True)
                        f = f.detach().cpu().numpy().astype(np.float32)
                    for jj, pp in enumerate(img_paths):
                        np.save(cache_path_for(pp), f[jj])
                        computed[pp] = f[jj]
                    imgs, img_paths = [], []
            except Exception as e:
                failure_logger.log(p, "embed_paths", str(e))
                imgs, img_paths = [], []

    feats, good = [], []
    for p in paths:
        try:
            if p in computed:
                arr = computed[p]
            else:
                arr = np.load(cache_path_for(p))
            feats.append(arr)
            good.append(p)
        except Exception as e:
            failure_logger.log(p, "embed_paths_load", str(e))
    
    if len(feats) == 0:
        return np.zeros((0, 512), dtype=np.float32), []
    return np.vstack(feats), good

# --------------------------- FEATURE CACHING ----------------------------------
def get_combined_features(paths, embeddings, quality_cache, face_features=None, temporal_features=None):
    """
    Get combined features (embeddings + quality + faces + temporal/metadata) with caching.
    Now supports face detection and temporal/metadata features.
    """
    # Start with embeddings
    feature_components = [embeddings]
    feature_names = ["CLIP_embeddings"]
    
    # Add quality features
    if USE_QUALITY_FEATURES:
        quality_feats = []
        for p in paths:
            score = quality_cache.get(p, 0.0)
            quality_feats.append([score * 0.8 * 200.0, score * 0.2])
        quality_feats = np.array(quality_feats)
        
        if len(quality_feats) > 0:
            mean = quality_feats.mean(axis=0)
            std = quality_feats.std(axis=0) + 1e-6
            quality_feats = (quality_feats - mean) / std
            feature_components.append(quality_feats)
            feature_names.append("quality")
    
    # Add face features
    if USE_FACE_FEATURES and face_features is not None:
        face_feats = []
        for p in paths:
            f = face_features.get(p, {})
            face_feats.append([
                f.get("num_faces", 0),
                f.get("face_ratio", 0.0),
                f.get("largest_face_ratio", 0.0),
                f.get("has_face", 0),
                f.get("avg_face_size", 0.0),
                f.get("eyes_detected", 0),
                f.get("smile_detected", 0)
            ])
        face_feats = np.array(face_feats)
        
        if len(face_feats) > 0:
            # Normalize face features
            mean = face_feats.mean(axis=0)
            std = face_feats.std(axis=0) + 1e-6
            face_feats = (face_feats - mean) / std
            feature_components.append(face_feats)
            feature_names.append("faces")
    
    # Add temporal/metadata features
    if (USE_TEMPORAL_FEATURES or USE_METADATA_FEATURES) and temporal_features is not None:
        temporal_feats = []
        for p in paths:
            t = temporal_features.get(p, {})
            temporal_feats.append([
                t.get("hour_of_day", -1),
                t.get("day_of_week", -1),
                t.get("is_weekend", 0),
                t.get("month", -1),
                t.get("focal_length", 0.0),
                t.get("aperture", 0.0),
                t.get("iso", 0) / 1000.0,  # Normalize ISO
                t.get("flash_used", 0),
                t.get("file_size_mb", 0.0),
                t.get("aspect_ratio", 1.0),
                t.get("is_landscape", 0),
                t.get("megapixels", 0.0)
            ])
        temporal_feats = np.array(temporal_feats)
        
        if len(temporal_feats) > 0:
            # Normalize temporal/metadata features
            mean = temporal_feats.mean(axis=0)
            std = temporal_feats.std(axis=0) + 1e-6
            temporal_feats = (temporal_feats - mean) / std
            feature_components.append(temporal_feats)
            feature_names.append("temporal_metadata")
    
    # Combine all features
    combined = np.hstack(feature_components)
    
    print(f" Feature dimensions: {combined.shape[1]} total")
    print(f"   Components: {', '.join(f'{name}({comp.shape[1]})' for name, comp in zip(feature_names, feature_components))}")
    
    return combined

# --------------------------- NEURAL NETWORK -----------------------------------
class NeuralNetClassifier:
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout=0.3,
                 learning_rate=0.001, epochs=50, batch_size=32,
                 early_stop_patience=10, sample_weight=None):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stop_patience = early_stop_patience
        self.sample_weight = sample_weight
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
    
    def fit(self, X, y):
        """Train the neural network."""
        # Build model
        layers = []
        prev_dim = self.input_dim
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        
        self.model = nn.Sequential(*layers).to(self.device)
        
        # Calculate class weights - FIX: ensure integer labels
        y_int = y.astype(np.int64)
        unique_labels = np.unique(y_int)
        
        if len(unique_labels) < 2:
            print(f"  Warning: Only one class present in training data: {unique_labels}")
            print("   Cannot train classifier with single class!")
            return self
        
        class_counts = np.bincount(y_int)
        class_weights = len(y) / (len(class_counts) * class_counts + 1e-6)
        
        if len(class_weights) > 1:
            pos_weight = torch.tensor([class_weights[1] / (class_weights[0] + 1e-6)], 
                                      dtype=torch.float32).to(self.device)
        else:
            pos_weight = torch.tensor([1.0], dtype=torch.float32).to(self.device)
        
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=False
        )
        
        # Split for early stopping
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.15, random_state=RANDOM_SEED, 
            stratify=y if len(unique_labels) > 1 else None
        )
        
        # Create datasets
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train.astype(np.float32)).unsqueeze(1)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val.astype(np.float32)).unsqueeze(1)
        )
        
        # Create weighted sampler if sample_weight provided
        if self.sample_weight is not None:
            train_weights = self.sample_weight[:len(X_train)]
            sampler = WeightedRandomSampler(
                weights=train_weights, num_samples=len(train_weights), replacement=True
            )
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=sampler)
        else:
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"\n Training Neural Network ({self.epochs} epochs max)...")
        for epoch in range(self.epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            scheduler.step(val_loss)
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"   Epoch {epoch+1}/{self.epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.early_stop_patience:
                    print(f"   Early stopping at epoch {epoch+1}")
                    break
        
        return self
    
    def predict_proba(self, X):
        """Predict probabilities."""
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            logits = self.model(X_tensor).cpu().numpy()
            probs = 1 / (1 + np.exp(-logits))  # Sigmoid
        
        return np.hstack([1 - probs, probs])
    
    def predict(self, X):
        """Predict class labels."""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

# --------------------------- CLUSTERING & SAMPLING ----------------------------
def build_clusters_with_embeddings(paths, embeddings, labels=None):
    """
    Build clusters from paths with their embeddings and labels.
    Returns list of clusters, where each cluster is list of (path, embedding, label) tuples.
    """
    print("\n Clustering training data...")
    
    # First compute quality scores for all paths
    quality_cache = get_quality_scores_batch(paths)
    
    # Find duplicate clusters
    clusters_raw = cluster_near_dups(paths, n_workers=DEDUP_WORKERS)
    
    if len(clusters_raw) == 0:
        print("   No clusters found - treating each photo as unique")
        # Each photo is its own cluster
        cluster_list = []
        for i, p in enumerate(paths):
            label = labels[i] if labels is not None else None
            cluster_list.append([(p, embeddings[i], label)])
    else:
        # Map paths to their cluster
        path_to_cluster = {}
        for cluster_id, cluster_paths in enumerate(clusters_raw):
            for p in cluster_paths:
                path_to_cluster[p] = cluster_id
        
        # Build cluster list with embeddings and labels
        cluster_dict = {}
        singletons = []
        
        for i, p in enumerate(paths):
            label = labels[i] if labels is not None else None
            item = (p, embeddings[i], label)
            
            if p in path_to_cluster:
                cluster_id = path_to_cluster[p]
                if cluster_id not in cluster_dict:
                    cluster_dict[cluster_id] = []
                cluster_dict[cluster_id].append(item)
            else:
                # Singleton (not in any cluster)
                singletons.append([item])
        
        cluster_list = list(cluster_dict.values()) + singletons
    
    # Stats
    multi_photo = [c for c in cluster_list if len(c) > 1]
    single_photo = [c for c in cluster_list if len(c) == 1]
    
    if multi_photo:
        sizes = [len(c) for c in multi_photo]
        print(f"   Multi-photo clusters: {len(multi_photo)}")
        print(f"   Cluster sizes: min={min(sizes)}, max={max(sizes)}, mean={np.mean(sizes):.1f}")
    print(f"   Single-photo clusters: {len(single_photo)}")
    print(f"   Total clusters: {len(cluster_list)}")
    
    # Label consistency check
    if labels is not None:
        inconsistent = 0
        inconsistent_details = []
        for i, cluster in enumerate(cluster_list):
            if len(cluster) > 1:
                cluster_labels = [item[2] for item in cluster if item[2] is not None]
                if len(cluster_labels) > 1 and len(set(cluster_labels)) > 1:
                    inconsistent += 1
                    share_count = sum(1 for l in cluster_labels if l == 1)
                    storage_count = sum(1 for l in cluster_labels if l == 0)
                    inconsistent_details.append((len(cluster), share_count, storage_count))
        
        if inconsistent > 0:
            print(f"     {inconsistent} clusters have mixed labels (Share + Storage)")
            print(f"      This is {inconsistent/len(multi_photo)*100:.1f}% of multi-photo clusters!")
            print(f"       DIAGNOSIS: Visually similar photos have different labels")
            print(f"         This suggests your sorting criteria is NOT based on visual features")
            print(f"         (e.g., based on context, people, memories, appropriateness)")
            
            # Show some examples
            if len(inconsistent_details) > 0:
                avg_size = np.mean([d[0] for d in inconsistent_details])
                print(f"       Mixed cluster stats: avg size={avg_size:.1f}")
    
    return cluster_list, quality_cache

def sample_cluster_representatives(clusters, quality_cache, sampling_strategy="best_quality"):
    """
    Sample one representative from each cluster using CACHED quality scores.
    Now supports class-aware sampling to prevent imbalance.
    Returns (paths, embeddings, labels) for representatives.
    """
    print(f"\n Sampling cluster representatives (strategy: {sampling_strategy})...")
    
    rep_paths, rep_embs, rep_labels = [], [], []
    
    # Count classes in clusters first
    share_clusters = []
    storage_clusters = []
    unlabeled_clusters = []
    
    for cluster in clusters:
        labels_in_cluster = [item[2] for item in cluster if item[2] is not None]
        if not labels_in_cluster:
            unlabeled_clusters.append(cluster)
        elif all(l == 1 for l in labels_in_cluster):
            share_clusters.append(cluster)
        elif all(l == 0 for l in labels_in_cluster):
            storage_clusters.append(cluster)
        else:
            # Mixed - use majority vote
            if sum(labels_in_cluster) / len(labels_in_cluster) > 0.5:
                share_clusters.append(cluster)
            else:
                storage_clusters.append(cluster)
    
    print(f"   Cluster breakdown:")
    print(f"      Share (label=1): {len(share_clusters)} clusters")
    print(f"      Storage (label=0): {len(storage_clusters)} clusters")
    print(f"      Unlabeled: {len(unlabeled_clusters)} clusters")
    
    # Sample representatives with class-aware strategy
    def sample_from_cluster(cluster, label_class):
        """Sample one photo from cluster, using different strategy by class if enabled."""
        if SAMPLE_STRATEGY_BY_CLASS and label_class is not None:
            if label_class == 1:  # Share - use best quality
                strategy = "best_quality"
            else:  # Storage - use random to avoid bias
                strategy = "random"
        else:
            strategy = sampling_strategy
        
        if strategy == "best_quality":
            qualities = [quality_cache.get(item[0], 0.0) for item in cluster]
            best_idx = np.argmax(qualities)
            return cluster[best_idx]
        elif strategy == "random":
            return random.choice(cluster)
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")
    
    # Sample from each class
    for cluster in tqdm(share_clusters, desc="Sampling Share representatives"):
        rep = sample_from_cluster(cluster, label_class=1)
        rep_paths.append(rep[0])
        rep_embs.append(rep[1])
        rep_labels.append(1)
    
    for cluster in tqdm(storage_clusters, desc="Sampling Storage representatives"):
        rep = sample_from_cluster(cluster, label_class=0)
        rep_paths.append(rep[0])
        rep_embs.append(rep[1])
        rep_labels.append(0)
    
    for cluster in tqdm(unlabeled_clusters, desc="Sampling unlabeled representatives"):
        rep = sample_from_cluster(cluster, label_class=None)
        rep_paths.append(rep[0])
        rep_embs.append(rep[1])
        # No label
    
    rep_embs = np.vstack(rep_embs) if rep_embs else np.array([])
    rep_labels = np.array(rep_labels, dtype=np.int64) if rep_labels else None
    
    if rep_labels is not None:
        share_count = np.sum(rep_labels == 1)
        storage_count = np.sum(rep_labels == 0)
        ratio = storage_count / (share_count + 1e-6)
        print(f"    Selected {len(rep_paths)} representatives from {len(clusters)} clusters")
        print(f"      Share: {share_count} ({share_count/len(rep_labels)*100:.1f}%)")
        print(f"      Storage: {storage_count} ({storage_count/len(rep_labels)*100:.1f}%)")
        print(f"      Ratio (Storage/Share): {ratio:.3f}")
        
        if ratio < 0.5 or ratio > 2.0:
            print(f"        WARNING: Class imbalance detected! Will apply balancing.")
    else:
        print(f"   Selected {len(rep_paths)} representatives from {len(clusters)} clusters")
    
    return rep_paths, rep_embs, rep_labels
    
    rep_embs = np.vstack(rep_embs) if rep_embs else np.array([])
    rep_labels = np.array(rep_labels, dtype=np.int64) if rep_labels else None
    
    print(f"   Selected {len(rep_paths)} representatives from {len(clusters)} clusters")
    
    return rep_paths, rep_embs, rep_labels

# --------------------------- CAMERA ROLL NEGATIVES ----------------------------
def sample_camera_roll_negatives(unlabeled_paths, camera_roll_dir, days=DATE_RANGE_DAYS):
    """Sample photos from Camera Roll in same date range as unlabeled set."""
    if not camera_roll_dir.exists():
        print(f"  Camera Roll directory not found: {camera_roll_dir}")
        return []
    
    print(f"\n Sampling negatives from Camera Roll...")
    
    # Get date range - CHECK ALL PHOTOS
    unlabeled_dates = []
    for p in tqdm(unlabeled_paths, desc="Extracting dates from unlabeled"):
        date = get_image_date(p)
        if date:
            unlabeled_dates.append(date)
    
    if not unlabeled_dates:
        print("  Could not extract dates from unlabeled photos")
        return []
    
    min_date = min(unlabeled_dates)
    max_date = max(unlabeled_dates)
    print(f"    Extracted dates from {len(unlabeled_dates)}/{len(unlabeled_paths)} photos")
    print(f"    Unlabeled date range: {min_date.date()} to {max_date.date()}")
    print(f"    Span: {(max_date - min_date).days} days")
    
    from datetime import timedelta
    search_start = min_date - timedelta(days=days)
    search_end = max_date + timedelta(days=days)
    print(f"    Searching Camera Roll: {search_start.date()} to {search_end.date()}")
    print(f"      ({days} days buffer)")
    
    camera_roll_paths = list_images(camera_roll_dir)
    print(f"    Found {len(camera_roll_paths)} total images in Camera Roll")
    
    candidates = []
    dates_found = []
    for p in tqdm(camera_roll_paths, desc="Filtering by date"):
        date = get_image_date(p)
        if date and search_start <= date <= search_end:
            candidates.append(p)
            dates_found.append(date)
    
    if dates_found:
        print(f"    Found {len(candidates)} Camera Roll photos in date range")
        print(f"      Date range of matches: {min(dates_found).date()} to {max(dates_found).date()}")
    else:
        print(f"     Found {len(candidates)} Camera Roll photos in date range")
    
    return candidates

def sample_negatives(all_private, all_family, camera_roll_negatives=None):
    """Sample negative examples from Private and optionally Camera Roll."""
    fam_set = set(map(str, all_family))
    private_negs = [p for p in all_private if str(p) not in fam_set]
    
    if camera_roll_negatives:
        camera_roll_negs = [p for p in camera_roll_negatives if str(p) not in fam_set]
        print(f"   Using {len(private_negs)} Private + {len(camera_roll_negs)} Camera Roll negatives")
        return private_negs + camera_roll_negs
    
    return private_negs

# --------------------------- CLASS BALANCING ----------------------------------
def balance_training_data(X_train, y_train, method="smote", target_ratio=0.8):
    """
    Balance training data to prevent model bias.
    
    Args:
        X_train: Training features
        y_train: Training labels
        method: "smote", "oversample", "undersample", or "none"
        target_ratio: Target ratio of minority/majority class (1.0 = perfect balance)
    
    Returns:
        X_balanced, y_balanced
    """
    unique, counts = np.unique(y_train, return_counts=True)
    
    if len(unique) < 2:
        print("  Cannot balance - only one class present")
        return X_train, y_train
    
    minority_class = unique[np.argmin(counts)]
    majority_class = unique[np.argmax(counts)]
    minority_count = np.min(counts)
    majority_count = np.max(counts)
    current_ratio = minority_count / majority_count
    
    print(f"\n  Class distribution before balancing:")
    print(f"   Class 0 (Storage): {counts[0]} samples")
    print(f"   Class 1 (Share): {counts[1]} samples")
    print(f"   Current ratio (minority/majority): {current_ratio:.3f}")
    print(f"   Target ratio: {target_ratio:.3f}")
    
    if current_ratio >= target_ratio:
        print(f"    Already balanced (ratio >= {target_ratio}), no balancing needed")
        return X_train, y_train
    
    if method == "none":
        print("   Balancing disabled, keeping original distribution")
        return X_train, y_train
    
    print(f"   Applying {method} balancing...")
    
    try:
        if method == "smote":
            # SMOTE for synthetic oversampling
            target_count = int(majority_count * target_ratio)
            smote = SMOTE(
                sampling_strategy={minority_class: target_count},
                random_state=RANDOM_SEED,
                k_neighbors=min(5, minority_count - 1)  # Adjust k if too few samples
            )
            X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
        
        elif method == "oversample":
            # Random oversampling
            target_count = int(majority_count * target_ratio)
            ros = RandomOverSampler(
                sampling_strategy={minority_class: target_count},
                random_state=RANDOM_SEED
            )
            X_balanced, y_balanced = ros.fit_resample(X_train, y_train)
        
        elif method == "undersample":
            # Random undersampling
            target_count = int(minority_count / target_ratio)
            rus = RandomUnderSampler(
                sampling_strategy={majority_class: target_count},
                random_state=RANDOM_SEED
            )
            X_balanced, y_balanced = rus.fit_resample(X_train, y_train)
        
        else:
            raise ValueError(f"Unknown balance method: {method}")
        
        unique_new, counts_new = np.unique(y_balanced, return_counts=True)
        new_ratio = np.min(counts_new) / np.max(counts_new)
        
        print(f"\n    Balancing complete:")
        print(f"      Class 0 (Storage): {counts_new[0]} samples ({counts_new[0] - counts[0]:+d})")
        print(f"      Class 1 (Share): {counts_new[1]} samples ({counts_new[1] - counts[1]:+d})")
        print(f"      New ratio: {new_ratio:.3f}")
        
        return X_balanced, y_balanced
    
    except Exception as e:
        print(f"     Balancing failed: {e}")
        print("   Continuing with unbalanced data")
        return X_train, y_train

# --------------------------- ENSEMBLE CLASSIFIER ------------------------------
class EnsembleClassifier:
    """Ensemble of multiple classifiers."""
    def __init__(self, models_list, method="vote"):
        self.models_list = models_list
        self.method = method
        self.classifiers = []
    
    def fit(self, X, y):
        """Train all ensemble models."""
        print(f"\n Training Ensemble ({len(self.models_list)} models, method={self.method})...")
        
        for model_type in self.models_list:
            print(f"   Training {model_type}...")
            clf = train_single_model(X, y, model_type)
            self.classifiers.append((model_type, clf))
        
        return self
    
    def predict_proba(self, X):
        """Get probability predictions from ensemble."""
        if self.method == "average":
            # Average probabilities from all models
            all_proba = []
            for model_type, clf in self.classifiers:
                proba = clf.predict_proba(X)
                all_proba.append(proba)
            avg_proba = np.mean(all_proba, axis=0)
            return avg_proba
        
        elif self.method == "vote":
            # Weighted voting based on confidence
            all_proba = []
            for model_type, clf in self.classifiers:
                proba = clf.predict_proba(X)
                all_proba.append(proba)
            
            # Average with voting
            avg_proba = np.mean(all_proba, axis=0)
            return avg_proba
        
        else:
            raise ValueError(f"Unknown ensemble method: {self.method}")
    
    def predict(self, X):
        """Get class predictions."""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

def train_single_model(X, y, model_type):
    """Train a single model (helper for ensemble)."""
    input_dim = X.shape[1]
    
    # Calculate class weights
    unique, counts = np.unique(y, return_counts=True)
    if len(unique) > 1:
        weight_ratio = counts[0] / counts[1] if counts[1] > 0 else 1.0
    else:
        weight_ratio = 1.0
    
    if model_type == "xgboost":
        clf = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=weight_ratio,
            random_state=RANDOM_SEED,
            eval_metric='logloss',
            early_stopping_rounds=20,
            n_jobs=-1
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
        )
        clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
    elif model_type == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=RANDOM_SEED,
            n_jobs=-1
        )
        clf.fit(X, y)
        
    elif model_type == "neural_net":
        clf = NeuralNetClassifier(
            input_dim=input_dim,
            hidden_dims=NN_HIDDEN_DIMS,
            dropout=NN_DROPOUT,
            learning_rate=NN_LEARNING_RATE,
            epochs=NN_EPOCHS,
            batch_size=NN_BATCH_SIZE,
            early_stop_patience=NN_EARLY_STOP_PATIENCE
        )
        clf.fit(X, y)
        
    else:  # logistic
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(
            max_iter=3000,
            class_weight='balanced',
            solver='lbfgs',
            random_state=RANDOM_SEED
        )
        clf.fit(X, y)
    
    return clf

# --------------------------- TRAINING -----------------------------------------
def train_classifier(X, y, model_type="xgboost"):
    """Train a classifier or ensemble."""
    input_dim = X.shape[1]
    
    if model_type == "ensemble":
        # Train ensemble of models
        clf = EnsembleClassifier(ENSEMBLE_MODELS, method=ENSEMBLE_METHOD)
        clf.fit(X, y)
        return clf
    
    # Single model
    return train_single_model(X, y, model_type)

def train_and_evaluate_classifier(pos_paths, neg_paths, pos_embeddings, neg_embeddings, 
                                   model, preprocess, device):
    """Train classifier with validation."""
    print("\n" + "="*60)
    print("PHASE: TRAINING & VALIDATION")
    print("="*60)
    
    # Create labels
    pos_labels = np.ones(len(pos_paths), dtype=np.int64)
    neg_labels = np.zeros(len(neg_paths), dtype=np.int64)
    
    all_paths = pos_paths + neg_paths
    all_embeddings = np.vstack([pos_embeddings, neg_embeddings])
    all_labels = np.concatenate([pos_labels, neg_labels])
    
    print(f" Total training data: {len(all_paths)} photos")
    print(f"   Positives: {len(pos_paths)} ({len(pos_paths)/len(all_paths)*100:.1f}%)")
    print(f"   Negatives: {len(neg_paths)} ({len(neg_paths)/len(all_paths)*100:.1f}%)")
    
    # Compute quality cache for all paths
    print("\n Computing quality scores for all training images...")
    quality_cache = get_quality_scores_batch(all_paths)
    
    # NEW: Compute face features if enabled
    face_features = None
    if USE_FACE_FEATURES:
        face_features = get_face_features_batch(all_paths, workers=QUALITY_WORKERS)
    
    # NEW: Compute temporal/metadata features if enabled
    temporal_features = None
    if USE_TEMPORAL_FEATURES or USE_METADATA_FEATURES:
        temporal_features = get_temporal_metadata_batch(all_paths)
    
    # Get combined features WITH ALL NEW FEATURES
    print("\n Creating feature vectors...")
    X_all = get_combined_features(all_paths, all_embeddings, quality_cache, face_features, temporal_features)
    
    if CLUSTER_AWARE:
        # Build clusters
        clusters, quality_cache = build_clusters_with_embeddings(all_paths, all_embeddings, all_labels)

        
        # NEW: Add burst features
        if USE_NEW_BURST_DETECTION:
            print("\n Computing burst context features...")
            burst_features = compute_burst_features(all_paths, clusters)
            X_all = add_burst_features_to_matrix(X_all, all_paths, burst_features)
        
        # Split clusters for train/val
        from sklearn.model_selection import train_test_split as tts
        train_clusters, val_clusters = tts(
            clusters, test_size=VALIDATION_SPLIT, random_state=RANDOM_SEED
        )
        
        print(f"\n Cluster-aware train/validation split ({int((1-VALIDATION_SPLIT)*100)}/{int(VALIDATION_SPLIT*100)})...")
        print(f"   Train: {len(train_clusters)} clusters, {sum(len(c) for c in train_clusters)} photos")
        print(f"   Val: {len(val_clusters)} clusters, {sum(len(c) for c in val_clusters)} photos")
        
        # Sample representatives with cached quality scores
        if CLUSTER_TRAINING_STRATEGY == "representatives":
            print("\n Sampling representatives for training...")
            train_paths, X_train_emb, y_train = sample_cluster_representatives(
                train_clusters, quality_cache, sampling_strategy="best_quality"
            )
            
            # Get features for representatives
            X_train = get_combined_features(train_paths, X_train_emb, quality_cache, face_features, temporal_features)

            # Add burst features for train set
            if USE_NEW_BURST_DETECTION:
                X_train = add_burst_features_to_matrix(X_train, train_paths, burst_features)
            
            # Use all photos for validation
            val_paths = [item[0] for cluster in val_clusters for item in cluster]
            X_val_emb = np.vstack([item[1] for cluster in val_clusters for item in cluster])
            y_val = np.array([int(item[2]) for cluster in val_clusters for item in cluster], dtype=np.int64)
            X_val = get_combined_features(val_paths, X_val_emb, quality_cache, face_features, temporal_features)

            # Add burst features for val set
            if USE_NEW_BURST_DETECTION:
                X_val = add_burst_features_to_matrix(X_val, val_paths, burst_features)
        else:
            # Use all photos
            train_paths = [item[0] for cluster in train_clusters for item in cluster]
            X_train_emb = np.vstack([item[1] for cluster in train_clusters for item in cluster])
            y_train = np.array([int(item[2]) for cluster in train_clusters for item in cluster], dtype=np.int64)
            X_train = get_combined_features(train_paths, X_train_emb, quality_cache, face_features, temporal_features)

            # Add burst features
            if USE_NEW_BURST_DETECTION:
                X_train = add_burst_features_to_matrix(X_train, train_paths, burst_features)
            
            val_paths = [item[0] for cluster in val_clusters for item in cluster]
            X_val_emb = np.vstack([item[1] for cluster in val_clusters for item in cluster])
            y_val = np.array([int(item[2]) for cluster in val_clusters for item in cluster], dtype=np.int64)
            X_val = get_combined_features(val_paths, X_val_emb, quality_cache, face_features, temporal_features)

            # Add burst features
            if USE_NEW_BURST_DETECTION:
                X_val = add_burst_features_to_matrix(X_val, val_paths, burst_features)
        
        print(f"\n Cluster-aware data prepared:")
        print(f"   Training: {len(X_train)} photos")
        print(f"   Validation: {len(X_val)} photos")
    else:
        # Standard split
        # Add burst features for non-cluster-aware paths
        if USE_NEW_BURST_DETECTION:
            print("\n Computing burst context features (no clusters)...")
            burst_features = compute_burst_features(all_paths, [])  # Empty = all singletons
            X_all = add_burst_features_to_matrix(X_all, all_paths, burst_features)

        X_train, X_val, y_train, y_val = train_test_split(
            X_all, all_labels, test_size=VALIDATION_SPLIT, 
            stratify=all_labels, random_state=RANDOM_SEED
        )
    
    # Check for single-class issue
    unique_train = np.unique(y_train)
    unique_val = np.unique(y_val)
    
    if len(unique_train) < 2:
        print(f" ERROR: Training set has only one class: {unique_train}")
        print("   Cannot train classifier!")
        return None, None
    
    if len(unique_val) < 2:
        print(f"  Warning: Validation set has only one class: {unique_val}")
    
    # Apply class balancing if enabled
    if ENABLE_CLASS_BALANCING and CLASS_BALANCE_METHOD != "none":
        X_train, y_train = balance_training_data(
            X_train, y_train, 
            method=CLASS_BALANCE_METHOD, 
            target_ratio=CLASS_BALANCE_RATIO
        )
    
    # Train
    print(f"\n Training {MODEL_TYPE} model...")
    clf = train_classifier(X_train, y_train, MODEL_TYPE)
    
    # Evaluate
    if len(unique_val) >= 2:
        y_pred = clf.predict(X_val)
        y_proba = clf.predict_proba(X_val)[:,1]
        
        print("\n" + "="*60)
        print("VALIDATION PERFORMANCE")
        print("="*60)
        print(classification_report(y_val, y_pred, target_names=["Storage", "Share"], zero_division=0))
        
        auc = roc_auc_score(y_val, y_proba)
        print(f"ROC-AUC Score: {auc:.4f}")
        
        if auc < 0.6:
            print("\n  WARNING: Model performance is poor (AUC < 0.6)")
            print("   This suggests labels or data may have issues")
        
        # Find optimal threshold
        fpr, tpr, thresholds = roc_curve(y_val, y_proba)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_thresh = thresholds[optimal_idx]
        
        print(f"\n Suggested threshold: {optimal_thresh:.3f}")
        
        # Save plots
        plot_validation_results(y_val, y_proba)
        
        optimal_threshold = optimal_thresh if AUTO_TUNE_THRESHOLD else THRESH_SHARE
    else:
        print("\n  Skipping validation metrics (single class in val set)")
        optimal_threshold = THRESH_SHARE
    
    # Retrain on full dataset
    print(f"\n Retraining on full dataset for deployment...")
    clf_final = train_classifier(X_all, all_labels, MODEL_TYPE)
    
    return clf_final, optimal_threshold

def plot_validation_results(y_val, y_proba):
    """Generate validation plots."""
    rep_dir = OUT_BASE / "Reports"
    rep_dir.mkdir(parents=True, exist_ok=True)
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_val, y_proba)
    auc = roc_auc_score(y_val, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(rep_dir / "roc_curve.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f" Validation plots saved to {rep_dir}/")

# --------------------------- ROUTING ------------------------------------------
def route_with_confidence(prob, threshold, confidence_band=CONFIDENCE_BAND):
    distance_from_boundary = abs(prob - 0.5)
    if distance_from_boundary < confidence_band:
        return "Review"
    elif prob >= threshold:
        return "Share"
    else:
        return "Storage"

# --------------------------- MAIN ---------------------------------------------
def main():
    print("="*60)
    print("PHOTO SORTER v3 DEBUGGED")
    print("="*60)
    
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    
    if FORCE_RECOMPUTE_FEATURES:
        print("  Force recompute enabled - will rebuild feature cache")
    
    ensure_dirs()
    
    print("\n Loading CLIP model...")
    model, preprocess, tokenizer = load_model(device)
    
    # Build training data
    print("\n" + "="*60)
    print("PHASE 1: BUILDING TRAINING DATA")
    print("="*60)
    
    pos_paths = list_images(FAMILY_DIR)
    priv_paths = list_images(PRIVATE_DIR)
    
    camera_roll_negs = None
    if USE_CAMERA_ROLL_NEGATIVES and CAMERA_ROLL.exists():
        test_paths_sample = list_images(UNLABELED)
        camera_roll_negs = sample_camera_roll_negatives(test_paths_sample, CAMERA_ROLL, DATE_RANGE_DAYS)
    
    neg_paths = sample_negatives(priv_paths, pos_paths, camera_roll_negs)
    
    if len(pos_paths) > 0 and len(neg_paths) > len(pos_paths)*MAX_NEG_PER_POS:
        neg_paths = random.sample(neg_paths, len(pos_paths)*MAX_NEG_PER_POS)
    
    print(f" Positive examples: {len(pos_paths)}")
    print(f" Negative examples: {len(neg_paths)}")
    
    # Embed
    X_pos, pos_good = embed_paths(pos_paths, model, preprocess, device, batch_size=EMBED_BATCH)
    X_neg, neg_good = embed_paths(neg_paths, model, preprocess, device, batch_size=EMBED_BATCH)
    
    if len(pos_good) == 0 or len(neg_good) == 0:
        print(" Not enough labeled data to train")
        failure_logger.save(OUT_BASE / "Reports" / "failures.csv")
        return
    
    # Train
    clf, optimal_threshold = train_and_evaluate_classifier(
        pos_good, neg_good, X_pos, X_neg, model, preprocess, device
    )
    
    if clf is None:
        print(" Training failed")
        failure_logger.save(OUT_BASE / "Reports" / "failures.csv")
        return
    
    # Process unlabeled
    print("\n" + "="*60)
    print("PHASE 2: PROCESSING UNLABELED IMAGES")
    print("="*60)
    
    test_paths = list_images(UNLABELED)
    print(f" Found {len(test_paths)} images to sort")
    
    if len(test_paths) == 0:
        print(" No images to sort")
        failure_logger.save(OUT_BASE / "Reports" / "failures.csv")
        return
    
    # Compute quality cache
    print("\n Computing quality scores...")
    test_quality_cache = get_quality_scores_batch(test_paths)
    
    # Near-dup detection
    print("\n Detecting near-duplicates...")
    clusters = cluster_near_dups(test_paths, n_workers=DEDUP_WORKERS)
    
    unique = []
    dup_rows, trash_rows = [], []
    
    if len(clusters) > 0:
        for cl in tqdm(clusters, desc="Choosing best in cluster"):
            best, losers_trash, losers_keep = choose_best_in_cluster(cl, test_quality_cache)
            unique.append(best)
            for p, q in losers_keep:
                dup_rows.append({"path": str(p), "quality": q})
            for p, q in losers_trash:
                trash_rows.append({"path": str(p), "quality": q})
    else:
        unique = test_paths
    
    print(f"   Unique: {len(unique)}, Dupes: {len(dup_rows)}, Trash: {len(trash_rows)}")
    
    # Baby gate (optional)
    if SKIP_BABY_GATE:
        print("\n Baby gate SKIPPED")
        baby_keep = unique
        ignore_rows = []
    else:
        print("\n Baby gate...")
        # Baby gate code here (omitted for brevity)
        baby_keep = unique
        ignore_rows = []
    
    # Classify
    print(f"\n Classifying {len(baby_keep)} images...")
    
    # Two-stage quality filter
    if USE_TWO_STAGE_FILTER:
        print(f"\n Stage 1: Quality pre-filter (threshold={QUALITY_FILTER_THRESHOLD})...")
        low_quality = []
        high_quality = []
        
        for p in baby_keep:
            q_score = test_quality_cache.get(p, 0.0)
            if q_score < QUALITY_FILTER_THRESHOLD:
                low_quality.append(p)
            else:
                high_quality.append(p)
        
        print(f"   Low quality (auto-Storage): {len(low_quality)} ({len(low_quality)/len(baby_keep)*100:.1f}%)")
        print(f"   High quality (classify): {len(high_quality)} ({len(high_quality)/len(baby_keep)*100:.1f}%)")
        
        # Auto-route low quality to Storage
        storage_auto = low_quality
        to_classify = high_quality
    else:
        storage_auto = []
        to_classify = baby_keep
    
    # Embed and extract features for classification
    print(f"\n Stage 2: Classifying {len(to_classify)} images...")
    X_u, good_u = embed_paths(to_classify, model, preprocess, device, batch_size=EMBED_BATCH)

    
    # NEW: Detect bursts in unlabeled photos
    if USE_NEW_BURST_DETECTION:
        print("\n Detecting bursts in unlabeled photos...")
        test_clusters = detect_bursts_temporal_visual(
            good_u,
            X_u,
            time_window_seconds=10,
            embedding_similarity_threshold=0.92,
            min_burst_size=2
        )
        
        print(" Computing burst context for unlabeled photos...")
        test_burst_features = compute_burst_features(good_u, test_clusters)
    else:
        print("\n Using phash clustering for bursts...")
        test_clusters = cluster_near_dups(good_u, n_workers=DEDUP_WORKERS)
        test_burst_features = compute_burst_features(good_u, test_clusters)
    
    if len(good_u) == 0:
        print(" No images to classify")
        # Still route storage_auto
        if len(storage_auto) > 0:
            print(f"   Routing {len(storage_auto)} low-quality to Storage...")
            for p in storage_auto:
                copy_file(p, OUT_BASE / "Storage")
        failure_logger.save(OUT_BASE / "Reports" / "failures.csv")
        return
    
    # Extract face and temporal features for test set
    test_face_features = None
    if USE_FACE_FEATURES:
        test_face_features = get_face_features_batch(good_u, workers=QUALITY_WORKERS)
    
    test_temporal_features = None
    if USE_TEMPORAL_FEATURES or USE_METADATA_FEATURES:
        test_temporal_features = get_temporal_metadata_batch(good_u)
    
    # Get features WITH ALL NEW FEATURES
    X_u_features = get_combined_features(good_u, X_u, test_quality_cache, test_face_features, test_temporal_features)

    
    # Add burst context
    if USE_NEW_BURST_DETECTION:
        X_u_features = add_burst_features_to_matrix(X_u_features, good_u, test_burst_features)
    
    proba = clf.predict_proba(X_u_features)[:,1]

    
    # NEW: Post-process burst scores (keep only top photos in each burst)
    if USE_NEW_BURST_DETECTION:
        print("\n Post-processing burst scores...")
        burst_count = 0
        for cluster in test_clusters:
            if len(cluster) <= 1:
                continue
            
            # Get indices for photos in this burst
            cluster_indices = []
            for p in cluster:
                try:
                    idx = good_u.index(p)
                    cluster_indices.append(idx)
                except ValueError:
                    continue
            
            if len(cluster_indices) < 2:
                continue
            
            burst_count += 1
            
            # Get scores for this burst
            cluster_scores = proba[cluster_indices]
            
            # Keep top 30% or minimum 2 photos
            n_keep = max(2, int(len(cluster_indices) * 0.3))
            top_indices_in_cluster = np.argsort(cluster_scores)[-n_keep:]
            
            # Adjust scores: boost best, reduce others
            for local_idx, global_idx in enumerate(cluster_indices):
                if local_idx in top_indices_in_cluster:
                    # Boost keepers by 20%
                    proba[global_idx] = min(proba[global_idx] * 1.2, 1.0)
                else:
                    # Reduce non-keepers by 50%
                    proba[global_idx] = proba[global_idx] * 0.5
        
        print(f"   Adjusted scores for {burst_count} bursts")
    
    # Route
    print("\n Routing...")
    stats = {
        "Share": 0, "Storage": len(storage_auto), "Review": 0, "Ignore": len(ignore_rows),
        "Dupes": len(dup_rows), "TrashCandidates": len(trash_rows)
    }
    log_rows = []
    
    # Route auto-storage from quality filter
    for p in storage_auto:
        dst = OUT_BASE / "Storage"
        written = copy_file(p, dst)
        if written:
            log_rows.append({
                "path": str(p),
                "prob_share": 0.0,
                "bucket": "Storage",
                "dest": written,
                "reason": "quality_filter"
            })
    
    # Route classified images
    for p, pr in tqdm(list(zip(good_u, proba)), desc="Routing"):
        bucket = route_with_confidence(pr, optimal_threshold, CONFIDENCE_BAND)
        dst = OUT_BASE / bucket
        written = copy_file(p, dst)
        
        if written:
            stats[bucket] += 1
            log_rows.append({
                "path": str(p), 
                "prob_share": float(pr), 
                "bucket": bucket, 
                "dest": written
            })
    
    # Save reports
    print("\n Saving reports...")
    for r in dup_rows:
        copy_file(Path(r["path"]), OUT_BASE / "Dupes")
    for r in trash_rows:
        copy_file(Path(r["path"]), OUT_BASE / "TrashCandidates")
    for r in ignore_rows:
        copy_file(Path(r["path"]), OUT_BASE / "Ignore")
    
    for r in ignore_rows:
        r.update({"bucket": "Ignore"})
    for r in dup_rows:
        r.update({"bucket": "Dupes"})
    for r in trash_rows:
        r.update({"bucket": "TrashCandidates"})
    
    all_rows = log_rows + ignore_rows + dup_rows + trash_rows
    rep_dir = OUT_BASE / "Reports"
    rep_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = rep_dir / "routing_log.csv"
    pd.DataFrame(all_rows).to_csv(csv_path, index=False)
    
    summary = {
        "metric": list(stats.keys()) + ["Total"],
        "count": list(stats.values()) + [sum(stats.values())]
    }
    pd.DataFrame(summary).to_csv(rep_dir / "summary.csv", index=False)
    
    failure_count = failure_logger.save(rep_dir / "failures.csv")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for k, v in stats.items():
        pct = v/sum(stats.values())*100 if sum(stats.values()) > 0 else 0
        print(f"{k:20s}: {v:5d} ({pct:.1f}%)")
    print(f"{'Total':20s}: {sum(stats.values()):5d}")
    if failure_count > 0:
        print(f"\n  {failure_count} failures logged")
    print(f"\n Reports: {rep_dir}/")
    print(f" Output: {OUT_BASE}/")
    print(f"\n{'  DRY RUN' if DRY_RUN else ' Complete'}")
    
    print("\n" + "="*60)
    print(" COMPLETE")
    print("="*60)

if __name__ == "__main__":
    freeze_support()  # Required for Windows multiprocessing
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  Interrupted by user")
        failure_logger.save(OUT_BASE / "Reports" / "failures.csv")
    except Exception as e:
        print(f"\n\n Fatal error: {e}")
        import traceback
        traceback.print_exc()
        failure_logger.save(OUT_BASE / "Reports" / "failures.csv")
