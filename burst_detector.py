"""
Burst detection using temporal proximity + visual similarity
"""
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import imagehash
from PIL import Image

def get_photo_timestamp(photo_path):
    """Extract photo timestamp from EXIF or file metadata."""
    try:
        from taste_sort_win import get_image_date
        date = get_image_date(photo_path)
        if date:
            return date
    except:
        pass
    
    try:
        return datetime.fromtimestamp(photo_path.stat().st_mtime)
    except:
        return None

def detect_bursts_temporal_visual(photos, embeddings, 
                                   time_window_seconds=10,
                                   embedding_similarity_threshold=0.92,
                                   min_burst_size=2,
                                   max_burst_size=50):
    """
    Detect photo bursts using temporal proximity + visual similarity.
    
    Args:
        photos: List of photo paths
        embeddings: numpy array of CLIP embeddings (normalized)
        time_window_seconds: Max seconds between photos in a burst
        embedding_similarity_threshold: Min cosine similarity for same burst (0-1)
        min_burst_size: Minimum photos to form a burst
        max_burst_size: Maximum photos in a burst (split if larger)
    
    Returns:
        List of bursts (each burst is list of photo paths)
    """
    print(f"\n[BURST] Detecting bursts (temporal + visual)...")
    print(f"[BURST] Time window: {time_window_seconds}s")
    print(f"[BURST] Similarity threshold: {embedding_similarity_threshold}")
    
    # Get timestamps for all photos
    photo_times = []
    valid_indices = []
    
    for i, photo in enumerate(photos):
        timestamp = get_photo_timestamp(photo)
        if timestamp:
            photo_times.append((i, photo, timestamp))
            valid_indices.append(i)
    
    if not photo_times:
        print("[BURST] No timestamps found, using visual-only clustering")
        return detect_bursts_visual_only(photos, embeddings, 
                                         embedding_similarity_threshold, 
                                         min_burst_size)
    
    # Sort by timestamp
    photo_times.sort(key=lambda x: x[2])
    
    print(f"[BURST] Extracted timestamps from {len(photo_times)}/{len(photos)} photos")
    
    # Create temporal groups (photos within time_window)
    temporal_groups = []
    current_group = [photo_times[0]]
    
    for i in range(1, len(photo_times)):
        prev_time = current_group[-1][2]
        curr_time = photo_times[i][2]
        
        time_diff = (curr_time - prev_time).total_seconds()
        
        if time_diff <= time_window_seconds:
            current_group.append(photo_times[i])
        else:
            if len(current_group) >= min_burst_size:
                temporal_groups.append(current_group)
            current_group = [photo_times[i]]
    
    # Don't forget last group
    if len(current_group) >= min_burst_size:
        temporal_groups.append(current_group)
    
    print(f"[BURST] Found {len(temporal_groups)} temporal groups")
    
    # Within each temporal group, cluster by visual similarity
    bursts = []
    singletons = []
    
    for group in temporal_groups:
        if len(group) < min_burst_size:
            singletons.extend([item[1] for item in group])
            continue
        
        # Get embeddings for this group
        group_indices = [item[0] for item in group]
        group_photos = [item[1] for item in group]
        group_embeddings = embeddings[group_indices]
        
        # Compute pairwise similarity
        similarity_matrix = cosine_similarity(group_embeddings)
        
        # Cluster using similarity threshold
        visited = set()
        group_bursts = []
        
        for i in range(len(group_photos)):
            if i in visited:
                continue
            
            # Start new cluster
            cluster = [i]
            visited.add(i)
            
            # Find all photos similar to this one
            for j in range(i + 1, len(group_photos)):
                if j in visited:
                    continue
                
                # Check if j is similar to any photo in current cluster
                max_sim = max(similarity_matrix[i, j] for i in cluster)
                
                if max_sim >= embedding_similarity_threshold:
                    cluster.append(j)
                    visited.add(j)
            
            if len(cluster) >= min_burst_size:
                burst_photos = [group_photos[idx] for idx in cluster]
                
                # Split if too large
                if len(burst_photos) > max_burst_size:
                    for chunk_start in range(0, len(burst_photos), max_burst_size):
                        chunk = burst_photos[chunk_start:chunk_start + max_burst_size]
                        if len(chunk) >= min_burst_size:
                            group_bursts.append(chunk)
                else:
                    group_bursts.append(burst_photos)
            else:
                singletons.extend([group_photos[idx] for idx in cluster])
        
        bursts.extend(group_bursts)
    
    # Add photos without timestamps as singletons
    no_timestamp_photos = [p for i, p in enumerate(photos) if i not in valid_indices]
    singletons.extend(no_timestamp_photos)
    
    # Stats
    print(f"\n[BURST] Detection complete:")
    print(f"[BURST]   Bursts: {len(bursts)}")
    if bursts:
        sizes = [len(b) for b in bursts]
        print(f"[BURST]   Burst sizes: min={min(sizes)}, max={max(sizes)}, avg={np.mean(sizes):.1f}")
        print(f"[BURST]   Total photos in bursts: {sum(sizes)}")
    print(f"[BURST]   Singletons: {len(singletons)}")
    
    # Return bursts + singletons as individual "bursts" of size 1
    return bursts + [[p] for p in singletons]

def detect_bursts_visual_only(photos, embeddings, 
                               similarity_threshold=0.92,
                               min_burst_size=2):
    """
    Fallback: detect bursts using only visual similarity (no timestamps).
    """
    print("[BURST] Using visual-only clustering...")
    
    similarity_matrix = cosine_similarity(embeddings)
    
    visited = set()
    bursts = []
    
    for i in range(len(photos)):
        if i in visited:
            continue
        
        cluster = [i]
        visited.add(i)
        
        for j in range(i + 1, len(photos)):
            if j in visited:
                continue
            
            max_sim = max(similarity_matrix[k, j] for k in cluster)
            
            if max_sim >= similarity_threshold:
                cluster.append(j)
                visited.add(j)
        
        if len(cluster) >= min_burst_size:
            bursts.append([photos[idx] for idx in cluster])
        else:
            bursts.append([photos[i]])
    
    return bursts

def detect_exact_duplicates(photos, hamming_threshold=5):
    """
    Detect exact/near-exact duplicates using perceptual hashing.
    """
    print(f"\n[DUPES] Detecting exact duplicates (phash, hamming<={hamming_threshold})...")
    
    def compute_phash(photo):
        try:
            with Image.open(photo) as img:
                img = img.convert("RGB")
                return imagehash.phash(img, hash_size=16)
        except:
            return None
    
    # Compute hashes
    hashes = {}
    for photo in photos:
        h = compute_phash(photo)
        if h:
            hashes[photo] = h
    
    # Find duplicates
    duplicates = defaultdict(list)
    checked = set()
    
    for photo1, hash1 in hashes.items():
        if photo1 in checked:
            continue
        
        for photo2, hash2 in hashes.items():
            if photo1 == photo2 or photo2 in checked:
                continue
            
            if hash1 - hash2 <= hamming_threshold:
                duplicates[photo1].append(photo2)
                checked.add(photo2)
    
    print(f"[DUPES] Found {len(duplicates)} duplicate groups")
    return duplicates
