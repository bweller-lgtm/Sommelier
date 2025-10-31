# burst_features.py
# Add burst context features to help model learn context-dependent judgments

import numpy as np
from pathlib import Path
from collections import defaultdict

def compute_burst_features(photo_paths, burst_clusters):
    """
    Compute burst context features for each photo.
    
    Args:
        photo_paths: List of photo paths
        burst_clusters: List of bursts (each burst is list of photo paths)
    
    Returns:
        dict mapping photo_path -> burst_features dict
    """
    # Build photo -> burst mapping
    photo_to_burst = {}
    burst_sizes = {}
    burst_positions = {}
    
    for burst_id, burst in enumerate(burst_clusters):
        for position, photo in enumerate(burst):
            photo_str = str(photo)
            photo_to_burst[photo_str] = burst_id
            burst_sizes[photo_str] = len(burst)
            burst_positions[photo_str] = position
    
    # Compute features for each photo
    features = {}
    for photo in photo_paths:
        photo_str = str(photo)
        
        if photo_str in photo_to_burst:
            # Part of a burst
            features[photo] = {
                "is_burst_member": 1,
                "burst_size": burst_sizes[photo_str],
                "burst_size_normalized": min(burst_sizes[photo_str] / 20.0, 1.0),  # normalize to 0-1
                "burst_position_normalized": burst_positions[photo_str] / max(burst_sizes[photo_str] - 1, 1),  # 0 to 1
                "is_large_burst": 1 if burst_sizes[photo_str] >= 5 else 0,
                "is_small_burst": 1 if 2 <= burst_sizes[photo_str] < 5 else 0,
            }
        else:
            # Singleton
            features[photo] = {
                "is_burst_member": 0,
                "burst_size": 0,
                "burst_size_normalized": 0.0,
                "burst_position_normalized": 0.0,
                "is_large_burst": 0,
                "is_small_burst": 0,
            }
    
    return features


def add_burst_features_to_matrix(X, photo_paths, burst_features):
    """
    Add burst features to existing feature matrix.
    
    Args:
        X: Existing feature matrix (N x D)
        photo_paths: List of photo paths corresponding to X rows
        burst_features: Dict from compute_burst_features
    
    Returns:
        Enhanced feature matrix (N x D+6)
    """
    burst_feature_matrix = []
    
    for photo in photo_paths:
        features = burst_features.get(photo, {
            "is_burst_member": 0,
            "burst_size": 0,
            "burst_size_normalized": 0.0,
            "burst_position_normalized": 0.0,
            "is_large_burst": 0,
            "is_small_burst": 0,
        })
        
        burst_feature_matrix.append([
            features["is_burst_member"],
            features["burst_size"],
            features["burst_size_normalized"],
            features["burst_position_normalized"],
            features["is_large_burst"],
            features["is_small_burst"],
        ])
    
    burst_feature_matrix = np.array(burst_feature_matrix)
    
    # Normalize burst_size (raw count)
    if burst_feature_matrix[:, 1].max() > 0:
        burst_feature_matrix[:, 1] = burst_feature_matrix[:, 1] / burst_feature_matrix[:, 1].max()
    
    # Concatenate with existing features
    X_enhanced = np.hstack([X, burst_feature_matrix])
    
    print(f"[BURST] Added 6 burst context features")
    print(f"   Original: {X.shape[1]} features")
    print(f"   Enhanced: {X_enhanced.shape[1]} features")
    
    return X_enhanced


# Example usage in taste_sort_win.py:
"""
# After computing combined features, add burst context:
from burst_features import compute_burst_features, add_burst_features_to_matrix

# During training
burst_features = compute_burst_features(all_paths, clusters)
X_all_enhanced = add_burst_features_to_matrix(X_all, all_paths, burst_features)

# Train with enhanced features
clf = train_classifier(X_all_enhanced, all_labels, MODEL_TYPE)

# During inference
test_burst_features = compute_burst_features(test_paths, test_clusters)
X_test_enhanced = add_burst_features_to_matrix(X_test, test_paths, test_burst_features)
proba = clf.predict_proba(X_test_enhanced)[:, 1]
"""
