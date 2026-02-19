#!/usr/bin/env python3
"""
Learn from manual review decisions.

This script scans the output folders (Share, Review, Storage) and compares
against the classification log to find photos that were manually moved.
It then adds these corrections as new training examples.

Usage:
    python learn_from_reviews.py

The script will:
1. Load the latest classification log
2. Scan actual folder locations
3. Find discrepancies (manual moves)
4. Add corrections to training labels
5. Optionally regenerate taste profile
"""

import json
import pandas as pd
from pathlib import Path
from collections import defaultdict
import sys

# Import from existing code
sys.path.insert(0, '..')
sys.path.insert(0, '..')

from taste_sort_win import OUT_BASE, CACHE_ROOT

LABELING_CACHE = CACHE_ROOT / "labeling_pairwise"
LABELS_FILE = LABELING_CACHE / "pairwise_labels.json"
CLASSIFICATION_LOG = OUT_BASE / "Reports" / "gemini_v4_grouped_log.csv"


def load_classification_log():
    """Load the latest classification log."""
    if not CLASSIFICATION_LOG.exists():
        print(f"❌ Classification log not found: {CLASSIFICATION_LOG}")
        return None

    df = pd.read_csv(CLASSIFICATION_LOG)
    print(f"✅ Loaded classification log: {len(df)} photos")
    return df


def scan_actual_locations():
    """Scan actual folder locations to see where photos ended up."""
    actual_locations = {}

    folders = {
        "Share": OUT_BASE / "Share",
        "Review": OUT_BASE / "Review",
        "Storage": OUT_BASE / "Storage"
    }

    for bucket, folder in folders.items():
        if not folder.exists():
            continue

        for photo_path in folder.glob("*"):
            if photo_path.is_file() and photo_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.heic']:
                actual_locations[photo_path.name] = bucket

    print(f"✅ Scanned folders: {len(actual_locations)} photos found")
    return actual_locations


def find_manual_moves(classification_log_df, actual_locations):
    """Find photos that were manually moved to different folders."""
    manual_moves = []

    for _, row in classification_log_df.iterrows():
        dest_path = Path(row["destination"])
        photo_name = dest_path.name
        original_bucket = row["bucket"]

        # Check if photo was moved
        actual_bucket = actual_locations.get(photo_name)

        if actual_bucket and actual_bucket != original_bucket:
            manual_moves.append({
                "photo_name": photo_name,
                "photo_path": str(dest_path),
                "original_bucket": original_bucket,
                "actual_bucket": actual_bucket,
                "original_classification": row["classification"],
                "original_confidence": row["confidence"],
                "original_reasoning": row.get("reasoning", "")
            })

    print(f"✅ Found {len(manual_moves)} manual moves")
    return manual_moves


def add_corrections_to_training(manual_moves, labels_file):
    """Add manual corrections to training labels."""
    if not labels_file.exists():
        print(f"⚠️  Labels file not found: {labels_file}")
        return 0

    with open(labels_file, 'r') as f:
        labels = json.load(f)

    if "single" not in labels:
        labels["single"] = {}

    # Track new additions
    new_count = 0
    updated_count = 0

    for move in manual_moves:
        photo_path = move["photo_path"]

        # Convert bucket to action
        action_map = {
            "Share": "Share",
            "Storage": "Storage",
            "Review": "Share"  # Moving to Review suggests it might be shareable
        }

        action = action_map.get(move["actual_bucket"], "Storage")

        # Determine score based on the move
        if move["actual_bucket"] == "Share" and move["original_bucket"] == "Storage":
            # Strong correction: Storage -> Share
            confidence = 5
            reason = f"Manually moved from Storage to Share (was {move['original_classification']})"
        elif move["actual_bucket"] == "Storage" and move["original_bucket"] == "Share":
            # Strong correction: Share -> Storage
            confidence = 5
            reason = f"Manually moved from Share to Storage (was {move['original_classification']})"
        elif move["actual_bucket"] == "Share" and move["original_bucket"] == "Review":
            # Review -> Share (confirming it's shareable)
            confidence = 4
            reason = f"Manually promoted from Review to Share"
        elif move["actual_bucket"] == "Storage" and move["original_bucket"] == "Review":
            # Review -> Storage (confirming it's not shareable)
            confidence = 4
            reason = f"Manually demoted from Review to Storage"
        else:
            # Other moves
            confidence = 4
            reason = f"Manually moved from {move['original_bucket']} to {move['actual_bucket']}"

        if photo_path in labels["single"]:
            updated_count += 1
        else:
            new_count += 1

        labels["single"][photo_path] = {
            "action": action,
            "confidence": confidence,
            "reason": reason,
            "source": "manual_review_correction",
            "original_bucket": move["original_bucket"],
            "corrected_bucket": move["actual_bucket"]
        }

    # Save updated labels
    with open(labels_file, 'w') as f:
        json.dump(labels, f, indent=2)

    print(f"✅ Added {new_count} new corrections, updated {updated_count} existing labels")
    print(f"   Labels saved to: {labels_file}")

    return new_count + updated_count


def generate_summary_report(manual_moves):
    """Generate a summary report of corrections."""
    print("\n" + "="*60)
    print("MANUAL REVIEW CORRECTIONS SUMMARY")
    print("="*60)

    if not manual_moves:
        print("\n✅ No manual corrections found - all photos are in their original folders")
        return

    # Group by correction type
    corrections = defaultdict(list)
    for move in manual_moves:
        key = f"{move['original_bucket']} → {move['actual_bucket']}"
        corrections[key].append(move)

    for correction_type, moves in sorted(corrections.items()):
        print(f"\n{correction_type}: {len(moves)} photos")
        for move in moves[:5]:  # Show first 5 examples
            print(f"   - {move['photo_name']}")
        if len(moves) > 5:
            print(f"   ... and {len(moves) - 5} more")

    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)

    # Provide insights
    share_to_storage = len(corrections.get("Share → Storage", []))
    storage_to_share = len(corrections.get("Storage → Share", []))
    review_to_share = len(corrections.get("Review → Share", []))
    review_to_storage = len(corrections.get("Review → Storage", []))

    if share_to_storage > storage_to_share:
        print("\n⚠️  More photos moved Share → Storage than Storage → Share")
        print("   Suggestion: Classifier may be too generous. Consider:")
        print("   - Raising SHARE_THRESHOLD (currently 0.60)")
        print("   - Reviewing taste profile for overly broad criteria")

    if storage_to_share > share_to_storage:
        print("\n⚠️  More photos moved Storage → Share than Share → Storage")
        print("   Suggestion: Classifier may be too strict. Consider:")
        print("   - Lowering SHARE_THRESHOLD (currently 0.60)")
        print("   - Reviewing taste profile for missing criteria")

    if review_to_share + review_to_storage > 0:
        print(f"\n✅ Review folder is working: {review_to_share + review_to_storage} decisions made")
        print(f"   - Promoted to Share: {review_to_share}")
        print(f"   - Demoted to Storage: {review_to_storage}")

    print("\n" + "="*60)


def main():
    print("="*60)
    print("LEARNING FROM MANUAL REVIEW DECISIONS")
    print("="*60)

    # Step 1: Load classification log
    print("\n📂 Loading classification log...")
    df = load_classification_log()
    if df is None:
        return

    # Step 2: Scan actual folder locations
    print("\n🔍 Scanning actual photo locations...")
    actual_locations = scan_actual_locations()

    # Step 3: Find manual moves
    print("\n🔄 Comparing original vs. actual locations...")
    manual_moves = find_manual_moves(df, actual_locations)

    # Step 4: Generate report
    generate_summary_report(manual_moves)

    if not manual_moves:
        print("\n✅ No corrections to learn from")
        return

    # Step 5: Ask user if they want to add corrections
    print("\n" + "="*60)
    print("Would you like to add these corrections to training data?")
    print("="*60)
    print(f"This will add {len(manual_moves)} new training examples to:")
    print(f"   {LABELS_FILE}")
    print("\nThese corrections will be used in future classifications.")
    print("You can regenerate the taste profile after this.")

    response = input("\nAdd corrections to training? (y/n): ").strip().lower()

    if response == 'y':
        count = add_corrections_to_training(manual_moves, LABELS_FILE)
        print(f"\n✅ Added {count} corrections to training data")

        print("\n📝 Next steps:")
        print("   1. Run: python generate_taste_profile.py")
        print("      (This will regenerate taste profile with new corrections)")
        print("   2. Run: python taste_classify_gemini_v4.py")
        print("      (This will classify new photos with updated preferences)")
    else:
        print("\n⏭️  Skipped adding corrections")

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
