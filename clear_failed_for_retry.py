#!/usr/bin/env python3
"""
Clear Failed Classifications for Retry

Identifies photos that failed classification (error fallbacks) in a previous run,
clears their cache entries, and removes them from the Review folder so they can
be re-classified on the next run.

For burst photos with errors, clears the entire burst so it can be re-evaluated together.

Usage:
    python clear_failed_for_retry.py "E:\\Photos\\folder_sorted"
    python clear_failed_for_retry.py "E:\\Photos\\folder_sorted" --dry-run
    python clear_failed_for_retry.py "E:\\Photos\\folder_sorted" --report classification_report_XXXX.csv
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict
import pandas as pd

from sommelier.core.cache import CacheManager, CacheKey


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Clear failed classifications for retry"
    )
    parser.add_argument(
        "output_folder",
        type=str,
        help="Sorted output folder (contains Reports/, Review/, etc.)"
    )
    parser.add_argument(
        "--report",
        type=str,
        help="Specific report CSV (default: most recent)"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=".taste_cache",
        help="Cache directory (default: .taste_cache)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually doing it"
    )
    return parser.parse_args()


def find_latest_report(reports_dir: Path) -> Path:
    """Find the most recent classification report."""
    reports = list(reports_dir.glob("classification_report_*.csv"))
    if not reports:
        raise FileNotFoundError(f"No classification reports found in {reports_dir}")
    reports.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return reports[0]


def identify_error_photos(df: pd.DataFrame) -> pd.DataFrame:
    """Identify photos with error fallback responses."""
    # Check for new-style error tracking first
    if "is_error_fallback" in df.columns:
        errors = df[df["is_error_fallback"] == True]
        print(f"   Using is_error_fallback column")
    else:
        # Fall back to reasoning pattern matching for old reports
        errors = df[df["reasoning"].str.contains("Fallback response:", na=False)]
        print(f"   Using reasoning pattern matching (old report format)")

    return errors


def reconstruct_burst_groupings(df: pd.DataFrame) -> dict:
    """
    Reconstruct burst groupings from the report.

    Returns dict mapping burst_key -> list of photo paths in that burst
    """
    bursts = {}
    current_burst = []

    for idx, row in df.iterrows():
        # Handle potential NaN values
        burst_size = row.get("burst_size", 1)
        burst_index = row.get("burst_index", -1)
        if pd.isna(burst_size):
            burst_size = 1
        if pd.isna(burst_index):
            burst_index = -1

        source = row["source"]

        if burst_size > 1 and burst_index >= 0:
            # Part of a burst
            if burst_index == 0:
                # Start of new burst
                if current_burst:
                    # Save previous burst
                    burst_key = CacheKey.from_files([Path(p) for p in current_burst])
                    bursts[burst_key] = current_burst
                current_burst = [source]
            else:
                # Continuation of burst
                current_burst.append(source)
        else:
            # Singleton - save any pending burst first
            if current_burst:
                burst_key = CacheKey.from_files([Path(p) for p in current_burst])
                bursts[burst_key] = current_burst
                current_burst = []

    # Don't forget last burst
    if current_burst:
        burst_key = CacheKey.from_files([Path(p) for p in current_burst])
        bursts[burst_key] = current_burst

    return bursts


def main():
    args = parse_arguments()

    output_folder = Path(args.output_folder)
    if not output_folder.exists():
        print(f"[ERROR] Error: Output folder not found: {output_folder}")
        sys.exit(1)

    reports_dir = output_folder / "Reports"

    if not reports_dir.exists():
        print(f"[ERROR] Error: Reports folder not found: {reports_dir}")
        sys.exit(1)

    # Find report
    if args.report:
        report_path = reports_dir / args.report
        if not report_path.exists():
            print(f"[ERROR] Error: Report not found: {report_path}")
            sys.exit(1)
    else:
        report_path = find_latest_report(reports_dir)

    print(f"[Report] Using: {report_path.name}")

    # Load report
    df = pd.read_csv(report_path)
    print(f"   Total photos in report: {len(df)}")

    # Identify error photos
    errors = identify_error_photos(df)
    print(f"   Error fallbacks found: {len(errors)}")

    if len(errors) == 0:
        print("\n[OK] No error fallbacks found!")
        sys.exit(0)

    # Separate singletons from burst photos (handle NaN values)
    errors_burst_size = errors["burst_size"].fillna(1)
    singleton_errors = errors[errors_burst_size <= 1]
    burst_errors = errors[errors_burst_size > 1]

    print(f"   Singleton errors: {len(singleton_errors)}")
    print(f"   Burst photo errors: {len(burst_errors)}")

    # Reconstruct burst groupings to find ALL photos in bursts with errors
    burst_groupings = reconstruct_burst_groupings(df)

    # Find which bursts have errors
    bursts_with_errors = set()
    for _, row in burst_errors.iterrows():
        source = row["source"]
        for burst_key, photos in burst_groupings.items():
            if source in photos:
                bursts_with_errors.add(burst_key)
                break

    print(f"   Bursts with errors: {len(bursts_with_errors)}")

    # Collect all photos to clear
    photos_to_clear = set()
    burst_cache_keys_to_clear = set()

    # Add singleton errors
    for _, row in singleton_errors.iterrows():
        photos_to_clear.add(row["source"])

    # Add ALL photos from bursts with errors (not just the failed ones)
    for burst_key in bursts_with_errors:
        burst_cache_keys_to_clear.add(burst_key)
        for photo in burst_groupings[burst_key]:
            photos_to_clear.add(photo)

    # CRITICAL: Verify source files still exist
    # Cache keys include file size+mtime, so if source files are missing/modified,
    # the cache keys won't match and nothing will be cleared!
    missing_sources = []
    for photo_source in photos_to_clear:
        source_path = Path(photo_source)
        if not source_path.exists():
            missing_sources.append(photo_source)

    if missing_sources:
        print(f"\n[WARN]  WARNING: {len(missing_sources)} source files not found!")
        print(f"   Cache clearing may fail for these files.")
        print(f"   (Cache keys require original files to compute size+mtime)")
        for p in missing_sources[:5]:
            print(f"      {Path(p).name}")
        if len(missing_sources) > 5:
            print(f"      ... and {len(missing_sources) - 5} more")

        if len(missing_sources) == len(photos_to_clear):
            print(f"\n[ERROR] ALL source files are missing. Cannot proceed.")
            print(f"   Make sure original photos still exist in source folder.")
            sys.exit(1)

    print(f"\n[STATS] Summary:")
    print(f"   Photos to remove from sorted folders: {len(photos_to_clear)}")
    print(f"   Source files verified: {len(photos_to_clear) - len(missing_sources)}/{len(photos_to_clear)}")
    print(f"   Burst cache entries to clear: {len(burst_cache_keys_to_clear)}")
    print(f"   Singleton cache entries to clear: {len(singleton_errors)}")

    if args.dry_run:
        print("\n[DRY-RUN] DRY RUN - would clear:")

        # Show where each photo would be removed from
        destination_folders = [
            output_folder / "Share",
            output_folder / "Storage",
            output_folder / "Review",
            output_folder / "Ignore",
        ]

        by_folder = defaultdict(list)
        for photo_source in photos_to_clear:
            photo_name = Path(photo_source).name
            for dest_folder in destination_folders:
                if (dest_folder / photo_name).exists():
                    by_folder[dest_folder.name].append(photo_name)
                    break

        for folder, photos in sorted(by_folder.items()):
            print(f"\n   From {folder}/ ({len(photos)} files):")
            for photo in photos[:5]:
                print(f"      {photo}")
            if len(photos) > 5:
                print(f"      ... and {len(photos) - 5} more")

        print(f"\n   Would also clear {len(burst_cache_keys_to_clear)} burst cache entries")
        print(f"   Would also clear {len(singleton_errors)} singleton cache entries")
        sys.exit(0)

    # Initialize cache manager
    cache_manager = CacheManager(Path(args.cache_dir))
    gemini_cache_dir = cache_manager.cache_dirs["gemini"]
    burst_context_dir = cache_manager.cache_dirs.get("burst_context", gemini_cache_dir.parent / "burst_context")

    # Clear cache entries
    print("\n[DELETE]  Clearing cache entries...")
    cache_cleared = 0
    cache_not_found = 0

    # Clear singleton cache entries
    for _, row in singleton_errors.iterrows():
        photo_path = Path(row["source"])
        if not photo_path.exists():
            cache_not_found += 1
            continue  # Skip - can't compute correct cache key without file stats
        key = CacheKey.from_file(photo_path)
        cache_file = gemini_cache_dir / f"{key}.json"
        if cache_file.exists():
            cache_file.unlink()
            cache_cleared += 1
        else:
            cache_not_found += 1

    # Clear burst cache entries
    for burst_key in burst_cache_keys_to_clear:
        # Gemini cache
        cache_file = gemini_cache_dir / f"{burst_key}.json"
        if cache_file.exists():
            cache_file.unlink()
            cache_cleared += 1
        else:
            cache_not_found += 1

        # Burst context cache (if exists)
        context_file = burst_context_dir / f"{burst_key}.json"
        if context_file.exists():
            context_file.unlink()

    print(f"   Cleared {cache_cleared} cache entries")
    if cache_not_found > 0:
        print(f"   [WARN]  {cache_not_found} cache entries not found (may already be cleared or keys mismatched)")

    # Remove photos from ALL destination folders (not just Review)
    # This is important for bursts where some photos succeeded and some failed
    print("\n[DELETE]  Removing photos from sorted folders...")

    destination_folders = [
        output_folder / "Share",
        output_folder / "Storage",
        output_folder / "Review",
        output_folder / "Ignore",
    ]

    files_removed = 0
    removal_by_folder = defaultdict(int)

    for photo_source in photos_to_clear:
        photo_name = Path(photo_source).name

        # Check all possible destination folders
        for dest_folder in destination_folders:
            dest_path = dest_folder / photo_name
            if dest_path.exists():
                dest_path.unlink()
                files_removed += 1
                removal_by_folder[dest_folder.name] += 1
                break  # Photo can only be in one folder

    print(f"   Removed {files_removed} photos total:")
    for folder, count in sorted(removal_by_folder.items()):
        print(f"      {folder}: {count}")

    # Summary
    print("\n" + "="*60)
    print("[OK] CLEANUP COMPLETE")
    print("="*60)
    print(f"   Cache entries cleared: {cache_cleared}")
    print(f"   Photos removed from sorted folders: {files_removed}")
    print(f"\nNext steps:")
    print(f"   1. Run: python taste_classify.py \"{output_folder.parent / output_folder.name.replace('_sorted', '')}\"")
    print(f"   2. The {len(photos_to_clear)} photos will be re-classified with retry logic")
    print("="*60)


if __name__ == "__main__":
    main()
