#!/usr/bin/env python3
"""
Reprocess Failed Classifications

Re-runs classification on photos that failed due to errors (API failures, timeouts, etc.)
in a previous run. Supports moving files to correct destinations after re-classification.

Usage:
    python reprocess_failed.py "C:\\Photos\\folder_sorted"
    python reprocess_failed.py "C:\\Photos\\folder_sorted" --report classification_report_20241225_120000.csv
    python reprocess_failed.py "C:\\Photos\\folder_sorted" --dry-run
    python reprocess_failed.py "C:\\Photos\\folder_sorted" --move-files
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

# Import infrastructure
from sommelier.core import (
    load_config,
    CacheManager,
    create_ai_client,
)
from sommelier.classification import (
    PromptBuilder,
    MediaClassifier,
    Router,
)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Re-process failed photo classifications",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "C:\\Photos\\folder_sorted"
  %(prog)s "C:\\Photos\\folder_sorted" --report classification_report_20241225_120000.csv
  %(prog)s "C:\\Photos\\folder_sorted" --dry-run
  %(prog)s "C:\\Photos\\folder_sorted" --move-files
  %(prog)s "C:\\Photos\\folder_sorted" --error-types api_error,timeout
        """
    )

    parser.add_argument(
        "output_folder",
        type=str,
        help="Sorted output folder (contains Reports/, Review/, etc.)"
    )

    parser.add_argument(
        "--report",
        type=str,
        help="Specific report CSV to use (default: most recent in Reports/)"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be reprocessed without actually doing it"
    )

    parser.add_argument(
        "--move-files",
        action="store_true",
        help="Move files from Review to correct destination after re-classification"
    )

    parser.add_argument(
        "--error-types",
        type=str,
        help="Comma-separated list of error types to reprocess (default: api_error,invalid_response,timeout,rate_limit)"
    )

    parser.add_argument(
        "--cache-dir",
        type=str,
        help="Cache directory (default: .taste_cache)"
    )

    return parser.parse_args()


def find_latest_report(reports_dir: Path) -> Path:
    """Find the most recent classification report."""
    reports = list(reports_dir.glob("classification_report_*.csv"))
    if not reports:
        raise FileNotFoundError(f"No classification reports found in {reports_dir}")

    # Sort by modification time, newest first
    reports.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return reports[0]


def load_failed_classifications(report_path: Path, error_types: list) -> pd.DataFrame:
    """
    Load items from CSV that are error fallbacks with retriable error types.

    Args:
        report_path: Path to classification report CSV.
        error_types: List of error types to consider for reprocessing.

    Returns:
        DataFrame of failed classifications.
    """
    df = pd.read_csv(report_path)

    # Check if error tracking columns exist
    if "is_error_fallback" not in df.columns:
        print("⚠️  This report doesn't have error tracking columns.")
        print("   Run taste_classify.py again to generate a report with error tracking.")
        return pd.DataFrame()

    # Filter to error fallbacks with retriable error types
    failed = df[df["is_error_fallback"] == True]
    if "error_type" in df.columns:
        failed = failed[failed["error_type"].isin(error_types)]

    return failed


def group_by_burst(failed_df: pd.DataFrame) -> dict:
    """
    Group failed items by their burst membership.

    Args:
        failed_df: DataFrame of failed classifications.

    Returns:
        Dictionary mapping burst_id to list of failed items.
    """
    bursts = defaultdict(list)

    for _, row in failed_df.iterrows():
        burst_id = row.get("burst_id", "")
        if pd.isna(burst_id) or burst_id == "":
            # Singleton - use source path as key
            burst_id = f"singleton_{row['source']}"
        bursts[burst_id].append(row.to_dict())

    return dict(bursts)


def get_burst_context(cache_manager: CacheManager, burst_id: str) -> dict:
    """
    Get burst context from cache.

    Args:
        cache_manager: Cache manager instance.
        burst_id: Burst ID to look up.

    Returns:
        Burst context dict or empty dict if not found.
    """
    if burst_id.startswith("singleton_"):
        return {}

    context = cache_manager.get("burst_context", burst_id)
    return context if context else {}


def reprocess_singleton(
    photo_path: Path,
    classifier: MediaClassifier,
    router: Router,
    burst_context: dict = None,
    failed_item: dict = None
) -> dict:
    """
    Re-classify a singleton photo.

    Args:
        photo_path: Path to photo.
        classifier: MediaClassifier instance.
        router: Router instance.
        burst_context: Optional burst context for photos that were part of a burst.
        failed_item: Original failed classification info.

    Returns:
        New classification result.
    """
    # Check if this was part of a burst
    if burst_context and burst_context.get("results"):
        # Get sibling classifications
        sibling_classifications = [
            r.get("classification", "Unknown")
            for i, r in enumerate(burst_context.get("results", []))
            if i not in burst_context.get("failed_indices", [])
        ]
        original_position = failed_item.get("burst_index", 0) if failed_item else 0

        # Use context-aware classification
        prompt = classifier.prompt_builder.build_singleton_with_burst_context(
            burst_size=len(burst_context.get("photo_paths", [])),
            sibling_classifications=sibling_classifications,
            original_position=original_position
        )
        # Note: For now, we use regular singleton classification but the prompt_builder
        # method is available for future integration

    # Classify (cache is disabled to force re-evaluation)
    classification = classifier.classify_singleton(photo_path, use_cache=False)
    destination = router.route_singleton(classification)

    return {
        "path": photo_path,
        "classification": classification,
        "destination": destination,
        "previous_destination": "Review",
        "reprocessed": True
    }


def reprocess_burst(
    burst_photos: list,
    classifier: MediaClassifier,
    router: Router,
    embeddings_extractor=None
) -> list:
    """
    Re-classify an entire burst.

    Args:
        burst_photos: List of photo paths in burst.
        classifier: MediaClassifier instance.
        router: Router instance.
        embeddings_extractor: Optional embeddings extractor for routing.

    Returns:
        List of classification results.
    """
    # Re-classify the burst
    classifications = classifier.classify_burst(burst_photos, use_cache=False)

    # For routing, we need embeddings - use None for now (router will skip diversity check)
    destinations = router.route_burst(burst_photos, classifications, embeddings=None)

    results = []
    for i, (photo, classification, destination) in enumerate(zip(burst_photos, classifications, destinations)):
        results.append({
            "path": photo,
            "classification": classification,
            "destination": destination,
            "previous_destination": "Review",
            "reprocessed": True
        })

    return results


def move_file(source: Path, dest_folder: Path, dry_run: bool = False) -> bool:
    """
    Move a file from Review to correct destination.

    Args:
        source: Source file path.
        dest_folder: Destination folder.
        dry_run: If True, just print what would happen.

    Returns:
        True if successful, False otherwise.
    """
    dest_path = dest_folder / source.name

    if dry_run:
        print(f"   [DRY RUN] Would move: {source.name} → {dest_folder.name}/")
        return True

    try:
        shutil.move(str(source), str(dest_path))
        return True
    except Exception as e:
        print(f"   ⚠️ Error moving {source.name}: {e}")
        return False


def save_reprocessing_report(results: list, report_dir: Path):
    """Save reprocessing results to CSV."""
    rows = []
    for result in results:
        classification = result["classification"]
        rows.append({
            "source": str(result["path"]),
            "previous_destination": result.get("previous_destination", "Review"),
            "new_destination": result["destination"],
            "classification": classification.get("classification", "Unknown"),
            "confidence": classification.get("confidence", 0),
            "reasoning": classification.get("reasoning", ""),
            "is_error_fallback": classification.get("is_error_fallback", False),
            "error_type": classification.get("error_type", ""),
            "retry_count": classification.get("retry_count", 0),
            "moved": result.get("moved", False),
        })

    df = pd.DataFrame(rows)
    report_path = report_dir / f"reprocessing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(report_path, index=False)
    print(f"\n📊 Reprocessing report saved: {report_path}")


def print_summary(results: list):
    """Print summary of reprocessing results."""
    total = len(results)
    if total == 0:
        print("\n⚠️  No items were reprocessed.")
        return

    # Count by new destination
    dest_counts = defaultdict(int)
    still_review = 0
    still_error = 0

    for result in results:
        dest = result["destination"]
        dest_counts[dest] += 1

        if dest == "Review":
            still_review += 1
        if result["classification"].get("is_error_fallback"):
            still_error += 1

    print("\n" + "="*60)
    print("📊 REPROCESSING SUMMARY")
    print("="*60)
    print(f"\nTotal reprocessed: {total}")
    print("\nNew destinations:")
    for dest, count in sorted(dest_counts.items()):
        pct = (count / total) * 100
        print(f"   {dest:15s}: {count:4d} ({pct:.1f}%)")

    if still_error > 0:
        print(f"\n⚠️  {still_error} items still have errors (may need manual review)")

    recovered = total - still_review
    if recovered > 0:
        print(f"\n✅ Recovered {recovered} items from Review folder!")

    print("="*60)


def main():
    """Main entry point."""
    # Load environment variables
    load_dotenv()

    # Parse arguments
    args = parse_arguments()

    # Validate output folder
    output_folder = Path(args.output_folder)
    if not output_folder.exists():
        print(f"❌ Error: Output folder not found: {output_folder}")
        sys.exit(1)

    reports_dir = output_folder / "Reports"
    review_dir = output_folder / "Review"

    if not reports_dir.exists():
        print(f"❌ Error: Reports folder not found: {reports_dir}")
        sys.exit(1)

    # Find report
    if args.report:
        report_path = reports_dir / args.report
        if not report_path.exists():
            print(f"❌ Error: Report not found: {report_path}")
            sys.exit(1)
    else:
        try:
            report_path = find_latest_report(reports_dir)
        except FileNotFoundError as e:
            print(f"❌ Error: {e}")
            sys.exit(1)

    print(f"📄 Using report: {report_path.name}")

    # Load configuration
    print(f"⚙️  Loading configuration from {args.config}...")
    try:
        config = load_config(Path(args.config))
    except FileNotFoundError:
        print(f"❌ Error: Config file not found: {args.config}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        sys.exit(1)

    # Determine error types to reprocess
    if args.error_types:
        error_types = [e.strip() for e in args.error_types.split(",")]
    else:
        error_types = config.classification.retry_on_errors

    print(f"🔍 Looking for error types: {', '.join(error_types)}")

    # Load failed classifications
    failed_df = load_failed_classifications(report_path, error_types)

    if len(failed_df) == 0:
        print("\n✅ No failed classifications found to reprocess!")
        sys.exit(0)

    print(f"\n📊 Found {len(failed_df)} failed classifications to reprocess")

    # Group by burst
    burst_groups = group_by_burst(failed_df)
    singleton_count = sum(1 for k in burst_groups.keys() if k.startswith("singleton_"))
    burst_count = len(burst_groups) - singleton_count

    print(f"   Singletons: {singleton_count}")
    print(f"   Burst groups: {burst_count}")

    if args.dry_run:
        print("\n🔍 DRY RUN - showing what would be reprocessed:")
        for burst_id, items in burst_groups.items():
            if burst_id.startswith("singleton_"):
                print(f"   Singleton: {Path(items[0]['source']).name}")
            else:
                print(f"   Burst ({len(items)} photos): {[Path(i['source']).name for i in items]}")
        sys.exit(0)

    # Initialize cache manager
    cache_dir = Path(args.cache_dir) if args.cache_dir else config.paths.cache_root
    cache_manager = CacheManager(
        cache_dir,
        ttl_days=config.caching.ttl_days,
        enabled=config.caching.enabled
    )

    # Initialize AI client
    try:
        gemini_client = create_ai_client(config)
        print(f"AI client initialized: {gemini_client.provider_name}")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Initialize classification components
    prompt_builder = PromptBuilder(config, training_examples={})
    classifier = MediaClassifier(config, gemini_client, prompt_builder, cache_manager)
    router = Router(config, gemini_client)

    # Process each group
    print(f"\n🔄 Reprocessing {len(failed_df)} items...")
    results = []

    for burst_id, items in tqdm(burst_groups.items(), desc="Processing"):
        if burst_id.startswith("singleton_"):
            # Singleton
            item = items[0]
            photo_path = Path(item["source"])

            # Check if file exists in Review folder
            review_path = review_dir / photo_path.name
            if review_path.exists():
                actual_path = review_path
            elif photo_path.exists():
                actual_path = photo_path
            else:
                print(f"   ⚠️ File not found: {photo_path.name}")
                continue

            result = reprocess_singleton(
                actual_path,
                classifier,
                router,
                burst_context=None,
                failed_item=item
            )
            results.append(result)
        else:
            # Burst - get context
            burst_context = get_burst_context(cache_manager, burst_id)

            # Check if all burst photos failed or just some
            if burst_context:
                all_failed = len(burst_context.get("failed_indices", [])) == len(burst_context.get("photo_paths", []))
            else:
                all_failed = True

            if all_failed:
                # Re-run entire burst
                burst_photos = []
                for item in items:
                    photo_path = Path(item["source"])
                    review_path = review_dir / photo_path.name
                    if review_path.exists():
                        burst_photos.append(review_path)
                    elif photo_path.exists():
                        burst_photos.append(photo_path)
                    else:
                        print(f"   ⚠️ File not found: {photo_path.name}")

                if burst_photos:
                    burst_results = reprocess_burst(burst_photos, classifier, router)
                    results.extend(burst_results)
            else:
                # Process each failed photo as singleton with context
                for item in items:
                    photo_path = Path(item["source"])
                    review_path = review_dir / photo_path.name
                    if review_path.exists():
                        actual_path = review_path
                    elif photo_path.exists():
                        actual_path = photo_path
                    else:
                        print(f"   ⚠️ File not found: {photo_path.name}")
                        continue

                    result = reprocess_singleton(
                        actual_path,
                        classifier,
                        router,
                        burst_context=burst_context,
                        failed_item=item
                    )
                    results.append(result)

    # Move files if requested
    if args.move_files and results:
        print(f"\n📦 Moving files to correct destinations...")
        moved_count = 0

        for result in results:
            # Only move if destination changed from Review
            if result["destination"] != "Review":
                source = result["path"]
                dest_folder = output_folder / result["destination"]
                dest_folder.mkdir(parents=True, exist_ok=True)

                if move_file(source, dest_folder, dry_run=False):
                    result["moved"] = True
                    moved_count += 1
                else:
                    result["moved"] = False

        print(f"   Moved {moved_count} files")

    # Save report
    save_reprocessing_report(results, reports_dir)

    # Print summary
    print_summary(results)

    print("\n✅ Reprocessing complete!")


if __name__ == "__main__":
    main()
