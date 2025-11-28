#!/usr/bin/env python3
"""
LLM Taste Cloner - Main Classification Script

Automatically classify and sort family photos/videos using Google Gemini AI.
Refactored version using new modular infrastructure.
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

# Import refactored infrastructure
from src.core import (
    load_config,
    CacheManager,
    GeminiClient,
    FileTypeRegistry,
)
from src.features import (
    QualityScorer,
    BurstDetector,
    EmbeddingExtractor,
)
from src.classification import (
    PromptBuilder,
    MediaClassifier,
    Router,
)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Classify family photos/videos using Google Gemini AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "C:\\Photos\\MyFolder"
  %(prog)s "C:\\Photos\\MyFolder" --classify-videos
  %(prog)s "C:\\Photos\\MyFolder" --dry-run
  %(prog)s "C:\\Photos\\MyFolder" --config custom_config.yaml
        """
    )

    parser.add_argument(
        "folder",
        type=str,
        help="Folder containing photos/videos to classify"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)"
    )

    parser.add_argument(
        "--classify-videos",
        action="store_true",
        help="Enable video classification (ON by default in config.yaml, use --no-classify-videos to disable)"
    )

    parser.add_argument(
        "--no-classify-videos",
        action="store_true",
        help="Disable video classification"
    )

    parser.add_argument(
        "--parallel-videos",
        type=int,
        help="Number of parallel workers for video classification (default: from config)"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't move files, just show what would happen"
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Output directory (default: <folder>_sorted)"
    )

    parser.add_argument(
        "--cache-dir",
        type=str,
        help="Cache directory (default: .taste_cache)"
    )

    return parser.parse_args()


def setup_output_directories(output_base: Path) -> dict:
    """Create output directories for sorted media."""
    directories = {
        "Share": output_base / "Share",
        "Storage": output_base / "Storage",
        "Review": output_base / "Review",
        "Ignore": output_base / "Ignore",
        "Videos": output_base / "Videos",
        "Reports": output_base / "Reports",
    }

    for dir_path in directories.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    return directories


def collect_media_files(folder: Path, config) -> dict:
    """Collect all media files from folder."""
    print(f"\n📁 Scanning folder: {folder}")

    images = FileTypeRegistry.list_images(folder, recursive=False)
    videos = FileTypeRegistry.list_videos(folder, recursive=False)

    print(f"   Found {len(images)} images, {len(videos)} videos")

    return {
        "images": images,
        "videos": videos
    }


def process_images(
    images: list,
    config,
    cache_manager: CacheManager,
    gemini_client: GeminiClient,
    output_dirs: dict,
    dry_run: bool
):
    """Process all images (detect bursts, classify, route)."""
    if not images:
        print("\n📸 No images to process")
        return []

    print(f"\n📸 Processing {len(images)} images...")

    # 1. Extract embeddings for burst detection
    print("\n🔍 Step 1: Extracting CLIP embeddings...")
    extractor = EmbeddingExtractor(config.model, config.performance, cache_manager)
    embeddings = extractor.extract_embeddings_batch(images, use_cache=True, show_progress=True)

    # 2. Detect bursts
    print("\n🔍 Step 2: Detecting photo bursts...")
    detector = BurstDetector(config.burst_detection)
    bursts = detector.detect_bursts(images, embeddings)

    # 3. Initialize classification components
    prompt_builder = PromptBuilder(config, training_examples={})
    classifier = MediaClassifier(config, gemini_client, prompt_builder, cache_manager)
    router = Router(config, gemini_client)

    # 4. Classify all photos
    print(f"\n🤖 Step 3: Classifying photos with Gemini...")
    results = []

    for burst_idx, burst in enumerate(tqdm(bursts, desc="Processing bursts/singletons")):
        if len(burst) == 1:
            # Singleton
            photo = burst[0]
            classification = classifier.classify_singleton(photo, use_cache=True)
            destination = router.route_singleton(classification)

            results.append({
                "path": photo,
                "burst_size": 1,
                "burst_index": -1,
                "classification": classification,
                "destination": destination
            })

        else:
            # Burst
            # Get burst indices for embeddings
            burst_indices = [images.index(photo) for photo in burst]
            burst_embeddings = embeddings[burst_indices]

            classifications = classifier.classify_burst(burst, use_cache=True)
            destinations = router.route_burst(burst, classifications, burst_embeddings)

            for i, (photo, classification, destination) in enumerate(zip(burst, classifications, destinations)):
                results.append({
                    "path": photo,
                    "burst_size": len(burst),
                    "burst_index": i,
                    "classification": classification,
                    "destination": destination
                })

    # 5. Move files to destinations
    print(f"\n📦 Step 4: Moving files to destination folders...")
    stats = move_files(results, output_dirs, dry_run)

    # 6. Generate report
    generate_report(results, output_dirs["Reports"])

    return results


def process_videos(
    videos: list,
    config,
    cache_manager: CacheManager,
    gemini_client: GeminiClient,
    output_dirs: dict,
    dry_run: bool,
    classify_videos: bool
):
    """Process all videos."""
    if not videos:
        print("\n🎥 No videos to process")
        return []

    print(f"\n🎥 Processing {len(videos)} videos...")

    results = []

    if not classify_videos:
        # Just copy videos to Videos folder
        print("   Video classification disabled, copying to Videos/ folder...")
        for video in tqdm(videos, desc="Copying videos"):
            results.append({
                "path": video,
                "classification": {"classification": "Videos", "confidence": None},
                "destination": "Videos"
            })
    else:
        # Classify videos with Gemini
        print(f"   Classifying videos with {config.classification.parallel_video_workers} parallel workers...")

        prompt_builder = PromptBuilder(config, training_examples={})
        classifier = MediaClassifier(config, gemini_client, prompt_builder, cache_manager)
        router = Router(config, gemini_client)

        for video in tqdm(videos, desc="Classifying videos"):
            classification = classifier.classify_video(video, use_cache=True)
            destination = router.route_video(classification)

            results.append({
                "path": video,
                "classification": classification,
                "destination": destination
            })

    # Move files
    print(f"\n📦 Moving videos to destination folders...")
    stats = move_files(results, output_dirs, dry_run)

    return results


def move_files(results: list, output_dirs: dict, dry_run: bool) -> dict:
    """Move files to their destination folders."""
    stats = defaultdict(int)

    for result in results:
        source = result["path"]
        destination = result["destination"]

        dst_dir = output_dirs.get(destination, output_dirs["Review"])
        dst_path = dst_dir / source.name

        if dry_run:
            print(f"   [DRY RUN] {source.name} → {destination}/")
        else:
            try:
                shutil.copy2(source, dst_path)
                stats[destination] += 1
            except Exception as e:
                print(f"   ⚠️ Error copying {source.name}: {e}")
                stats["Error"] += 1

    return stats


def generate_report(results: list, report_dir: Path):
    """Generate classification report CSV."""
    rows = []

    for result in results:
        classification = result["classification"]
        rows.append({
            "source": str(result["path"]),
            "destination": result["destination"],
            "classification": classification.get("classification", "Unknown"),
            "confidence": classification.get("confidence", 0),
            "burst_size": result.get("burst_size", 1),
            "burst_index": result.get("burst_index", -1),
            "rank": classification.get("rank", None),
            "reasoning": classification.get("reasoning", ""),
            "contains_children": classification.get("contains_children", None),
            "is_appropriate": classification.get("is_appropriate", None),
        })

    df = pd.DataFrame(rows)
    report_path = report_dir / f"classification_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(report_path, index=False)
    print(f"\n📊 Report saved: {report_path}")


def print_summary(image_stats: dict, video_stats: dict):
    """Print classification summary."""
    print("\n" + "="*60)
    print("📊 CLASSIFICATION SUMMARY")
    print("="*60)

    if image_stats:
        print("\n📸 Images:")
        for destination, count in sorted(image_stats.items()):
            print(f"   {destination:10s}: {count:4d} photos")
        print(f"   {'TOTAL':10s}: {sum(image_stats.values()):4d} photos")

    if video_stats:
        print("\n🎥 Videos:")
        for destination, count in sorted(video_stats.items()):
            print(f"   {destination:10s}: {count:4d} videos")
        print(f"   {'TOTAL':10s}: {sum(video_stats.values()):4d} videos")

    print("\n" + "="*60)


def main():
    """Main entry point."""
    # Load environment variables
    load_dotenv()

    # Parse arguments
    args = parse_arguments()

    # Validate input folder
    folder = Path(args.folder)
    if not folder.exists():
        print(f"❌ Error: Folder not found: {folder}")
        sys.exit(1)

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

    # Override config with CLI arguments
    if args.no_classify_videos:
        config.classification.classify_videos = False
    elif args.classify_videos:
        config.classification.classify_videos = True

    if args.parallel_videos:
        config.classification.parallel_video_workers = args.parallel_videos

    if args.dry_run:
        config.system.dry_run = True

    if args.cache_dir:
        config.paths.cache_root = Path(args.cache_dir)

    # Determine output directory
    if args.output:
        output_base = Path(args.output)
    else:
        output_base = folder.parent / f"{folder.name}_sorted"

    print(f"📂 Output directory: {output_base}")

    # Setup output directories
    output_dirs = setup_output_directories(output_base)

    # Initialize cache manager
    cache_manager = CacheManager(
        config.paths.cache_root,
        ttl_days=config.caching.ttl_days,
        enabled=config.caching.enabled
    )

    # Initialize Gemini client
    try:
        gemini_client = GeminiClient(
            model_name=config.model.name,
            max_retries=config.system.max_retries,
            retry_delay=config.system.retry_delay_seconds
        )
        print(f"✅ Gemini client initialized ({config.model.name})")
    except ValueError as e:
        print(f"❌ Error: {e}")
        print("   Make sure GEMINI_API_KEY is set in .env file")
        sys.exit(1)

    # Collect media files
    media = collect_media_files(folder, config)

    # Process images
    image_results = process_images(
        media["images"],
        config,
        cache_manager,
        gemini_client,
        output_dirs,
        config.system.dry_run
    )

    # Process videos
    video_results = process_videos(
        media["videos"],
        config,
        cache_manager,
        gemini_client,
        output_dirs,
        config.system.dry_run,
        config.classification.classify_videos
    )

    # Calculate statistics
    image_stats = defaultdict(int)
    for result in image_results:
        image_stats[result["destination"]] += 1

    video_stats = defaultdict(int)
    for result in video_results:
        video_stats[result["destination"]] += 1

    # Print summary
    print_summary(image_stats, video_stats)

    if config.system.dry_run:
        print("\n💡 This was a dry run. No files were moved.")
        print("   Remove --dry-run to actually move files.")

    print("\n✅ Classification complete!")


if __name__ == "__main__":
    main()
