#!/usr/bin/env python3
"""
LLM Taste Cloner - Main Classification Script

Automatically classify and sort family photos/videos/documents using Google Gemini AI.
Supports multiple taste profiles and mixed media types.
"""

import os
import sys
import argparse
import shutil
import csv
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
    ProfileManager,
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
from src.pipelines import MixedPipeline, PhotoPipeline, DocumentPipeline


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Classify family photos/videos/documents using Google Gemini AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "C:\\Photos\\MyFolder"
  %(prog)s "C:\\Photos\\MyFolder" --profile default-photos
  %(prog)s "C:\\Photos\\MyFolder" --classify-videos
  %(prog)s "C:\\Photos\\MyFolder" --dry-run
  %(prog)s "C:\\Photos\\MyFolder" --config custom_config.yaml
        """
    )

    parser.add_argument(
        "folder",
        type=str,
        help="Folder containing photos/videos/documents to classify"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)"
    )

    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Taste profile name to use (default: from config or auto-detect)"
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


def setup_output_directories(output_base: Path, classify_videos: bool = True, photo_improvement_enabled: bool = False, profile=None) -> dict:
    """Create output directories for sorted media."""
    if profile:
        # Use profile-defined categories
        directories = {}
        for cat in profile.categories:
            directories[cat.name] = output_base / cat.name
        directories["Reports"] = output_base / "Reports"
    else:
        # Default directories
        directories = {
            "Share": output_base / "Share",
            "Storage": output_base / "Storage",
            "Review": output_base / "Review",
            "Ignore": output_base / "Ignore",
            "Reports": output_base / "Reports",
        }

    # Only create Videos folder when video classification is disabled
    if not classify_videos:
        directories["Videos"] = output_base / "Videos"

    # Create improvement directories if enabled
    if photo_improvement_enabled:
        directories["ImprovementCandidates"] = output_base / "ImprovementCandidates"
        directories["Improved"] = output_base / "Improved"

    for dir_path in directories.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    return directories


def generate_report(results: list, report_dir: Path, include_improvement: bool = False):
    """Generate classification report CSV."""
    rows = []

    for result in results:
        classification = result.get("classification", {})
        row = {
            "source": str(result["path"]),
            "destination": result.get("destination", "Unknown"),
            "classification": classification.get("classification", "Unknown"),
            "confidence": classification.get("confidence", 0),
            "burst_size": result.get("burst_size", result.get("group_size", 1)),
            "burst_index": result.get("burst_index", result.get("group_index", -1)),
            "rank": classification.get("rank", None),
            "reasoning": classification.get("reasoning", ""),
            "contains_children": classification.get("contains_children", None),
            "is_appropriate": classification.get("is_appropriate", None),
            "content_summary": classification.get("content_summary", ""),
            "key_topics": ",".join(classification.get("key_topics", [])),
            # Error tracking fields
            "is_error_fallback": classification.get("is_error_fallback", False),
            "error_type": classification.get("error_type", ""),
            "error_message": classification.get("error_message", ""),
            "retry_count": classification.get("retry_count", 0),
            "burst_id": classification.get("burst_id", ""),
        }

        # Add improvement fields if enabled
        if include_improvement:
            row["improvement_candidate"] = classification.get("improvement_candidate", False)
            row["improvement_reasons"] = ",".join(classification.get("improvement_reasons", []))
            row["contextual_value"] = classification.get("contextual_value", "low")
            row["contextual_value_reasoning"] = classification.get("contextual_value_reasoning", "")

        rows.append(row)

    df = pd.DataFrame(rows)
    report_path = report_dir / f"classification_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(report_path, index=False)
    print(f"\nReport saved: {report_path}")


def save_improvement_candidates_csv(results: list, output_dirs: dict, cost_per_image: float):
    """Save improvement candidates to a CSV file for human review."""
    candidates_dir = output_dirs.get("ImprovementCandidates")
    if candidates_dir is None:
        return 0

    candidates = [r for r in results if r.get("destination") == "ImprovementCandidates"]
    if not candidates:
        return 0

    csv_path = candidates_dir / "improvement_candidates.csv"
    rows = []
    for result in candidates:
        classification = result["classification"]
        rows.append({
            "filename": result["path"].name,
            "original_path": str(result["path"]),
            "contextual_value": classification.get("contextual_value", ""),
            "contextual_reasoning": classification.get("contextual_value_reasoning", ""),
            "improvement_reasons": ",".join(classification.get("improvement_reasons", [])),
            "estimated_cost": f"${cost_per_image:.3f}",
            "approved": "",
            "status": "pending"
        })

    fieldnames = ["filename", "original_path", "contextual_value", "contextual_reasoning",
                  "improvement_reasons", "estimated_cost", "approved", "status"]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nImprovement candidates CSV saved: {csv_path}")
    return len(candidates)


def prompt_for_improvement_review(candidates_count: int, output_dirs: dict, total_cost: float) -> bool:
    """Prompt user to review improvement candidates."""
    if candidates_count == 0:
        return False

    print("\n" + "="*60)
    print("IMPROVEMENT CANDIDATES DETECTED")
    print("="*60)
    print(f"\nFound {candidates_count} photos with high contextual value but technical issues.")
    print(f"Estimated improvement cost: ${total_cost:.2f}")
    print(f"\nCandidates saved to: {output_dirs['ImprovementCandidates']}")
    print("CSV file: improvement_candidates.csv")
    print("\nOptions:")
    print("  [y] Review candidates now (opens Gradio UI)")
    print("  [n] Review later (edit CSV or run improve_photos.py --review)")

    try:
        response = input("\nReview improvement candidates now? [y/N]: ").strip().lower()
        return response == "y"
    except (EOFError, KeyboardInterrupt):
        return False


def main():
    """Main entry point."""
    # Load environment variables
    load_dotenv()

    # Parse arguments
    args = parse_arguments()

    # Validate input folder
    folder = Path(args.folder)
    if not folder.exists():
        print(f"Error: Folder not found: {folder}")
        sys.exit(1)

    # Load configuration
    print(f"Loading configuration from {args.config}...")
    try:
        config = load_config(Path(args.config))
    except FileNotFoundError:
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading config: {e}")
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

    # Load taste profile
    profile = None
    profile_name = args.profile or config.profiles.active_profile
    profile_manager = ProfileManager(config.profiles.profiles_dir)

    if profile_manager.profile_exists(profile_name):
        profile = profile_manager.load_profile(profile_name)
        print(f"Using taste profile: {profile.name} ({profile.description})")
    else:
        # Auto-detect media type and use appropriate default
        if config.profiles.auto_detect_media_type:
            media_type = FileTypeRegistry.detect_media_type(folder)
            print(f"Auto-detected media type: {media_type}")
            profile = profile_manager.get_default_profile(media_type)
            print(f"Using default profile: {profile.name}")
        else:
            print(f"Profile '{profile_name}' not found, using legacy taste preferences")

    # Determine output directory
    if args.output:
        output_base = Path(args.output)
    else:
        output_base = folder.parent / f"{folder.name}_sorted"

    print(f"Output directory: {output_base}")

    # Setup output directories
    output_dirs = setup_output_directories(
        output_base,
        config.classification.classify_videos,
        config.photo_improvement.enabled,
        profile=profile
    )

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
        print(f"Gemini client initialized ({config.model.name})")
    except ValueError as e:
        print(f"Error: {e}")
        print("   Make sure GEMINI_API_KEY is set in .env file")
        sys.exit(1)

    # Scan folder contents
    media = FileTypeRegistry.list_all_media(folder)
    total = len(media["images"]) + len(media["videos"]) + len(media["documents"])
    print(f"\nScanning folder: {folder}")
    print(f"   Found {len(media['images'])} images, {len(media['videos'])} videos, {len(media['documents'])} documents")

    if total == 0:
        print("\nNo media files found to process.")
        sys.exit(0)

    # Run the appropriate pipeline
    pipeline = MixedPipeline(
        config, profile, cache_manager, gemini_client
    )
    result = pipeline.run(
        folder, output_base, config.system.dry_run,
        classify_videos=config.classification.classify_videos
    )

    # Generate report
    if result.results:
        report_dir = output_dirs.get("Reports", output_base / "Reports")
        report_dir.mkdir(parents=True, exist_ok=True)
        generate_report(result.results, report_dir, config.photo_improvement.enabled)

    # Print summary
    result.print_summary("media files")

    if config.system.dry_run:
        print("\nThis was a dry run. No files were moved.")
        print("   Remove --dry-run to actually move files.")

    # Handle improvement candidates
    improvement_count = 0
    if config.photo_improvement.enabled and not config.system.dry_run:
        improvement_count = save_improvement_candidates_csv(
            result.results,
            output_dirs,
            config.photo_improvement.cost_per_image
        )

        if improvement_count > 0 and config.photo_improvement.review_after_sort:
            total_cost = improvement_count * config.photo_improvement.cost_per_image
            wants_review = prompt_for_improvement_review(improvement_count, output_dirs, total_cost)

            if wants_review:
                try:
                    from src.improvement.review_ui import launch_review_ui
                    from src.improvement import PhotoImprover
                    from src.core.models import GeminiImageClient

                    print("\nLaunching Gradio review UI...")
                    launch_review_ui(output_dirs["ImprovementCandidates"])

                    print("\nReview complete!")
                    improver = PhotoImprover(config)
                    approved = improver.load_candidates(output_dirs["ImprovementCandidates"])

                    if approved:
                        approved_cost = len(approved) * config.photo_improvement.cost_per_image
                        print(f"\n{len(approved)} photos approved for improvement")
                        print(f"   Estimated cost: ${approved_cost:.2f}")
                        print(f"   Model: {config.photo_improvement.model_name}")

                        print("\nInitializing Gemini image client...")
                        try:
                            gemini_image_client = GeminiImageClient(
                                model_name=config.photo_improvement.model_name,
                                max_output_tokens=config.photo_improvement.max_output_tokens
                            )
                            improver.client = gemini_image_client
                            print(f"   Ready")

                            output_dirs["Improved"].mkdir(parents=True, exist_ok=True)

                            print(f"\nProcessing {len(approved)} photos...")
                            imp_results = improver.process_batch(
                                approved,
                                output_dirs["ImprovementCandidates"],
                                output_dirs["Improved"],
                                show_progress=True
                            )

                            improver.save_results_csv(imp_results, output_dirs["Improved"])
                            improver.print_summary(imp_results)
                            print(f"\nImproved photos saved to: {output_dirs['Improved']}")

                        except ImportError as e:
                            print(f"   Missing dependency: {e}")
                            print("   Run: pip install google-genai")
                        except Exception as e:
                            print(f"   Error: {e}")
                    else:
                        print("\nNo photos were approved for improvement.")

                except ImportError:
                    print("\nGradio review UI not available.")
                    print(f"   Run: python improve_photos.py \"{output_base}\" --review")

    print("\nClassification complete!")


if __name__ == "__main__":
    main()
