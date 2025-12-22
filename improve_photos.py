#!/usr/bin/env python3
"""
Photo Improvement Script

Process approved improvement candidates using Gemini image generation.
"""

import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

from src.core import load_config
from src.improvement import PhotoImprover, launch_review_ui


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Improve photos flagged as improvement candidates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "C:\\Photos\\MyFolder_sorted"
  %(prog)s "C:\\Photos\\MyFolder_sorted" --review
  %(prog)s "C:\\Photos\\MyFolder_sorted" --dry-run
        """
    )

    parser.add_argument(
        "folder",
        type=str,
        help="Sorted folder containing ImprovementCandidates/"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)"
    )

    parser.add_argument(
        "--review",
        action="store_true",
        help="Open Gradio UI to review/approve candidates before processing"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without actually processing"
    )

    parser.add_argument(
        "--parallel",
        type=int,
        help="Number of parallel workers (default: from config)"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    # Load environment variables
    load_dotenv()

    # Parse arguments
    args = parse_arguments()

    # Validate folder
    folder = Path(args.folder)
    if not folder.exists():
        print(f"❌ Error: Folder not found: {folder}")
        sys.exit(1)

    # Find ImprovementCandidates folder
    candidates_folder = folder / "ImprovementCandidates"
    improved_folder = folder / "Improved"

    if not candidates_folder.exists():
        print(f"❌ Error: ImprovementCandidates folder not found at: {candidates_folder}")
        print("   Run taste_classify.py first to identify improvement candidates.")
        sys.exit(1)

    # Check for candidates CSV
    csv_path = candidates_folder / "improvement_candidates.csv"
    if not csv_path.exists():
        print(f"❌ Error: No improvement_candidates.csv found at: {csv_path}")
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
    if args.parallel:
        config.photo_improvement.parallel_workers = args.parallel

    # Launch review UI if requested
    if args.review:
        print("\n🎨 Launching Gradio review UI...")
        print("   Approve or reject candidates, then close the window to continue.")
        launch_review_ui(candidates_folder)
        print("\n✅ Review complete. Continuing with approved candidates...")

    # Initialize improver
    improver = PhotoImprover(config)

    # Load approved candidates
    candidates = improver.load_candidates(candidates_folder)

    if not candidates:
        print("\n⚠️  No approved candidates found.")
        print("   Mark candidates as approved in the CSV (approved=Y) or use --review")
        sys.exit(0)

    # Show cost estimate
    total_cost = improver.estimate_cost(candidates)

    print("\n" + "="*60)
    print("🎨 PHOTO IMPROVEMENT")
    print("="*60)
    print(f"\nApproved candidates: {len(candidates)}")
    print(f"Cost per image:      ${config.photo_improvement.cost_per_image:.3f}")
    print(f"Estimated total:     ${total_cost:.2f}")
    print(f"\nParallel workers:    {config.photo_improvement.parallel_workers}")

    if args.dry_run:
        print("\n[DRY RUN] Would process the following photos:")
        for i, candidate in enumerate(candidates, 1):
            print(f"   {i}. {candidate.get('filename', 'unknown')}")
            print(f"      Issues: {candidate.get('improvement_reasons', '')}")
            print(f"      Context: {candidate.get('contextual_reasoning', '')[:50]}...")
        print("\n💡 Remove --dry-run to actually process photos.")
        sys.exit(0)

    # Confirm before processing (skip if coming from review UI)
    if not args.review:
        print("\n⚠️  This will use the Gemini image generation API.")
        print("   Processing cannot be undone (but originals are preserved).")

        try:
            response = input("\nProceed with improvement? [y/N]: ").strip().lower()
            if response != "y":
                print("\n❌ Cancelled.")
                sys.exit(0)
        except (EOFError, KeyboardInterrupt):
            print("\n\n❌ Cancelled.")
            sys.exit(0)
    else:
        print("\n✅ Proceeding with approved candidates from review...")

    # Initialize Gemini image client
    print("\n🤖 Initializing Gemini image client...")
    try:
        from src.core.models import GeminiImageClient
        gemini_client = GeminiImageClient(
            model_name=config.photo_improvement.model_name,
            max_output_tokens=config.photo_improvement.max_output_tokens
        )
        improver.client = gemini_client
        print(f"   ✅ Initialized ({config.photo_improvement.model_name})")
    except ImportError:
        print("   ⚠️  GeminiImageClient not yet implemented.")
        print("   This is a placeholder - image generation requires API implementation.")
        sys.exit(1)
    except Exception as e:
        print(f"   ❌ Error: {e}")
        sys.exit(1)

    # Ensure Improved folder exists
    improved_folder.mkdir(parents=True, exist_ok=True)

    # Process candidates
    print(f"\n🚀 Processing {len(candidates)} photos...")
    results = improver.process_batch(
        candidates,
        candidates_folder,
        improved_folder,
        show_progress=True
    )

    # Save results
    improver.save_results_csv(results, improved_folder)

    # Print summary
    improver.print_summary(results)

    print(f"\n📂 Improved photos saved to: {improved_folder}")
    print("\n✅ Improvement complete!")


if __name__ == "__main__":
    main()
