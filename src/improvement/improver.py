"""Photo improvement using Gemini image generation."""

import csv
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


class PhotoImprover:
    """
    Improves photos using Gemini image generation.

    Handles:
    - Loading approved candidates from CSV
    - Crafting targeted improvement prompts
    - Calling Gemini image generation API
    - Saving improved photos
    """

    # Mapping of improvement reason codes to prompt instructions
    IMPROVEMENT_INSTRUCTIONS = {
        "motion_blur": "Reduce motion blur while preserving natural movement feel",
        "focus_blur": "Sharpen the main subjects, especially faces and people",
        "noise": "Reduce image noise while preserving fine detail",
        "underexposed": "Brighten shadows and improve overall exposure",
        "overexposed": "Recover blown highlights and balance exposure",
        "white_balance": "Correct color temperature for natural skin tones",
        "low_resolution": "Enhance resolution and detail",
        "composition": "The composition could be improved (crop suggestions provided separately)"
    }

    def __init__(
        self,
        config,
        gemini_image_client=None
    ):
        """
        Initialize the photo improver.

        Args:
            config: Configuration object.
            gemini_image_client: Gemini image generation client.
        """
        self.config = config
        self.client = gemini_image_client
        self.parallel_workers = config.photo_improvement.parallel_workers
        self.max_retries = config.photo_improvement.max_retries
        self.cost_per_image = config.photo_improvement.cost_per_image

    def load_candidates(self, candidates_folder: Path) -> List[Dict]:
        """
        Load approved candidates from CSV.

        Args:
            candidates_folder: Path to ImprovementCandidates folder.

        Returns:
            List of approved candidate dictionaries.
        """
        csv_path = candidates_folder / "improvement_candidates.csv"
        if not csv_path.exists():
            return []

        candidates = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Only include approved candidates
                if row.get("approved", "").upper() == "Y":
                    candidates.append(row)

        return candidates

    def estimate_cost(self, candidates: List[Dict]) -> float:
        """
        Calculate estimated cost for improving candidates.

        Args:
            candidates: List of candidate dictionaries.

        Returns:
            Total estimated cost.
        """
        return len(candidates) * self.cost_per_image

    def craft_improvement_prompt(self, candidate: Dict) -> str:
        """
        Build targeted improvement prompt based on detected issues.

        Args:
            candidate: Candidate dictionary from CSV.

        Returns:
            Improvement prompt string.
        """
        reasons = candidate.get("improvement_reasons", "").split(",")
        reasons = [r.strip() for r in reasons if r.strip()]
        context = candidate.get("contextual_reasoning", "")

        # Build improvement instructions
        improvements = []
        for reason in reasons:
            if reason in self.IMPROVEMENT_INSTRUCTIONS:
                improvements.append(self.IMPROVEMENT_INSTRUCTIONS[reason])

        if not improvements:
            improvements.append("Enhance overall image quality")

        prompt = f"""Act as a professional photo retoucher. This family photo has technical issues that need fixing.

CONTEXT: {context}

PROBLEMS TO FIX:
{chr(10).join(f"- {imp}" for imp in improvements)}

Create a RETOUCHED VERSION with these edits applied:
- Remove motion blur and sharpen all faces to be crystal clear
- Reduce noise/grain significantly
- Color grade for warm, flattering skin tones
- Add subtle contrast enhancement
- Brighten dark areas appropriately

Keep the same people, expressions, and composition. Output a professionally retouched photo that looks like it was edited in Lightroom/Photoshop."""

        return prompt

    def improve_photo(
        self,
        candidate: Dict,
        candidates_folder: Path,
        improved_folder: Path
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Improve a single photo using Gemini.

        Args:
            candidate: Candidate dictionary from CSV.
            candidates_folder: Path to ImprovementCandidates folder.
            improved_folder: Path to Improved folder.

        Returns:
            Tuple of (success, status_message, improved_path or None).
        """
        filename = candidate.get("filename", "")
        source_path = candidates_folder / filename

        if not source_path.exists():
            return (False, f"Source file not found: {filename}", None)

        # Craft prompt
        prompt = self.craft_improvement_prompt(candidate)

        # Call Gemini image generation
        try:
            if self.client is None:
                return (False, "Gemini image client not initialized", None)

            # Generate improved image
            improved_bytes = self.client.generate_improved_image(source_path, prompt)

            if improved_bytes is None:
                return (False, "No image generated", None)

            # Save improved image
            improved_filename = f"{source_path.stem}_improved{source_path.suffix}"
            improved_path = improved_folder / improved_filename

            with open(improved_path, "wb") as f:
                f.write(improved_bytes)

            return (True, "completed", str(improved_path))

        except Exception as e:
            return (False, f"Error: {str(e)}", None)

    def process_batch(
        self,
        candidates: List[Dict],
        candidates_folder: Path,
        improved_folder: Path,
        show_progress: bool = True
    ) -> List[Dict]:
        """
        Process a batch of candidates in parallel.

        Args:
            candidates: List of approved candidate dictionaries.
            candidates_folder: Path to ImprovementCandidates folder.
            improved_folder: Path to Improved folder.
            show_progress: Whether to show progress bar.

        Returns:
            List of result dictionaries.
        """
        results = []

        # Process in parallel
        with ThreadPoolExecutor(max_workers=self.parallel_workers) as executor:
            futures = {
                executor.submit(
                    self.improve_photo,
                    candidate,
                    candidates_folder,
                    improved_folder
                ): candidate
                for candidate in candidates
            }

            iterator = as_completed(futures)
            if show_progress:
                iterator = tqdm(iterator, total=len(futures), desc="Improving photos")

            for future in iterator:
                candidate = futures[future]
                try:
                    success, status, improved_path = future.result()
                    results.append({
                        "filename": candidate.get("filename", ""),
                        "original_path": candidate.get("original_path", ""),
                        "improved_path": improved_path or "",
                        "improvement_reasons": candidate.get("improvement_reasons", ""),
                        "prompt_used": self.craft_improvement_prompt(candidate)[:200] + "...",
                        "status": status,
                        "error": "" if success else status,
                        "processed_at": datetime.now().isoformat()
                    })
                except Exception as e:
                    results.append({
                        "filename": candidate.get("filename", ""),
                        "original_path": candidate.get("original_path", ""),
                        "improved_path": "",
                        "improvement_reasons": candidate.get("improvement_reasons", ""),
                        "prompt_used": "",
                        "status": "failed",
                        "error": str(e),
                        "processed_at": datetime.now().isoformat()
                    })

        return results

    def save_results_csv(self, results: List[Dict], improved_folder: Path):
        """
        Save improvement results to CSV.

        Args:
            results: List of result dictionaries.
            improved_folder: Path to Improved folder.
        """
        if not results:
            return

        csv_path = improved_folder / "improvement_results.csv"

        fieldnames = [
            "filename", "original_path", "improved_path",
            "improvement_reasons", "prompt_used", "status",
            "error", "processed_at"
        ]

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print(f"\n📊 Results saved: {csv_path}")

    def print_summary(self, results: List[Dict]):
        """Print improvement summary."""
        completed = sum(1 for r in results if r["status"] == "completed")
        failed = sum(1 for r in results if r["status"] != "completed")
        total_cost = completed * self.cost_per_image

        print("\n" + "="*60)
        print("📊 IMPROVEMENT SUMMARY")
        print("="*60)
        print(f"   Completed: {completed}")
        print(f"   Failed:    {failed}")
        print(f"   Total:     {len(results)}")
        print(f"   Est. Cost: ${total_cost:.2f}")
        print("="*60)
