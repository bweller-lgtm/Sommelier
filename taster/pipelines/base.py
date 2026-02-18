"""Abstract base class for classification pipelines."""
import json
import shutil
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict

from ..core.config import Config
from ..core.profiles import TasteProfile

# Per-item cost estimates (Gemini Flash)
_COST_ESTIMATES = {
    "image": 0.0013,
    "video": 0.011,  # per minute, assume ~1 min avg
    "document": 0.005,
}


@dataclass
class ClassificationResult:
    """Result from running a classification pipeline."""
    results: List[Dict[str, Any]] = field(default_factory=list)
    stats: Dict[str, int] = field(default_factory=dict)
    elapsed_seconds: float = 0.0

    def print_summary(self, media_label: str = "items"):
        """Print a human-readable summary."""
        total = sum(self.stats.values()) if self.stats else 0
        print(f"\n{'='*60}")
        print(f"CLASSIFICATION SUMMARY ({media_label})")
        print(f"{'='*60}")
        if self.stats:
            for destination, count in sorted(self.stats.items()):
                pct = (count / total * 100) if total else 0
                print(f"   {destination:25s}: {count:4d}  ({pct:.1f}%)")
            print(f"   {'':25s}  {'----':>4s}")
            print(f"   {'TOTAL':25s}: {total:4d}")
        else:
            print("   No items processed.")
        if self.elapsed_seconds > 0:
            minutes = int(self.elapsed_seconds // 60)
            seconds = int(self.elapsed_seconds % 60)
            print(f"\n   Time: {minutes}m {seconds}s")
        cost = self._estimate_cost()
        if cost > 0:
            print(f"   Est. cost: ~${cost:.2f} (Gemini Flash)")
        top = self._top_confident(5)
        if top:
            print(f"\n   Top {len(top)} highest-confidence:")
            for i, item in enumerate(top, 1):
                name = Path(item["path"]).name if isinstance(item["path"], (str, Path)) else str(item["path"])
                cls = item.get("classification", {})
                conf = cls.get("confidence", 0)
                reason = cls.get("reasoning", "")
                if len(reason) > 60:
                    reason = reason[:57] + "..."
                print(f"   {i}. {name} ({conf:.2f}) - {reason}")
        print(f"{'='*60}")

    def _estimate_cost(self) -> float:
        """Estimate API cost based on file types."""
        cost = 0.0
        for r in self.results:
            # Check for burst photos (shared prompt = cheaper)
            burst_size = r.get("burst_size", 1)
            if burst_size > 1:
                cost += _COST_ESTIMATES.get("image", 0) / 3  # shared prompt discount
            else:
                # Detect type from path extension
                path = Path(r["path"]) if isinstance(r["path"], (str, Path)) else r["path"]
                ext = path.suffix.lower()
                if ext in {".jpg", ".jpeg", ".png", ".gif", ".webp", ".heic", ".tif", ".tiff", ".bmp"}:
                    cost += _COST_ESTIMATES["image"]
                elif ext in {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}:
                    cost += _COST_ESTIMATES["video"]
                else:
                    cost += _COST_ESTIMATES["document"]
        return cost

    def _top_confident(self, n: int = 5) -> List[Dict[str, Any]]:
        """Return the top N highest-confidence results."""
        scored = []
        for r in self.results:
            cls = r.get("classification", {})
            conf = cls.get("confidence", 0)
            if conf and not cls.get("is_error_fallback", False):
                scored.append(r)
        scored.sort(key=lambda x: x.get("classification", {}).get("confidence", 0), reverse=True)
        return scored[:n]

    def generate_summary_report(self, report_dir: Path, provider_name: str = "gemini"):
        """Save a human-readable summary report to Reports/summary.txt."""
        total = sum(self.stats.values()) if self.stats else 0
        cost = self._estimate_cost()
        top = self._top_confident(5)

        lines = []
        lines.append(f"Classification Complete")
        lines.append("=" * 48)
        if self.stats:
            for dest, count in sorted(self.stats.items()):
                pct = (count / total * 100) if total else 0
                lines.append(f"  {dest:25s}: {count:4d}  ({pct:.1f}%)")
            lines.append(f"  {'':25s}  {'----':>4s}")
            lines.append(f"  {'Total':25s}: {total:4d}")
        lines.append("=" * 48)
        if self.elapsed_seconds > 0:
            minutes = int(self.elapsed_seconds // 60)
            seconds = int(self.elapsed_seconds % 60)
            lines.append(f"  Time: {minutes}m {seconds}s")
        if cost > 0:
            lines.append(f"  Est. cost: ~${cost:.2f} ({provider_name.title()})")
        lines.append("")

        # Error summary
        errors = [r for r in self.results if r.get("classification", {}).get("is_error_fallback")]
        if errors:
            lines.append(f"  Errors: {len(errors)} files fell back to default category")
            lines.append("")

        if top:
            lines.append(f"  Top {len(top)} highest-confidence:")
            for i, item in enumerate(top, 1):
                name = Path(item["path"]).name if isinstance(item["path"], (str, Path)) else str(item["path"])
                cls = item.get("classification", {})
                conf = cls.get("confidence", 0)
                dest = item.get("destination", "")
                reason = cls.get("reasoning", "")
                if len(reason) > 70:
                    reason = reason[:67] + "..."
                lines.append(f"  {i}. {name}")
                lines.append(f"     {dest} ({conf:.2f}) - {reason}")
            lines.append("")

        report_dir.mkdir(parents=True, exist_ok=True)
        summary_path = report_dir / "summary.txt"
        summary_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"Summary saved: {summary_path}")
        return "\n".join(lines)

    def merge(self, other: "ClassificationResult") -> "ClassificationResult":
        """Merge another result into this one."""
        combined_stats = defaultdict(int)
        for k, v in self.stats.items():
            combined_stats[k] += v
        for k, v in other.stats.items():
            combined_stats[k] += v
        return ClassificationResult(
            results=self.results + other.results,
            stats=dict(combined_stats),
        )


class ClassificationPipeline(ABC):
    """Abstract base for all classification pipelines."""

    def __init__(self, config: Config, profile: Optional[TasteProfile] = None):
        self.config = config
        self.profile = profile

    @abstractmethod
    def collect_files(self, folder: Path) -> List[Path]:
        """Collect files to process."""

    @abstractmethod
    def extract_features(self, files: List[Path]) -> Dict[Path, Any]:
        """Extract features from files."""

    @abstractmethod
    def group_files(self, files: List[Path], features: Dict) -> List[List[Path]]:
        """Group related files (bursts, similar docs)."""

    @abstractmethod
    def classify(self, groups: List[List[Path]], features: Dict) -> List[Dict[str, Any]]:
        """Classify all files."""

    @abstractmethod
    def route(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Route results to output categories."""

    def run(
        self,
        input_folder: Path,
        output_folder: Path,
        dry_run: bool = False
    ) -> ClassificationResult:
        """Execute the full pipeline."""
        files = self.collect_files(input_folder)
        if not files:
            return ClassificationResult()

        features = self.extract_features(files)
        groups = self.group_files(files, features)
        results = self.classify(groups, features)
        routed = self.route(results)

        if not dry_run:
            self.move_files(routed, output_folder)

        stats = self.compute_stats(routed)
        return ClassificationResult(results=routed, stats=stats)

    def move_files(self, results: List[Dict[str, Any]], output_folder: Path):
        """Copy files to their destination subfolders."""
        for result in results:
            source = result["path"]
            destination = result["destination"]
            dst_dir = output_folder / destination
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst_path = dst_dir / source.name
            try:
                shutil.copy2(source, dst_path)
            except Exception as e:
                print(f"Warning: Error copying {source.name}: {e}")

    def compute_stats(self, results: List[Dict[str, Any]]) -> Dict[str, int]:
        """Compute statistics from routed results."""
        stats = defaultdict(int)
        for result in results:
            stats[result["destination"]] += 1
        return dict(stats)
