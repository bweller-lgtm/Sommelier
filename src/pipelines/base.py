"""Abstract base class for classification pipelines."""
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict

from ..core.config import Config
from ..core.profiles import TasteProfile


@dataclass
class ClassificationResult:
    """Result from running a classification pipeline."""
    results: List[Dict[str, Any]] = field(default_factory=list)
    stats: Dict[str, int] = field(default_factory=dict)

    def print_summary(self, media_label: str = "items"):
        """Print a human-readable summary."""
        print(f"\n{'='*60}")
        print(f"CLASSIFICATION SUMMARY ({media_label})")
        print(f"{'='*60}")
        if self.stats:
            for destination, count in sorted(self.stats.items()):
                print(f"   {destination:25s}: {count:4d}")
            print(f"   {'TOTAL':25s}: {sum(self.stats.values()):4d}")
        else:
            print("   No items processed.")
        print(f"{'='*60}")

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
