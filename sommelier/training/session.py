"""Training session state and persistence for pairwise training."""

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class PairwiseComparison:
    """A single pairwise comparison between two photos."""

    photo_a: str  # absolute path
    photo_b: str  # absolute path
    choice: str  # "left" | "right" | "both" | "neither"
    reason: str  # free-text reasoning (the critical learning signal)
    comparison_type: str  # "within_burst" | "between_burst"
    weight: float  # 2.0 within-burst, 1.0 between-burst
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class GallerySelection:
    """A burst gallery selection where user picks keepers."""

    photos: list[str]  # all photo paths in burst
    selected_indices: list[int]  # indices of keepers
    reason: str  # free-text reasoning
    weight: float = 1.5
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class TrainingSession:
    """Persistent pairwise training session."""

    session_id: str
    profile_name: str
    folder_path: str
    status: str = "active"  # "active" | "completed"
    created_at: str = ""
    updated_at: str = ""
    bursts: list[list[str]] = field(default_factory=list)  # burst groups
    singletons: list[str] = field(default_factory=list)  # non-burst photos
    total_files: int = 0
    media_types: list[str] = field(default_factory=list)
    pairwise: list[PairwiseComparison] = field(default_factory=list)
    gallery: list[GallerySelection] = field(default_factory=list)
    current_comparison: Optional[dict] = None  # tracks what was last served
    comparisons_served: int = 0

    def __post_init__(self):
        now = datetime.now().isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now

    @classmethod
    def create(
        cls,
        profile_name: str,
        folder_path: str,
        bursts: list[list[str]],
        singletons: list[str],
        media_types: list[str] | None = None,
    ) -> "TrainingSession":
        """Create a new training session."""
        session_id = uuid.uuid4().hex[:12]
        total = sum(len(b) for b in bursts) + len(singletons)
        return cls(
            session_id=session_id,
            profile_name=profile_name,
            folder_path=folder_path,
            bursts=bursts,
            singletons=singletons,
            total_files=total,
            media_types=media_types or ["image"],
        )

    def save(self, profiles_dir: Path) -> None:
        """Save session to disk."""
        self.updated_at = datetime.now().isoformat()
        path = profiles_dir / f"training-session-{self.session_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self._to_dict()
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, session_id: str, profiles_dir: Path) -> "TrainingSession":
        """Load a session from disk."""
        path = profiles_dir / f"training-session-{session_id}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"Training session '{session_id}' not found at {path}"
            )
        with open(path) as f:
            data = json.load(f)
        return cls._from_dict(data)

    @classmethod
    def list_sessions(cls, profiles_dir: Path) -> list[dict]:
        """List all training sessions with summary info."""
        sessions = []
        for path in sorted(profiles_dir.glob("training-session-*.json")):
            try:
                with open(path) as f:
                    data = json.load(f)
                sessions.append({
                    "session_id": data["session_id"],
                    "profile_name": data["profile_name"],
                    "status": data["status"],
                    "total_files": data.get("total_files", data.get("total_photos", 0)),
                    "media_types": data.get("media_types", ["image"]),
                    "total_labeled": (
                        len(data.get("pairwise", []))
                        + len(data.get("gallery", []))
                    ),
                    "created_at": data["created_at"],
                    "updated_at": data["updated_at"],
                })
            except (json.JSONDecodeError, KeyError):
                continue
        return sessions

    def add_pairwise(
        self,
        photo_a: str,
        photo_b: str,
        choice: str,
        reason: str,
        comparison_type: str,
    ) -> PairwiseComparison:
        """Record a pairwise comparison."""
        weight = 2.0 if comparison_type == "within_burst" else 1.0
        comp = PairwiseComparison(
            photo_a=photo_a,
            photo_b=photo_b,
            choice=choice,
            reason=reason,
            comparison_type=comparison_type,
            weight=weight,
        )
        self.pairwise.append(comp)
        return comp

    def add_gallery(
        self,
        photos: list[str],
        selected_indices: list[int],
        reason: str,
    ) -> GallerySelection:
        """Record a gallery/burst selection."""
        sel = GallerySelection(
            photos=photos,
            selected_indices=selected_indices,
            reason=reason,
        )
        self.gallery.append(sel)
        return sel

    def get_labeled_photos(self) -> set[str]:
        """Get set of all photos that have been labeled."""
        labeled = set()
        for comp in self.pairwise:
            labeled.add(comp.photo_a)
            labeled.add(comp.photo_b)
        for sel in self.gallery:
            labeled.update(sel.photos)
        return labeled

    def get_stats(self) -> dict:
        """Get session statistics."""
        within = sum(
            1 for c in self.pairwise if c.comparison_type == "within_burst"
        )
        between = sum(
            1 for c in self.pairwise if c.comparison_type == "between_burst"
        )
        choices = {
            "left": sum(1 for c in self.pairwise if c.choice == "left"),
            "right": sum(1 for c in self.pairwise if c.choice == "right"),
            "both": sum(1 for c in self.pairwise if c.choice == "both"),
            "neither": sum(1 for c in self.pairwise if c.choice == "neither"),
        }
        labeled = self.get_labeled_photos()
        total_labeled = len(self.pairwise) + len(self.gallery)

        return {
            "session_id": self.session_id,
            "profile_name": self.profile_name,
            "status": self.status,
            "total_files": self.total_files,
            "media_types": self.media_types,
            "total_bursts": len(self.bursts),
            "total_singletons": len(self.singletons),
            "pairwise_count": len(self.pairwise),
            "within_burst": within,
            "between_burst": between,
            "gallery_count": len(self.gallery),
            "total_labeled": total_labeled,
            "unique_files_labeled": len(labeled),
            "coverage_pct": round(
                len(labeled) / self.total_files * 100, 1
            ) if self.total_files > 0 else 0,
            "choices": choices,
            "comparisons_served": self.comparisons_served,
            "ready_to_synthesize": total_labeled >= 15,
        }

    def _to_dict(self) -> dict:
        """Serialize to dict for JSON storage."""
        return {
            "session_id": self.session_id,
            "profile_name": self.profile_name,
            "folder_path": self.folder_path,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "bursts": self.bursts,
            "singletons": self.singletons,
            "total_files": self.total_files,
            "media_types": self.media_types,
            "pairwise": [asdict(c) for c in self.pairwise],
            "gallery": [asdict(g) for g in self.gallery],
            "current_comparison": self.current_comparison,
            "comparisons_served": self.comparisons_served,
        }

    @classmethod
    def _from_dict(cls, data: dict) -> "TrainingSession":
        """Deserialize from dict."""
        pairwise = [
            PairwiseComparison(**c) for c in data.get("pairwise", [])
        ]
        gallery = [
            GallerySelection(**g) for g in data.get("gallery", [])
        ]
        return cls(
            session_id=data["session_id"],
            profile_name=data["profile_name"],
            folder_path=data["folder_path"],
            status=data.get("status", "active"),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            bursts=data.get("bursts", []),
            singletons=data.get("singletons", []),
            total_files=data.get("total_files", data.get("total_photos", 0)),
            media_types=data.get("media_types", ["image"]),
            pairwise=pairwise,
            gallery=gallery,
            current_comparison=data.get("current_comparison"),
            comparisons_served=data.get("comparisons_served", 0),
        )
