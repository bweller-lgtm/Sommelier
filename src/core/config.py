"""Configuration management with YAML loading and validation."""
import os
import yaml
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field, asdict


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str = "gemini-3-flash-preview"
    max_output_tokens: int = 4096
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "laion2b_s34b_b79k"


@dataclass
class PathConfig:
    """Path configuration."""
    root: str = ""
    family_photos: str = ""
    private_photos: str = ""
    holding_cell: str = ""
    camera_roll: str = ""
    cache_root: str = ".taste_cache"
    taste_preferences: str = "taste_preferences.json"
    taste_preferences_generated: str = "taste_preferences_generated.json"

    def __post_init__(self):
        """Convert string paths to Path objects."""
        self.cache_root = Path(self.cache_root)
        self.taste_preferences = Path(self.taste_preferences)
        self.taste_preferences_generated = Path(self.taste_preferences_generated)


@dataclass
class FileTypeConfig:
    """File type configuration."""
    image_extensions: List[str] = field(default_factory=lambda: [
        ".jpg", ".jpeg", ".png", ".webp", ".heic", ".tif", ".tiff", ".bmp"
    ])
    video_extensions: List[str] = field(default_factory=lambda: [
        ".mp4", ".mov", ".avi", ".mkv", ".m4v", ".3gp", ".wmv", ".flv", ".webm"
    ])

    def __post_init__(self):
        """Convert to sets for fast lookup."""
        self.image_extensions = set(ext.lower() for ext in self.image_extensions)
        self.video_extensions = set(ext.lower() for ext in self.video_extensions)


@dataclass
class ClassificationConfig:
    """Classification configuration."""
    classify_videos: bool = True
    parallel_video_workers: int = 10
    share_threshold: float = 0.60
    review_threshold: float = 0.35
    burst_rank_consider: int = 2
    burst_rank_review: int = 4
    target_share_rate: float = 0.275
    enable_diversity_check: bool = True
    diversity_similarity_threshold: float = 0.95
    handle_blocked_safely: bool = True
    # Retry settings for failed classifications
    classification_retries: int = 2
    retry_delay_seconds: float = 2.0
    retry_on_errors: List[str] = field(default_factory=lambda: [
        "api_error", "invalid_response", "timeout", "rate_limit"
    ])

    def __post_init__(self):
        """Validate thresholds."""
        if not 0.0 <= self.share_threshold <= 1.0:
            raise ValueError(f"share_threshold must be between 0 and 1, got {self.share_threshold}")
        if not 0.0 <= self.review_threshold <= 1.0:
            raise ValueError(f"review_threshold must be between 0 and 1, got {self.review_threshold}")
        if self.review_threshold >= self.share_threshold:
            raise ValueError(f"review_threshold ({self.review_threshold}) must be < share_threshold ({self.share_threshold})")
        if self.parallel_video_workers < 1:
            raise ValueError(f"parallel_video_workers must be >= 1, got {self.parallel_video_workers}")
        if self.classification_retries < 0:
            raise ValueError(f"classification_retries must be >= 0, got {self.classification_retries}")


@dataclass
class BurstDetectionConfig:
    """Burst detection configuration."""
    enabled: bool = True
    time_window_seconds: int = 10
    embedding_similarity_threshold: float = 0.92
    min_burst_size: int = 2
    max_burst_size: int = 50
    enable_chunking: bool = True
    chunk_size: int = 10

    def __post_init__(self):
        """Validate parameters."""
        if not 0.0 <= self.embedding_similarity_threshold <= 1.0:
            raise ValueError(f"embedding_similarity_threshold must be between 0 and 1, got {self.embedding_similarity_threshold}")
        if self.min_burst_size < 2:
            raise ValueError(f"min_burst_size must be >= 2, got {self.min_burst_size}")
        if self.max_burst_size < self.min_burst_size:
            raise ValueError(f"max_burst_size must be >= min_burst_size")


@dataclass
class QualityConfig:
    """Quality scoring configuration."""
    sharpness_weight: float = 0.8
    brightness_weight: float = 0.2
    sharpness_threshold: float = 200.0
    quality_filter_threshold: float = 0.3
    enable_face_detection: bool = True
    face_detection_method: str = "mediapipe"
    max_faces_to_report: int = 10

    def __post_init__(self):
        """Validate parameters."""
        if abs(self.sharpness_weight + self.brightness_weight - 1.0) > 0.01:
            raise ValueError(f"sharpness_weight + brightness_weight must equal 1.0, got {self.sharpness_weight + self.brightness_weight}")
        if self.face_detection_method not in ["mediapipe", "face_recognition"]:
            raise ValueError(f"face_detection_method must be 'mediapipe' or 'face_recognition', got {self.face_detection_method}")
        if self.max_faces_to_report < 1:
            raise ValueError(f"max_faces_to_report must be >= 1, got {self.max_faces_to_report}")


@dataclass
class CachingConfig:
    """Caching configuration."""
    enabled: bool = True
    cache_embeddings: bool = True
    cache_quality_scores: bool = True
    cache_features: bool = True
    cache_faces: bool = True
    cache_gemini_responses: bool = True
    force_recompute: bool = False
    ttl_days: int = 365
    max_cache_size_gb: float = 10.0


@dataclass
class PerformanceConfig:
    """Performance configuration."""
    embedding_batch_size: int = 128
    quality_batch_size: int = 64
    classification_batch_size: int = 32
    use_multiprocessing: bool = True
    quality_workers: int = 4
    dedup_workers: int = 4
    device: str = "cuda"

    def __post_init__(self):
        """Validate parameters."""
        if self.embedding_batch_size < 1:
            raise ValueError(f"embedding_batch_size must be >= 1, got {self.embedding_batch_size}")
        if self.device not in ["cuda", "cpu"]:
            raise ValueError(f"device must be 'cuda' or 'cpu', got {self.device}")


@dataclass
class TrainingConfig:
    """Training configuration."""
    validation_split: float = 0.2
    random_seed: int = 1337
    max_negative_per_positive: int = 3
    balance_method: str = "undersample"
    within_cluster_weight: float = 2.0
    between_cluster_weight: float = 1.0
    use_camera_roll_negatives: bool = True
    date_range_days: int = 30


@dataclass
class CostConfig:
    """Cost estimation configuration."""
    per_1m_input_tokens: float = 0.50
    per_1m_output_tokens: float = 3.00
    per_photo: float = 0.0013
    per_burst_photo: float = 0.00043
    tokens_per_video_second: int = 300
    prompt_tokens_video: int = 1500
    output_tokens_video: int = 300


@dataclass
class SystemConfig:
    """System configuration."""
    dry_run: bool = False
    verbose: bool = True
    show_progress: bool = True
    checkpoint_every_n: int = 50
    continue_on_error: bool = True
    max_retries: int = 3
    retry_delay_seconds: float = 2.0


@dataclass
class PhotoImprovementConfig:
    """Photo improvement configuration for gray zone photos."""
    enabled: bool = True
    contextual_value_threshold: str = "high"  # "high" or "medium"
    min_issues_for_candidate: int = 1
    cost_per_image: float = 0.134  # Gemini 3 Pro Image pricing
    parallel_workers: int = 5
    max_retries: int = 2
    model_name: str = "gemini-3-pro-image-preview"
    max_output_tokens: int = 8192  # High token limit for image generation
    review_after_sort: bool = True  # Prompt to review candidates after sorting

    def __post_init__(self):
        """Validate parameters."""
        if self.contextual_value_threshold not in ["high", "medium"]:
            raise ValueError(f"contextual_value_threshold must be 'high' or 'medium', got {self.contextual_value_threshold}")
        if self.min_issues_for_candidate < 1:
            raise ValueError(f"min_issues_for_candidate must be >= 1, got {self.min_issues_for_candidate}")
        if self.cost_per_image < 0:
            raise ValueError(f"cost_per_image must be >= 0, got {self.cost_per_image}")
        if self.parallel_workers < 1:
            raise ValueError(f"parallel_workers must be >= 1, got {self.parallel_workers}")


@dataclass
class Config:
    """Master configuration object."""
    model: ModelConfig = field(default_factory=ModelConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    file_types: FileTypeConfig = field(default_factory=FileTypeConfig)
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)
    burst_detection: BurstDetectionConfig = field(default_factory=BurstDetectionConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    caching: CachingConfig = field(default_factory=CachingConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    photo_improvement: PhotoImprovementConfig = field(default_factory=PhotoImprovementConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create Config from dictionary."""
        return cls(
            model=ModelConfig(**data.get("model", {})),
            paths=PathConfig(**data.get("paths", {})),
            file_types=FileTypeConfig(**data.get("file_types", {})),
            classification=ClassificationConfig(**data.get("classification", {})),
            burst_detection=BurstDetectionConfig(**data.get("burst_detection", {})),
            quality=QualityConfig(**data.get("quality", {})),
            caching=CachingConfig(**data.get("caching", {})),
            performance=PerformanceConfig(**data.get("performance", {})),
            training=TrainingConfig(**data.get("training", {})),
            cost=CostConfig(**data.get("cost", {})),
            system=SystemConfig(**data.get("system", {})),
            photo_improvement=PhotoImprovementConfig(**data.get("photo_improvement", {})),
        )


def load_config(config_path: Optional[Path] = None) -> Config:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config.yaml file. If None, looks in current directory.

    Returns:
        Config object with validated settings.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If config validation fails.
    """
    if config_path is None:
        config_path = Path("config.yaml")

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return Config.from_dict(data)


def save_config(config: Config, config_path: Path):
    """
    Save configuration to YAML file.

    Args:
        config: Config object to save.
        config_path: Path to save config.yaml file.
    """
    # Convert config to dictionary using asdict
    def _convert_paths(obj):
        """Convert Path objects to strings for YAML serialization."""
        if isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: _convert_paths(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_convert_paths(item) for item in obj]
        elif isinstance(obj, set):
            return sorted(list(obj))  # Convert sets to sorted lists
        return obj

    # Convert dataclasses to dict
    data = {
        "model": asdict(config.model),
        "paths": asdict(config.paths),
        "file_types": asdict(config.file_types),
        "classification": asdict(config.classification),
        "burst_detection": asdict(config.burst_detection),
        "quality": asdict(config.quality),
        "caching": asdict(config.caching),
        "performance": asdict(config.performance),
        "training": asdict(config.training),
        "cost": asdict(config.cost),
        "system": asdict(config.system),
        "photo_improvement": asdict(config.photo_improvement),
    }

    # Convert Path objects and sets to YAML-serializable formats
    data = _convert_paths(data)

    # Write to file
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
