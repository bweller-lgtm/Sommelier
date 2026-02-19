"""Pydantic models for the API layer."""
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Profile models ──────────────────────────────────────────────────────

class CategoryDefinitionModel(BaseModel):
    """A single output category."""
    name: str
    description: str
    color: Optional[str] = None


class PhotoProfileSettingsModel(BaseModel):
    """Photo/video-specific settings."""
    enable_burst_detection: bool = True
    enable_face_detection: bool = True
    contains_children_check: bool = True
    appropriateness_check: bool = True


class DocumentProfileSettingsModel(BaseModel):
    """Document-specific settings."""
    extract_text: bool = True
    extract_metadata: bool = True
    enable_similarity_grouping: bool = True
    similarity_threshold: float = 0.85
    max_pages_to_analyze: int = 10


class ProfileCreate(BaseModel):
    """Request body for creating a new profile."""
    name: str = Field(..., min_length=1, max_length=128)
    description: str = ""
    media_types: List[str] = Field(default_factory=lambda: ["image"])
    categories: List[CategoryDefinitionModel] = Field(default_factory=list)
    default_category: str = "Review"
    top_priorities: List[str] = Field(default_factory=list)
    positive_criteria: Dict[str, List[str]] = Field(default_factory=dict)
    negative_criteria: Dict[str, List[str]] = Field(default_factory=dict)
    specific_guidance: List[str] = Field(default_factory=list)
    philosophy: str = ""
    thresholds: Dict[str, float] = Field(default_factory=dict)
    photo_settings: Optional[PhotoProfileSettingsModel] = None
    document_settings: Optional[DocumentProfileSettingsModel] = None


class ProfileUpdate(BaseModel):
    """Request body for updating a profile (all fields optional)."""
    description: Optional[str] = None
    media_types: Optional[List[str]] = None
    categories: Optional[List[CategoryDefinitionModel]] = None
    default_category: Optional[str] = None
    top_priorities: Optional[List[str]] = None
    positive_criteria: Optional[Dict[str, List[str]]] = None
    negative_criteria: Optional[Dict[str, List[str]]] = None
    specific_guidance: Optional[List[str]] = None
    philosophy: Optional[str] = None
    thresholds: Optional[Dict[str, float]] = None
    photo_settings: Optional[PhotoProfileSettingsModel] = None
    document_settings: Optional[DocumentProfileSettingsModel] = None


class ProfileSummary(BaseModel):
    """Summary view of a profile (for listing)."""
    name: str
    description: str
    media_types: List[str]
    categories: List[str]
    created_at: str
    updated_at: str
    version: int


class ProfileDetail(BaseModel):
    """Full detail view of a profile."""
    name: str
    description: str
    media_types: List[str]
    categories: List[CategoryDefinitionModel]
    default_category: str
    top_priorities: List[str]
    positive_criteria: Dict[str, List[str]]
    negative_criteria: Dict[str, List[str]]
    specific_guidance: List[str]
    philosophy: str
    thresholds: Dict[str, float]
    photo_settings: Optional[PhotoProfileSettingsModel] = None
    document_settings: Optional[DocumentProfileSettingsModel] = None
    created_at: str
    updated_at: str
    version: int


# ── Classification / job models ─────────────────────────────────────────

class JobStatus(str, Enum):
    """Status of a classification job."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ClassifyFolderRequest(BaseModel):
    """Request body for classifying a local folder."""
    folder_path: str
    profile_name: str = "default-photos"
    dry_run: bool = False


class JobStatusResponse(BaseModel):
    """Status information for a classification job."""
    job_id: str
    status: JobStatus
    profile_name: str
    folder_path: Optional[str] = None
    progress: float = 0.0
    total_files: int = 0
    processed_files: int = 0
    created_at: str
    completed_at: Optional[str] = None
    error: Optional[str] = None


class ClassificationResultItem(BaseModel):
    """A single file's classification result."""
    file_path: str
    file_name: str
    category: str
    score: Optional[int] = None
    reasoning: Optional[str] = None
    burst_size: int = 1


class JobResultsResponse(BaseModel):
    """Full results for a completed classification job."""
    job_id: str
    status: JobStatus
    profile_name: str
    stats: Dict[str, int] = Field(default_factory=dict)
    results: List[ClassificationResultItem] = Field(default_factory=list)


# ── Results / export models ──────────────────────────────────────────────

class FileInCategory(BaseModel):
    """A file listed within a category."""
    file_path: str
    file_name: str
    score: Optional[int] = None
    reasoning: Optional[str] = None


class CategoryFiles(BaseModel):
    """Files belonging to a specific category."""
    category: str
    count: int
    files: List[FileInCategory]


# ── Training / feedback models ───────────────────────────────────────────

class FeedbackItem(BaseModel):
    """User feedback on a single classification."""
    file_path: str
    original_category: str
    corrected_category: str
    notes: Optional[str] = None


class FeedbackRequest(BaseModel):
    """Request body for submitting feedback."""
    job_id: Optional[str] = None
    profile_name: str = "default-photos"
    feedback: List[FeedbackItem]


class GenerateProfileRequest(BaseModel):
    """Request body for generating a profile from feedback."""
    profile_name: str = Field(..., min_length=1)
    base_profile: Optional[str] = None
    examples_folder: Optional[str] = None


class TrainingStats(BaseModel):
    """Statistics about collected training data."""
    total_feedback_items: int = 0
    profiles_with_feedback: List[str] = Field(default_factory=list)
    corrections_by_category: Dict[str, int] = Field(default_factory=dict)
