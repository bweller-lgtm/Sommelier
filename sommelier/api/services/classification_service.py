"""Service layer for running classification jobs.

Manages background classification pipelines, tracking their progress and
results through an in-memory job store keyed by UUID.
"""
import logging
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from ...core.config import Config
from ...core.cache import CacheManager
from ...core.provider_factory import create_ai_client
from ...core.profiles import ProfileManager
from ...pipelines import MixedPipeline

logger = logging.getLogger(__name__)

# Valid job statuses
STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"


class ClassificationService:
    """Manages classification jobs that run pipelines in background threads."""

    def __init__(self, config: Config):
        """
        Initialize ClassificationService.

        Args:
            config: Application configuration object.
        """
        self.config = config

        # Initialize shared infrastructure
        self.cache_manager = CacheManager(
            cache_root=config.paths.cache_root,
            ttl_days=config.caching.ttl_days,
            enabled=config.caching.enabled,
            max_size_gb=config.caching.max_cache_size_gb,
        )
        self.gemini_client = create_ai_client(config)
        self.profile_manager = ProfileManager(
            profiles_dir=config.profiles.profiles_dir,
        )

        # In-memory job store: job_id -> job dict
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_job(
        self,
        folder_path: str,
        profile_name: str,
        dry_run: bool = False,
    ) -> str:
        """
        Start a classification job in a background thread.

        Args:
            folder_path: Path to the folder containing media files.
            profile_name: Name of the taste profile to use.
            dry_run: If True, classify but do not move/copy files.

        Returns:
            A unique job ID string (UUID4).

        Raises:
            FileNotFoundError: If *folder_path* does not exist or *profile_name*
                is not found.
        """
        input_folder = Path(folder_path)
        if not input_folder.is_dir():
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        # Validate profile exists
        profile = self.profile_manager.load_profile(profile_name)

        job_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()

        job = {
            "job_id": job_id,
            "status": STATUS_PENDING,
            "folder_path": str(input_folder),
            "profile_name": profile_name,
            "dry_run": dry_run,
            "progress": 0.0,
            "message": "Job queued",
            "results": [],
            "stats": {},
            "created_at": now,
            "started_at": None,
            "completed_at": None,
            "error": None,
        }

        with self._lock:
            self._jobs[job_id] = job

        # Launch the pipeline in a daemon thread
        thread = threading.Thread(
            target=self._run_job,
            args=(job_id, input_folder, profile, dry_run),
            daemon=True,
            name=f"classification-{job_id[:8]}",
        )
        thread.start()
        logger.info("Started classification job %s on %s", job_id, folder_path)

        return job_id

    def get_job_status(self, job_id: str) -> dict:
        """
        Return current status for a job.

        Args:
            job_id: The UUID of the job.

        Returns:
            Dictionary with ``job_id``, ``status``, ``progress`` (0-100),
            ``message``, ``stats``, timing fields, and ``error`` (if any).

        Raises:
            KeyError: If the job_id is unknown.
        """
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(f"Unknown job: {job_id}")

        return {
            "job_id": job["job_id"],
            "status": job["status"],
            "progress": job["progress"],
            "message": job["message"],
            "folder_path": job["folder_path"],
            "profile_name": job["profile_name"],
            "dry_run": job["dry_run"],
            "stats": job["stats"],
            "created_at": job["created_at"],
            "started_at": job["started_at"],
            "completed_at": job["completed_at"],
            "error": job["error"],
        }

    def get_job_results(self, job_id: str) -> List[dict]:
        """
        Return the per-file classification results for a completed job.

        Each result dict contains at minimum ``path``, ``destination``, and
        ``classification``.

        Args:
            job_id: The UUID of the job.

        Returns:
            List of result dictionaries.

        Raises:
            KeyError: If the job_id is unknown.
        """
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(f"Unknown job: {job_id}")

        return job["results"]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _update_job(self, job_id: str, **fields) -> None:
        """Thread-safe update of job fields."""
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].update(fields)

    def _run_job(
        self,
        job_id: str,
        input_folder: Path,
        profile,
        dry_run: bool,
    ) -> None:
        """Execute the classification pipeline (runs in a background thread)."""
        self._update_job(
            job_id,
            status=STATUS_RUNNING,
            started_at=datetime.utcnow().isoformat(),
            message="Initializing pipeline",
            progress=5.0,
        )

        try:
            # Determine output folder (sibling of input with _sorted suffix)
            output_folder = input_folder.parent / f"{input_folder.name}_sorted"

            pipeline = MixedPipeline(
                config=self.config,
                profile=profile,
                cache_manager=self.cache_manager,
                gemini_client=self.gemini_client,
            )

            self._update_job(
                job_id,
                message="Running classification",
                progress=10.0,
            )

            result = pipeline.run(
                input_folder=input_folder,
                output_folder=output_folder,
                dry_run=dry_run,
            )

            # Serialize results (convert Path objects to strings)
            serialized_results = []
            for r in result.results:
                serialized = {}
                for k, v in r.items():
                    if isinstance(v, Path):
                        serialized[k] = str(v)
                    else:
                        serialized[k] = v
                serialized_results.append(serialized)

            self._update_job(
                job_id,
                status=STATUS_COMPLETED,
                progress=100.0,
                message="Classification complete",
                results=serialized_results,
                stats=result.stats,
                completed_at=datetime.utcnow().isoformat(),
            )
            logger.info(
                "Job %s completed: %d items, stats=%s",
                job_id,
                len(result.results),
                result.stats,
            )

        except Exception as exc:
            logger.exception("Job %s failed", job_id)
            self._update_job(
                job_id,
                status=STATUS_FAILED,
                message=f"Failed: {exc}",
                error=str(exc),
                completed_at=datetime.utcnow().isoformat(),
            )
