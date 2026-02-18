"""Classification endpoints."""
from fastapi import APIRouter, HTTPException

from sommelier.api.models import ClassifyFolderRequest, JobStatusResponse
from sommelier.api.services.classification_service import ClassificationService
from sommelier.core.config import load_config

router = APIRouter(prefix="/api/classify", tags=["classify"])

_service = None


def _get_service() -> ClassificationService:
    global _service
    if _service is None:
        config = load_config()
        _service = ClassificationService(config)
    return _service


@router.post("/folder")
async def classify_folder(request: ClassifyFolderRequest):
    """Start a classification job on a local folder."""
    try:
        svc = _get_service()
        job_id = svc.start_job(
            folder_path=request.folder_path,
            profile_name=request.profile_name,
            dry_run=request.dry_run,
        )
        return {"job_id": job_id, "status": "started"}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{job_id}")
async def get_job_status(job_id: str):
    """Get the status of a classification job."""
    svc = _get_service()
    status = svc.get_job_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return status


@router.get("/{job_id}/results")
async def get_job_results(job_id: str):
    """Get the results of a completed classification job."""
    svc = _get_service()
    results = svc.get_job_results(job_id)
    if results is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return results
