"""Results and export endpoints."""
import csv
import io
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from sommelier.api.services.classification_service import ClassificationService
from sommelier.core.config import load_config

router = APIRouter(prefix="/api/results", tags=["results"])

_service = None


def _get_service() -> ClassificationService:
    global _service
    if _service is None:
        config = load_config()
        _service = ClassificationService(config)
    return _service


@router.get("/{job_id}")
async def get_results(job_id: str):
    """Get detailed results for a classification job."""
    svc = _get_service()
    results = svc.get_job_results(job_id)
    if results is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return results


@router.get("/{job_id}/export")
async def export_results(job_id: str, format: str = "csv"):
    """Export classification results as CSV."""
    svc = _get_service()
    results = svc.get_job_results(job_id)
    if results is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    if format == "csv":
        output = io.StringIO()
        if results:
            writer = csv.DictWriter(output, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        content = output.getvalue()
        return StreamingResponse(
            iter([content]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=results_{job_id}.csv"},
        )

    return results


@router.get("/{job_id}/files/{category}")
async def list_files_in_category(job_id: str, category: str):
    """List files classified into a specific category."""
    svc = _get_service()
    results = svc.get_job_results(job_id)
    if results is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    filtered = [r for r in results if r.get("destination") == category]
    return {"category": category, "count": len(filtered), "files": filtered}
