"""Training and feedback endpoints."""
from fastapi import APIRouter, HTTPException

from taster.api.models import FeedbackRequest, TrainingStats
from taster.api.services.training_service import TrainingService

router = APIRouter(prefix="/api/training", tags=["training"])

_service = TrainingService()


@router.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """Submit classification feedback/corrections."""
    results = []
    for item in request.feedback:
        result = _service.submit_feedback(
            file_path=item.file_path,
            correct_category=item.corrected_category,
            reasoning=item.notes or "",
        )
        results.append(result)
    return {"status": "received", "count": len(results)}


@router.post("/generate-profile")
async def generate_profile(profile_name: str):
    """Generate a taste profile from accumulated feedback."""
    try:
        profile = _service.generate_profile_from_feedback(profile_name)
        return {"status": "created", "profile": profile}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/stats", response_model=TrainingStats)
async def get_stats():
    """Get training data statistics."""
    stats = _service.get_stats()
    return TrainingStats(
        total_feedback_items=stats.get("total_feedback", 0),
        corrections_by_category=stats.get("by_category", {}),
    )
