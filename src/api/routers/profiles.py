"""Profile CRUD endpoints."""
from fastapi import APIRouter, HTTPException

from src.api.models import ProfileCreate, ProfileUpdate, ProfileSummary, ProfileDetail
from src.api.services.profile_service import ProfileService

router = APIRouter(prefix="/api/profiles", tags=["profiles"])

_service = ProfileService()


@router.get("/", response_model=list[ProfileSummary])
async def list_profiles():
    """List all available taste profiles."""
    return _service.list_profiles()


@router.get("/{name}")
async def get_profile(name: str):
    """Get full details of a taste profile."""
    try:
        return _service.get_profile(name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Profile '{name}' not found")


@router.post("/", status_code=201)
async def create_profile(data: ProfileCreate):
    """Create a new taste profile."""
    try:
        return _service.create_profile(data.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.put("/{name}")
async def update_profile(name: str, data: ProfileUpdate):
    """Update an existing taste profile."""
    try:
        updates = {k: v for k, v in data.model_dump().items() if v is not None}
        return _service.update_profile(name, updates)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Profile '{name}' not found")


@router.delete("/{name}")
async def delete_profile(name: str):
    """Delete a taste profile."""
    if _service.delete_profile(name):
        return {"status": "deleted", "name": name}
    raise HTTPException(status_code=404, detail=f"Profile '{name}' not found")
