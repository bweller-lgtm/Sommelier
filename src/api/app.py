"""FastAPI application factory for the Taste Cloner API."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src import __version__


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Taste Cloner API",
        description="Universal AI-powered media classification platform",
        version=__version__,
    )

    # CORS middleware for frontend access
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount routers
    from src.api.routers.profiles import router as profiles_router
    from src.api.routers.classify import router as classify_router
    from src.api.routers.results import router as results_router
    from src.api.routers.training import router as training_router

    app.include_router(profiles_router)
    app.include_router(classify_router)
    app.include_router(results_router)
    app.include_router(training_router)

    @app.get("/")
    async def root():
        return {
            "name": "Taste Cloner API",
            "version": __version__,
            "docs": "/docs",
        }

    return app
