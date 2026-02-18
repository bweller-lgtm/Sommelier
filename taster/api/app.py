"""FastAPI application factory for the Taster API."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from taster import __version__


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Taster API",
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
    from taster.api.routers.profiles import router as profiles_router
    from taster.api.routers.classify import router as classify_router
    from taster.api.routers.results import router as results_router
    from taster.api.routers.training import router as training_router

    app.include_router(profiles_router)
    app.include_router(classify_router)
    app.include_router(results_router)
    app.include_router(training_router)

    @app.get("/")
    async def root():
        return {
            "name": "Taster API",
            "version": __version__,
            "docs": "/docs",
        }

    return app
