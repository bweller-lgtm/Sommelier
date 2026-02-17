#!/usr/bin/env python3
"""Launch the Taste Cloner API server."""
import argparse
import uvicorn
from src.api.app import create_app


def main():
    parser = argparse.ArgumentParser(description="Taste Cloner API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
