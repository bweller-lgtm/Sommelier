"""MCP server exposing Taste Cloner tools for Claude Desktop integration."""
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# Lazy imports to avoid loading heavy dependencies at import time
_config = None
_profile_manager = None
_classification_service = None


def _get_config():
    global _config
    if _config is None:
        from ..core.config import load_config
        config_path = Path(os.environ.get("TASTE_CLONER_CONFIG", "config.yaml"))
        if config_path.exists():
            _config = load_config(config_path)
        else:
            from ..core.config import Config
            _config = Config()
    return _config


def _get_profile_manager():
    global _profile_manager
    if _profile_manager is None:
        from ..core.profiles import ProfileManager
        config = _get_config()
        _profile_manager = ProfileManager(config.profiles.profiles_dir)
    return _profile_manager


def _get_classification_service():
    global _classification_service
    if _classification_service is None:
        from ..api.services.classification_service import ClassificationService
        _classification_service = ClassificationService(_get_config())
    return _classification_service


def create_mcp_server():
    """Create and configure the MCP server with all Taste Cloner tools."""
    from mcp.server import Server
    from mcp.types import Tool, TextContent

    server = Server("taste-cloner")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="taste_cloner_list_profiles",
                description="List all available taste profiles with their names, descriptions, and media types.",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            Tool(
                name="taste_cloner_get_profile",
                description="Get the full details of a specific taste profile.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "profile_name": {
                            "type": "string",
                            "description": "Name of the profile to retrieve",
                        },
                    },
                    "required": ["profile_name"],
                },
            ),
            Tool(
                name="taste_cloner_create_profile",
                description="Create a new taste profile for classifying media.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Profile name (used as identifier)"},
                        "description": {"type": "string", "description": "Human-readable description"},
                        "media_types": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Media types: image, video, document, mixed",
                        },
                        "categories": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "description": {"type": "string"},
                                },
                                "required": ["name", "description"],
                            },
                            "description": "Output categories for classification",
                        },
                        "top_priorities": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Ranked list of what matters most",
                        },
                        "philosophy": {"type": "string", "description": "Overall philosophy statement"},
                    },
                    "required": ["name", "description", "media_types", "categories"],
                },
            ),
            Tool(
                name="taste_cloner_classify_folder",
                description="Classify all media files in a folder using a taste profile. Returns classification results.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "folder_path": {"type": "string", "description": "Path to folder containing files to classify"},
                        "profile_name": {"type": "string", "description": "Name of the taste profile to use"},
                        "dry_run": {"type": "boolean", "description": "If true, don't move files", "default": False},
                    },
                    "required": ["folder_path", "profile_name"],
                },
            ),
            Tool(
                name="taste_cloner_classify_files",
                description="Classify specific files using a taste profile.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_paths": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of file paths to classify",
                        },
                        "profile_name": {"type": "string", "description": "Name of the taste profile to use"},
                    },
                    "required": ["file_paths", "profile_name"],
                },
            ),
            Tool(
                name="taste_cloner_get_results",
                description="Get results from a previous classification run.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "job_id": {"type": "string", "description": "Job ID from a classification run"},
                    },
                    "required": ["job_id"],
                },
            ),
            Tool(
                name="taste_cloner_submit_feedback",
                description="Submit a correction/feedback on a classification result.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "Path to the file being corrected"},
                        "correct_category": {"type": "string", "description": "The correct category for this file"},
                        "reasoning": {"type": "string", "description": "Why this category is correct"},
                    },
                    "required": ["file_path", "correct_category"],
                },
            ),
            Tool(
                name="taste_cloner_generate_profile",
                description="AI-generate a taste profile from example files in folders.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "profile_name": {"type": "string", "description": "Name for the new profile"},
                        "good_examples_folder": {"type": "string", "description": "Folder with good/positive examples"},
                        "bad_examples_folder": {"type": "string", "description": "Folder with bad/negative examples (optional)"},
                    },
                    "required": ["profile_name", "good_examples_folder"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        try:
            result = _handle_tool(name, arguments)
            return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
        except Exception as e:
            return [TextContent(type="text", text=json.dumps({"error": str(e)}))]

    return server


def _handle_tool(name: str, arguments: dict) -> Any:
    """Dispatch tool calls to appropriate handlers."""
    pm = _get_profile_manager()

    if name == "taste_cloner_list_profiles":
        profiles = pm.list_profiles()
        return [
            {
                "name": p.name,
                "description": p.description,
                "media_types": p.media_types,
                "categories": [c.name for c in p.categories],
            }
            for p in profiles
        ]

    elif name == "taste_cloner_get_profile":
        profile = pm.load_profile(arguments["profile_name"])
        return profile.to_dict()

    elif name == "taste_cloner_create_profile":
        profile = pm.create_profile(
            name=arguments["name"],
            description=arguments["description"],
            media_types=arguments["media_types"],
            categories=arguments["categories"],
            top_priorities=arguments.get("top_priorities", []),
            philosophy=arguments.get("philosophy", ""),
        )
        return {"status": "created", "profile": profile.to_dict()}

    elif name == "taste_cloner_classify_folder":
        svc = _get_classification_service()
        job_id = svc.start_job(
            folder_path=arguments["folder_path"],
            profile_name=arguments["profile_name"],
            dry_run=arguments.get("dry_run", False),
        )

        # Wait for completion (MCP tools are synchronous from the user's perspective)
        import time
        for _ in range(600):  # 10 minute timeout
            status = svc.get_job_status(job_id)
            if status["status"] in ("completed", "failed"):
                break
            time.sleep(1)

        if status["status"] == "completed":
            results = svc.get_job_results(job_id)
            return {
                "status": "completed",
                "job_id": job_id,
                "stats": status.get("stats", {}),
                "result_count": len(results),
                "results": results[:50],  # Limit response size
            }
        elif status["status"] == "failed":
            return {"status": "failed", "job_id": job_id, "error": status.get("error", "Unknown error")}
        else:
            return {"status": "timeout", "job_id": job_id, "message": "Classification still running"}

    elif name == "taste_cloner_classify_files":
        # For individual files, classify them directly
        from ..core.config import load_config
        from ..core.cache import CacheManager
        from ..core.models import GeminiClient
        from ..classification.prompt_builder import PromptBuilder
        from ..classification.classifier import MediaClassifier
        from ..classification.routing import Router
        from ..core.file_utils import FileTypeRegistry

        config = _get_config()
        profile = pm.load_profile(arguments["profile_name"])

        cache_manager = CacheManager(
            config.paths.cache_root,
            ttl_days=config.caching.ttl_days,
            enabled=config.caching.enabled,
        )
        gemini_client = GeminiClient(
            model_name=config.model.name,
            max_retries=config.system.max_retries,
            retry_delay=config.system.retry_delay_seconds,
        )
        prompt_builder = PromptBuilder(config, profile=profile)
        classifier = MediaClassifier(config, gemini_client, prompt_builder, cache_manager, profile=profile)
        router = Router(config, gemini_client, profile=profile)

        results = []
        for fp in arguments["file_paths"]:
            path = Path(fp)
            if not path.exists():
                results.append({"file": fp, "error": "File not found"})
                continue

            if FileTypeRegistry.is_image(path):
                classification = classifier.classify_singleton(path)
                destination = router.route_singleton(classification)
            elif FileTypeRegistry.is_video(path):
                classification = classifier.classify_video(path)
                destination = router.route_video(classification)
            elif FileTypeRegistry.is_document(path):
                classification = classifier.classify_document(path)
                destination = router.route_document(classification)
            else:
                results.append({"file": fp, "error": "Unsupported file type"})
                continue

            results.append({
                "file": fp,
                "classification": classification.get("classification"),
                "confidence": classification.get("confidence"),
                "reasoning": classification.get("reasoning"),
                "destination": destination,
            })

        return results

    elif name == "taste_cloner_get_results":
        svc = _get_classification_service()
        results = svc.get_job_results(arguments["job_id"])
        return results

    elif name == "taste_cloner_submit_feedback":
        from ..api.services.training_service import TrainingService
        config = _get_config()
        ts = TrainingService(config.profiles.profiles_dir)
        return ts.submit_feedback(
            file_path=arguments["file_path"],
            correct_category=arguments["correct_category"],
            reasoning=arguments.get("reasoning", ""),
        )

    elif name == "taste_cloner_generate_profile":
        return {
            "status": "not_yet_implemented",
            "message": "AI profile generation from examples is planned for a future update.",
            "profile_name": arguments["profile_name"],
        }

    else:
        return {"error": f"Unknown tool: {name}"}
