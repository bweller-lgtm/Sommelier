"""MCP server exposing Taste Cloner tools for Claude Desktop integration."""
import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# Eager imports — do all heavy loading at module import time so the MCP
# server is ready before it starts accepting tool calls from Claude Desktop.
from ..core.config import load_config, Config
from ..core.profiles import ProfileManager

_config = None
_profile_manager = None
_classification_service = None


def _get_config():
    global _config
    if _config is None:
        config_path = Path(os.environ.get("TASTE_CLONER_CONFIG", "config.yaml"))
        if config_path.exists():
            _config = load_config(config_path)
        else:
            _config = Config()
    return _config


def _get_profile_manager():
    global _profile_manager
    if _profile_manager is None:
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

    # Pre-initialize config and profile manager during startup
    print("[taste-cloner] Pre-loading config and profiles...", file=sys.stderr, flush=True)
    _get_config()
    _get_profile_manager()
    print("[taste-cloner] Ready.", file=sys.stderr, flush=True)

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
        import sys
        print(f"[taste-cloner] call_tool invoked: {name}", file=sys.stderr, flush=True)
        try:
            result = _handle_tool(name, arguments)
            print(f"[taste-cloner] call_tool success: {name}", file=sys.stderr, flush=True)
            return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
        except Exception as e:
            print(f"[taste-cloner] call_tool error: {name}: {e}", file=sys.stderr, flush=True)
            return [TextContent(type="text", text=json.dumps({"error": str(e)}))]

    return server


def _handle_tool(name: str, arguments: dict) -> Any:
    """Dispatch tool calls to appropriate handlers."""
    import sys
    print(f"[taste-cloner] _handle_tool start: {name}", file=sys.stderr, flush=True)
    pm = _get_profile_manager()
    print(f"[taste-cloner] profile manager loaded", file=sys.stderr, flush=True)

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
        # Lightweight classification — uses Gemini directly, no CLIP/torch.
        # Classifies each photo individually (no burst detection).
        print(f"[taste-cloner] classify_folder: {arguments}", file=sys.stderr, flush=True)

        from ..core.cache import CacheManager
        from ..core.models import GeminiClient
        from ..classification.prompt_builder import PromptBuilder
        from ..classification.classifier import MediaClassifier
        from ..classification.routing import Router
        from ..core.file_utils import FileTypeRegistry

        config = _get_config()
        profile = pm.load_profile(arguments["profile_name"])
        folder = Path(arguments["folder_path"])

        if not folder.is_dir():
            return {"error": f"Folder not found: {folder}"}

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

        # Discover files
        all_files = FileTypeRegistry.list_all_media(folder)
        images = all_files.get("images", [])
        videos = all_files.get("videos", [])
        documents = all_files.get("documents", [])
        total = len(images) + len(videos) + len(documents)

        print(f"[taste-cloner] Found {len(images)} images, {len(videos)} videos, {len(documents)} documents", file=sys.stderr, flush=True)

        dry_run = arguments.get("dry_run", False)
        results = []
        stats = {}

        for i, img_path in enumerate(images):
            try:
                print(f"[taste-cloner] Classifying {i+1}/{total}: {img_path.name}", file=sys.stderr, flush=True)
                classification = classifier.classify_singleton(img_path)
                destination = router.route_singleton(classification)
                results.append({
                    "file": str(img_path),
                    "name": img_path.name,
                    "type": "image",
                    "classification": classification.get("classification"),
                    "confidence": classification.get("confidence"),
                    "reasoning": classification.get("reasoning", ""),
                    "destination": destination,
                })
                stats[destination] = stats.get(destination, 0) + 1
            except Exception as e:
                print(f"[taste-cloner] ERROR classifying {img_path.name}: {e}", file=sys.stderr, flush=True)
                results.append({"file": str(img_path), "name": img_path.name, "error": str(e)})

        for vid_path in videos:
            try:
                print(f"[taste-cloner] Classifying video: {vid_path.name}", file=sys.stderr, flush=True)
                classification = classifier.classify_video(vid_path)
                destination = router.route_video(classification)
                results.append({
                    "file": str(vid_path),
                    "name": vid_path.name,
                    "type": "video",
                    "classification": classification.get("classification"),
                    "confidence": classification.get("confidence"),
                    "reasoning": classification.get("reasoning", ""),
                    "destination": destination,
                })
                stats[destination] = stats.get(destination, 0) + 1
            except Exception as e:
                print(f"[taste-cloner] ERROR classifying {vid_path.name}: {e}", file=sys.stderr, flush=True)
                results.append({"file": str(vid_path), "name": vid_path.name, "error": str(e)})

        print(f"[taste-cloner] Classification complete. Stats: {stats}", file=sys.stderr, flush=True)

        return {
            "status": "completed",
            "dry_run": dry_run,
            "total_files": total,
            "stats": stats,
            "results": results[:50],
        }

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
        return {"error": "Job-based results are only available via the REST API. Use taste_cloner_classify_folder for direct results."}

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
