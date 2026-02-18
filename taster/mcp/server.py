"""MCP server exposing Taster tools for Claude Desktop integration."""
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

_SETUP_INSTRUCTIONS = (
    "To use Taster's AI features (classification, profile generation), "
    "you need an API key for at least one AI provider.\n\n"
    "Supported providers (pick one):\n"
    "  Gemini  (recommended — cheapest, native video/PDF)\n"
    "    Get a free key: https://aistudio.google.com/apikey\n"
    "    Env var: GEMINI_API_KEY\n\n"
    "  OpenAI  (GPT-4o / GPT-4.1)\n"
    "    Get a key: https://platform.openai.com/api-keys\n"
    "    Env var: OPENAI_API_KEY\n\n"
    "  Anthropic  (Claude)\n"
    "    Get a key: https://console.anthropic.com/settings/keys\n"
    "    Env var: ANTHROPIC_API_KEY\n\n"
    "Add the key to a .env file in the Taster folder, or to Claude Desktop's "
    "MCP config (claude_desktop_config.json) under taster > env.\n"
    "Then restart Claude Desktop.\n\n"
    "Profile browsing and manual profile creation work without an API key."
)


def _has_api_key() -> bool:
    """Check if any AI provider API key is configured."""
    return bool(
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
    )


def _require_api_key() -> dict | None:
    """Return a friendly error dict if no API key is set, or None if OK."""
    if not _has_api_key():
        return {
            "error": "No AI provider API key configured.",
            "setup": _SETUP_INSTRUCTIONS,
        }
    return None


def _get_config():
    global _config
    if _config is None:
        config_path = Path(os.environ.get("TASTER_CONFIG", "config.yaml"))
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


def create_mcp_server():
    """Create and configure the MCP server with all Taster tools."""
    from mcp.server import Server
    from mcp.types import Tool, TextContent, Prompt, PromptMessage, PromptArgument

    # Pre-initialize config and profile manager during startup
    print("[taster] Pre-loading config and profiles...", file=sys.stderr, flush=True)
    _get_config()
    _get_profile_manager()
    print("[taster] Ready.", file=sys.stderr, flush=True)

    server = Server("taster")

    # ── MCP Prompts ─────────────────────────────────────────────────────
    # These give Claude Desktop context about what Taster is and how
    # to guide users through common workflows.

    @server.list_prompts()
    async def list_prompts() -> list[Prompt]:
        return [
            Prompt(
                name="taster_getting_started",
                description="Introduction to Taster and how to use it. Start here if you're new.",
                arguments=[],
            ),
            Prompt(
                name="taster_create_profile_wizard",
                description="Step-by-step guide to create a new taste profile through conversation.",
                arguments=[
                    PromptArgument(
                        name="use_case",
                        description="Brief description of what you want to sort (e.g., 'family photos', 'resumes', 'product images')",
                        required=False,
                    ),
                ],
            ),
        ]

    @server.get_prompt()
    async def get_prompt(name: str, arguments: dict[str, str] | None = None) -> list[PromptMessage]:
        if name == "taster_getting_started":
            return [PromptMessage(
                role="user",
                content=TextContent(
                    type="text",
                    text=_GETTING_STARTED_PROMPT,
                ),
            )]
        elif name == "taster_create_profile_wizard":
            use_case = (arguments or {}).get("use_case", "")
            prompt = _PROFILE_WIZARD_PROMPT
            if use_case:
                prompt += f"\n\nThe user wants to sort: {use_case}"
            return [PromptMessage(
                role="user",
                content=TextContent(
                    type="text",
                    text=prompt,
                ),
            )]
        raise ValueError(f"Unknown prompt: {name}")

    # ── MCP Tools ───────────────────────────────────────────────────────

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="taster_status",
                description=(
                    "Check Taster setup status: API key, profiles, configuration. "
                    "Use this FIRST when a user is new or if any tool returns an API key error. "
                    "Returns what's configured, what's missing, and setup instructions."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            Tool(
                name="taster_list_profiles",
                description=(
                    "List all taste profiles available for classifying media. "
                    "Each profile defines categories (like Share/Storage/Ignore for photos) "
                    "and criteria for sorting files. Use this first to see what's available, "
                    "then use taster_classify_folder to sort files with a profile."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            Tool(
                name="taster_get_profile",
                description=(
                    "Get the full details of a taste profile, including its categories, "
                    "priorities, criteria, and philosophy. Use this to understand how a "
                    "profile makes classification decisions, or to review before classifying."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "profile_name": {
                            "type": "string",
                            "description": "Name of the profile (e.g., 'default-photos', 'default-documents')",
                        },
                    },
                    "required": ["profile_name"],
                },
            ),
            Tool(
                name="taster_create_profile",
                description=(
                    "Create a new taste profile with custom categories and criteria. "
                    "This is the structured version — you provide specific categories, "
                    "priorities, and criteria. For a simpler approach, use "
                    "taster_quick_profile which generates a profile from a "
                    "plain English description."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Profile name (lowercase with hyphens, e.g., 'wedding-photos')",
                        },
                        "description": {
                            "type": "string",
                            "description": "What this profile is for, in plain English",
                        },
                        "media_types": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "What kind of files: 'image', 'video', 'document', or 'mixed'",
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
                            "description": "Output categories (e.g., [{name: 'Keep', description: 'Worth keeping'}, ...])",
                        },
                        "top_priorities": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Ranked list of what matters most when classifying",
                        },
                        "positive_criteria": {
                            "type": "object",
                            "description": "What makes something good. Keys: 'must_have', 'highly_valued', 'bonus_points'. Values: arrays of strings.",
                        },
                        "negative_criteria": {
                            "type": "object",
                            "description": "What makes something bad. Keys: 'deal_breakers', 'negative_factors'. Values: arrays of strings.",
                        },
                        "specific_guidance": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Additional rules or guidance for the classifier",
                        },
                        "philosophy": {
                            "type": "string",
                            "description": "Overall philosophy statement guiding classification decisions",
                        },
                    },
                    "required": ["name", "description", "media_types", "categories"],
                },
            ),
            Tool(
                name="taster_update_profile",
                description=(
                    "Update an existing taste profile. You can change the description, "
                    "categories, priorities, criteria, guidance, or philosophy. "
                    "Only provide the fields you want to change — others stay the same. "
                    "Use taster_get_profile first to see the current values."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "profile_name": {
                            "type": "string",
                            "description": "Name of the profile to update",
                        },
                        "description": {
                            "type": "string",
                            "description": "New description for the profile",
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
                            "description": "New categories (replaces all existing categories)",
                        },
                        "top_priorities": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "New ranked priorities",
                        },
                        "positive_criteria": {
                            "type": "object",
                            "description": "New positive criteria (must_have, highly_valued, bonus_points)",
                        },
                        "negative_criteria": {
                            "type": "object",
                            "description": "New negative criteria (deal_breakers, negative_factors)",
                        },
                        "specific_guidance": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "New guidance rules",
                        },
                        "philosophy": {
                            "type": "string",
                            "description": "New philosophy statement",
                        },
                    },
                    "required": ["profile_name"],
                },
            ),
            Tool(
                name="taster_delete_profile",
                description=(
                    "Delete a taste profile. This is permanent and cannot be undone. "
                    "Use taster_list_profiles to see available profiles first."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "profile_name": {
                            "type": "string",
                            "description": "Name of the profile to delete",
                        },
                        "confirm": {
                            "type": "boolean",
                            "description": "Must be true to confirm deletion",
                        },
                    },
                    "required": ["profile_name", "confirm"],
                },
            ),
            Tool(
                name="taster_quick_profile",
                description=(
                    "Generate a complete taste profile from a plain English description. "
                    "Just describe what you want to sort and how — the AI will create "
                    "appropriate categories, criteria, and priorities. "
                    "Example: 'I want to sort my family vacation photos. Keep the ones "
                    "that show everyone having fun, discard blurry ones and duplicates.' "
                    "This is the easiest way to create a new profile."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "profile_name": {
                            "type": "string",
                            "description": "Name for the profile (lowercase with hyphens, e.g., 'vacation-photos')",
                        },
                        "description": {
                            "type": "string",
                            "description": "Plain English description of what you want to sort and how. Be as detailed as you like — the more context, the better the profile.",
                        },
                        "media_type": {
                            "type": "string",
                            "enum": ["image", "video", "document", "mixed"],
                            "description": "What kind of files you're sorting",
                            "default": "image",
                        },
                    },
                    "required": ["profile_name", "description"],
                },
            ),
            Tool(
                name="taster_classify_folder",
                description=(
                    "Classify all media files in a folder using a taste profile. "
                    "Each file is analyzed by AI and assigned to a category. "
                    "IMPORTANT: For any folder, call with estimate_only=true FIRST to show "
                    "the user a cost/time estimate and get confirmation before classifying. "
                    "For large folders (50+ files), use batch_size=30 to process in chunks "
                    "— the response includes next_offset so you can continue where you left off. "
                    "Use dry_run=true to classify but not move files."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "folder_path": {
                            "type": "string",
                            "description": "Full path to the folder (e.g., '/Users/me/Photos/Unsorted' or 'C:\\Users\\me\\Photos\\Unsorted')",
                        },
                        "profile_name": {
                            "type": "string",
                            "description": "Profile to use (run taster_list_profiles to see options)",
                        },
                        "estimate_only": {
                            "type": "boolean",
                            "description": "If true, scan folder and return cost/time estimate without classifying. Always do this first and show the user before proceeding.",
                            "default": False,
                        },
                        "dry_run": {
                            "type": "boolean",
                            "description": "If true, classify but don't move files. Good for previewing.",
                            "default": True,
                        },
                        "offset": {
                            "type": "integer",
                            "description": "Start from this file index. Use next_offset from a previous partial result to continue.",
                            "default": 0,
                        },
                        "batch_size": {
                            "type": "integer",
                            "description": "Max files to process. Use 30-50 for large folders. 0 = all files.",
                            "default": 0,
                        },
                    },
                    "required": ["folder_path", "profile_name"],
                },
            ),
            Tool(
                name="taster_classify_files",
                description=(
                    "Classify specific files (by path) using a taste profile. "
                    "Use this when you want to classify individual files rather than "
                    "a whole folder. Good for re-classifying files or testing a profile."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_paths": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Full paths to files to classify",
                        },
                        "profile_name": {
                            "type": "string",
                            "description": "Profile to use for classification",
                        },
                    },
                    "required": ["file_paths", "profile_name"],
                },
            ),
            Tool(
                name="taster_submit_feedback",
                description=(
                    "Correct a classification result. If a file was sorted into the wrong "
                    "category, submit the correction here. This feedback is stored and can "
                    "be used to improve profiles over time."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the file that was misclassified",
                        },
                        "correct_category": {
                            "type": "string",
                            "description": "The category it should have been sorted into",
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Why this category is correct (helps improve the profile)",
                        },
                    },
                    "required": ["file_path", "correct_category"],
                },
            ),
            Tool(
                name="taster_view_feedback",
                description=(
                    "View all submitted classification feedback and statistics. "
                    "Shows corrections, per-category breakdowns, and total counts. "
                    "Use this to review feedback before generating a profile from it, "
                    "or to check what corrections have been made."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            Tool(
                name="taster_generate_profile",
                description=(
                    "Generate a taste profile by analyzing example files. "
                    "Point it at a folder of 'good' examples (and optionally 'bad' examples) "
                    "and it will analyze them to create a profile that captures your taste. "
                    "Best with 5-20 examples in each folder."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "profile_name": {
                            "type": "string",
                            "description": "Name for the new profile",
                        },
                        "good_examples_folder": {
                            "type": "string",
                            "description": "Folder containing good/positive examples",
                        },
                        "bad_examples_folder": {
                            "type": "string",
                            "description": "Folder containing bad/negative examples (optional but recommended)",
                        },
                    },
                    "required": ["profile_name", "good_examples_folder"],
                },
            ),
            Tool(
                name="taster_refine_profile",
                description=(
                    "Apply post-classification corrections to refine an existing profile. "
                    "Unlike taster_submit_feedback (which just stores corrections), "
                    "this tool ACTS on corrections — it analyzes patterns in "
                    "misclassifications and updates the profile criteria accordingly. "
                    "This is the HIGHEST fidelity method for improving profiles."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "profile_name": {
                            "type": "string",
                            "description": "Name of the profile to refine",
                        },
                        "corrections": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "file_path": {
                                        "type": "string",
                                        "description": "Path to the misclassified file",
                                    },
                                    "original_category": {
                                        "type": "string",
                                        "description": "Category it was incorrectly sorted into",
                                    },
                                    "correct_category": {
                                        "type": "string",
                                        "description": "Category it should have been sorted into",
                                    },
                                    "reason": {
                                        "type": "string",
                                        "description": "Why this category is correct",
                                    },
                                },
                                "required": ["file_path", "original_category", "correct_category"],
                            },
                            "description": "Array of classification corrections with reasoning",
                        },
                    },
                    "required": ["profile_name", "corrections"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        import sys
        print(f"[taster] call_tool invoked: {name}", file=sys.stderr, flush=True)
        try:
            result = _handle_tool(name, arguments)
            print(f"[taster] call_tool success: {name}", file=sys.stderr, flush=True)
            return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
        except Exception as e:
            print(f"[taster] call_tool error: {name}: {e}", file=sys.stderr, flush=True)
            import traceback
            traceback.print_exc(file=sys.stderr)
            return [TextContent(type="text", text=json.dumps({"error": str(e)}))]

    return server


# ── Prompt templates ────────────────────────────────────────────────────

_GETTING_STARTED_PROMPT = """\
I have Taster connected. Help me understand what it can do and how to get started.

Taster is a media classification tool that sorts files (photos, videos, documents) \
into categories based on "taste profiles" — customizable rules that define what's good, \
what's bad, and how to sort things.

It supports multiple AI providers — Gemini (recommended, cheapest), OpenAI, and Anthropic. \
Just set an API key for any one of them and it auto-detects.

It comes with two built-in profiles:
- **default-photos**: Sorts family photos/videos into Share, Storage, Review, and Ignore
- **default-documents**: Sorts documents into Exemplary, Acceptable, Review, and Discard

I can also create custom profiles for any sorting task.

Please start by checking my setup status (taster_status) to see if everything \
is configured, then list my profiles and help me choose what to do next."""

_PROFILE_WIZARD_PROMPT = """\
Help me create a new taste profile for Taster. Walk me through it conversationally.

A taste profile needs:
1. **Name** — a short identifier (lowercase-with-hyphens)
2. **What I'm sorting** — photos, videos, documents, or a mix
3. **Categories** — the buckets files get sorted into (e.g., Keep/Maybe/Discard)
4. **What makes something good** — positive criteria
5. **What makes something bad** — negative criteria
6. **Overall philosophy** — the guiding principle

Ask me about each one conversationally. Once you have enough information, use the \
taster_quick_profile tool to generate the profile, or taster_create_profile \
if you want full control over every field.

Important: Don't ask all questions at once. Start with what I'm sorting and why, \
then build from there."""


# ── Tool dispatch ───────────────────────────────────────────────────────

def _handle_tool(name: str, arguments: dict) -> Any:
    """Dispatch tool calls to appropriate handlers."""
    import sys
    print(f"[taster] _handle_tool start: {name}", file=sys.stderr, flush=True)
    pm = _get_profile_manager()
    print(f"[taster] profile manager loaded", file=sys.stderr, flush=True)

    if name == "taster_status":
        return _handle_status(pm)
    elif name == "taster_list_profiles":
        return _handle_list_profiles(pm)
    elif name == "taster_get_profile":
        return _handle_get_profile(pm, arguments)
    elif name == "taster_create_profile":
        return _handle_create_profile(pm, arguments)
    elif name == "taster_update_profile":
        return _handle_update_profile(pm, arguments)
    elif name == "taster_delete_profile":
        return _handle_delete_profile(pm, arguments)
    elif name == "taster_quick_profile":
        return _handle_quick_profile(pm, arguments)
    elif name == "taster_classify_folder":
        return _handle_classify_folder(pm, arguments)
    elif name == "taster_classify_files":
        return _handle_classify_files(pm, arguments)
    elif name == "taster_submit_feedback":
        return _handle_submit_feedback(arguments)
    elif name == "taster_view_feedback":
        return _handle_view_feedback()
    elif name == "taster_generate_profile":
        return _handle_generate_profile(pm, arguments)
    elif name == "taster_refine_profile":
        return _handle_refine_profile(pm, arguments)
    else:
        return {"error": f"Unknown tool: {name}"}


def _handle_status(pm: ProfileManager) -> Any:
    """Check setup status: API keys, providers, profiles, config."""
    from ..core.provider_factory import detect_available_providers

    profiles = pm.list_profiles()
    has_key = _has_api_key()
    providers = detect_available_providers()

    status = {
        "providers": {
            name: ("configured" if avail else "not configured")
            for name, avail in providers.items()
        },
        "active_provider": next(
            (n for n in ["gemini", "openai", "anthropic"] if providers.get(n)),
            None,
        ),
        "profiles_count": len(profiles),
        "profiles": [p.name for p in profiles],
        "config_path": str(Path(os.environ.get("TASTER_CONFIG", "config.yaml")).resolve()),
        "profiles_dir": str(pm.profiles_dir.resolve()),
    }

    if has_key:
        status["ready"] = True
        configured = [n for n, a in providers.items() if a]
        status["message"] = (
            f"Taster is ready. Provider(s): {', '.join(configured)}. "
            f"{len(profiles)} profile(s) available."
        )
        status["available_features"] = [
            "list/view/create/update/delete profiles",
            "classify files and folders",
            "generate profiles from descriptions (AI)",
            "generate profiles from example files (AI)",
            "corrective profile refinement",
            "submit and review classification feedback",
        ]
    else:
        status["ready"] = False
        status["message"] = (
            "Taster is partially set up. You can browse and create profiles, "
            "but classification and AI profile generation require an API key."
        )
        status["setup"] = _SETUP_INSTRUCTIONS
        status["available_features"] = [
            "list/view profiles",
            "create profiles manually",
        ]
        status["requires_api_key"] = [
            "classify files and folders",
            "generate profiles from descriptions",
            "generate profiles from example files",
            "corrective profile refinement",
        ]

    return status


def _handle_list_profiles(pm: ProfileManager) -> Any:
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


def _handle_get_profile(pm: ProfileManager, arguments: dict) -> Any:
    profile_name = arguments["profile_name"]
    if not pm.profile_exists(profile_name):
        return {
            "error": f"Profile '{profile_name}' not found.",
            "hint": "Use taster_list_profiles to see available profiles.",
        }
    profile = pm.load_profile(profile_name)
    data = profile.to_dict()

    # Add human-readable summary
    cat_names = [c.name for c in profile.categories]
    data["summary"] = (
        f"{profile.description} "
        f"Sorts {', '.join(profile.media_types)} into {len(cat_names)} categories: "
        f"{', '.join(cat_names)}."
    )
    return data


def _handle_create_profile(pm: ProfileManager, arguments: dict) -> Any:
    profile = pm.create_profile(
        name=arguments["name"],
        description=arguments["description"],
        media_types=arguments["media_types"],
        categories=arguments["categories"],
        top_priorities=arguments.get("top_priorities", []),
        positive_criteria=arguments.get("positive_criteria", {}),
        negative_criteria=arguments.get("negative_criteria", {}),
        specific_guidance=arguments.get("specific_guidance", []),
        philosophy=arguments.get("philosophy", ""),
    )
    return {"status": "created", "profile": profile.to_dict()}


def _handle_update_profile(pm: ProfileManager, arguments: dict) -> Any:
    """Update an existing profile with provided fields."""
    profile_name = arguments["profile_name"]

    if not pm.profile_exists(profile_name):
        return {
            "error": f"Profile '{profile_name}' not found.",
            "hint": "Use taster_list_profiles to see available profiles.",
        }

    # Build kwargs from provided fields (excluding profile_name)
    updatable = [
        "description", "categories", "top_priorities",
        "positive_criteria", "negative_criteria",
        "specific_guidance", "philosophy",
    ]
    kwargs = {k: v for k, v in arguments.items() if k in updatable and v is not None}

    if not kwargs:
        return {
            "error": "No fields to update. Provide at least one field to change.",
            "updatable_fields": updatable,
        }

    profile = pm.update_profile(profile_name, **kwargs)
    return {
        "status": "updated",
        "message": f"Profile '{profile_name}' updated (v{profile.version}). Changed: {', '.join(kwargs.keys())}.",
        "profile": profile.to_dict(),
    }


def _handle_delete_profile(pm: ProfileManager, arguments: dict) -> Any:
    """Delete a taste profile permanently."""
    profile_name = arguments["profile_name"]

    if not arguments.get("confirm"):
        return {
            "error": "Deletion requires confirm=true. This action is permanent.",
            "hint": f"Call again with confirm=true to delete '{profile_name}'.",
        }

    if not pm.profile_exists(profile_name):
        return {
            "error": f"Profile '{profile_name}' not found.",
            "hint": "Use taster_list_profiles to see available profiles.",
        }

    pm.delete_profile(profile_name)
    return {
        "status": "deleted",
        "message": f"Profile '{profile_name}' has been permanently deleted.",
    }


def _handle_quick_profile(pm: ProfileManager, arguments: dict) -> Any:
    """Generate a complete taste profile from a natural language description."""
    api_err = _require_api_key()
    if api_err:
        return api_err

    config = _get_config()
    profile_name = arguments["profile_name"]
    description = arguments["description"]
    media_type = arguments.get("media_type", "image")

    # Check if profile already exists
    if pm.profile_exists(profile_name):
        return {
            "error": f"Profile '{profile_name}' already exists.",
            "hint": "Use taster_delete_profile to remove it first, "
                    "taster_update_profile to modify it, "
                    "or choose a different name.",
        }

    print(f"[taster] Generating profile '{profile_name}' from description...", file=sys.stderr, flush=True)

    from ..core.provider_factory import create_ai_client
    gemini_client = create_ai_client(config)

    prompt = f"""\
You are a taste profile generator for a media classification system.

The user wants to create a profile to sort their {media_type} files. Here's what they said:

"{description}"

Generate a complete taste profile as a JSON object with these fields:

{{
  "description": "A clear 1-sentence description of what this profile does",
  "media_types": ["{media_type}"],
  "categories": [
    {{"name": "CategoryName", "description": "When to put files here"}}
  ],
  "default_category": "The fallback category name",
  "top_priorities": ["Priority 1", "Priority 2", ...],
  "positive_criteria": {{
    "must_have": ["Required quality 1", ...],
    "highly_valued": ["Valued quality 1", ...],
    "bonus_points": ["Nice to have 1", ...]
  }},
  "negative_criteria": {{
    "deal_breakers": ["Automatic reject reason 1", ...],
    "negative_factors": ["Counts against 1", ...]
  }},
  "specific_guidance": ["Rule 1", "Rule 2", ...],
  "philosophy": "One sentence capturing the overall sorting philosophy"
}}

Rules:
- Create 3-5 categories that make sense for the use case
- Categories should be ordered from best to worst
- Include a middle/uncertain category for borderline cases
- Be specific and practical in criteria — vague criteria lead to bad classifications
- The philosophy should capture the user's intent in one clear sentence
- Only output valid JSON, no markdown or explanation"""

    result = gemini_client.generate_json(
        prompt=prompt,
        fallback=None,
    )

    if result is None:
        return {"error": "Failed to generate profile. Please try again or use taster_create_profile for manual creation."}

    # Create the profile
    profile = pm.create_profile(
        name=profile_name,
        description=result.get("description", description),
        media_types=result.get("media_types", [media_type]),
        categories=result.get("categories", []),
        default_category=result.get("default_category", "Review"),
        top_priorities=result.get("top_priorities", []),
        positive_criteria=result.get("positive_criteria", {}),
        negative_criteria=result.get("negative_criteria", {}),
        specific_guidance=result.get("specific_guidance", []),
        philosophy=result.get("philosophy", ""),
    )

    print(f"[taster] Profile '{profile_name}' generated with {len(profile.categories)} categories", file=sys.stderr, flush=True)

    return {
        "status": "created",
        "message": f"Profile '{profile_name}' created with {len(profile.categories)} categories: {', '.join(c.name for c in profile.categories)}",
        "profile": profile.to_dict(),
    }


def _estimate_folder_cost(images, videos, documents, profile_name) -> dict:
    """Scan folder and return cost/time estimate without classifying."""
    from ..pipelines.base import (
        _TOKENS_PER_IMAGE, _VIDEO_TOKENS_PER_SEC, _VIDEO_BITRATE_BPS,
        _PROMPT_TOKENS, _OUTPUT_TOKENS, _INPUT_PRICE_PER_M, _OUTPUT_PRICE_PER_M,
    )

    n_img, n_vid, n_doc = len(images), len(videos), len(documents)

    # Estimate video tokens from file sizes
    video_tokens = 0
    total_video_seconds = 0
    for v in videos:
        try:
            duration = max(v.stat().st_size * 8 / _VIDEO_BITRATE_BPS, 1)
        except OSError:
            duration = 60
        total_video_seconds += duration
        video_tokens += int(duration * _VIDEO_TOKENS_PER_SEC)

    image_tokens = n_img * _TOKENS_PER_IMAGE
    doc_tokens = n_doc * 2000

    # Estimate API calls (bursts share calls, rough ~60% burst rate)
    api_calls = max(int(n_img * 0.7) + n_vid + n_doc, 1)
    prompt_tokens = api_calls * _PROMPT_TOKENS
    output_tokens = api_calls * _OUTPUT_TOKENS

    total_input = video_tokens + image_tokens + doc_tokens + prompt_tokens
    input_cost = total_input / 1_000_000 * _INPUT_PRICE_PER_M
    output_cost = output_tokens / 1_000_000 * _OUTPUT_PRICE_PER_M
    total_cost = input_cost + output_cost

    # Estimate time: ~0.5s per image API call + ~2s per video (upload + process)
    est_minutes = (api_calls * 0.5 + n_vid * 60) / 60

    return {
        "status": "estimate",
        "profile": profile_name,
        "files": {"images": n_img, "videos": n_vid, "documents": n_doc, "total": n_img + n_vid + n_doc},
        "estimated_cost_usd": round(total_cost, 2),
        "estimated_minutes": round(est_minutes, 0),
        "estimated_video_minutes": round(total_video_seconds / 60, 1),
        "message": (
            f"Found {n_img} images, {n_vid} videos, {n_doc} documents. "
            f"Estimated cost: ~${total_cost:.2f}. "
            f"Estimated time: ~{est_minutes:.0f} minutes. "
            f"Call again without estimate_only to start classification."
        ),
    }


def _handle_classify_folder(pm: ProfileManager, arguments: dict) -> Any:
    """Classify media files in a folder with batch/pagination support."""
    from ..core.file_utils import FileTypeRegistry

    folder = Path(arguments["folder_path"])
    if not folder.is_dir():
        return {"error": f"Folder not found: {folder}"}

    print(f"[taster] classify_folder: {arguments}", file=sys.stderr, flush=True)

    # Discover files
    all_files = FileTypeRegistry.list_all_media(folder)
    images = all_files.get("images", [])
    videos = all_files.get("videos", [])
    documents = all_files.get("documents", [])
    all_media = (
        [(p, "image") for p in images]
        + [(p, "video") for p in videos]
        + [(p, "document") for p in documents]
    )
    total = len(all_media)

    if total == 0:
        return {
            "status": "completed",
            "total_files_in_folder": 0,
            "message": "No media files found in folder.",
        }

    # ── Estimate-only mode ─────────────────────────────────────────
    if arguments.get("estimate_only", False):
        return _estimate_folder_cost(images, videos, documents, arguments["profile_name"])

    # ── Full classification ────────────────────────────────────────
    api_err = _require_api_key()
    if api_err:
        return api_err

    import time as _time

    from ..core.cache import CacheManager
    from ..core.provider_factory import create_ai_client
    from ..classification.prompt_builder import PromptBuilder
    from ..classification.classifier import MediaClassifier
    from ..classification.routing import Router

    config = _get_config()
    profile = pm.load_profile(arguments["profile_name"])

    cache_manager = CacheManager(
        config.paths.cache_root,
        ttl_days=config.caching.ttl_days,
        enabled=config.caching.enabled,
    )
    ai_client = create_ai_client(config)
    prompt_builder = PromptBuilder(config, profile=profile)
    classifier = MediaClassifier(config, ai_client, prompt_builder, cache_manager, profile=profile)
    router = Router(config, ai_client, profile=profile)

    # Pagination: offset and batch_size let Claude call this repeatedly
    # for large folders without hitting the 4-minute MCP timeout.
    offset = int(arguments.get("offset", 0))
    batch_size = int(arguments.get("batch_size", 0)) or total
    batch = all_media[offset:offset + batch_size]

    print(f"[taster] Found {total} total files. Processing batch: offset={offset}, size={len(batch)}", file=sys.stderr, flush=True)

    dry_run = arguments.get("dry_run", True)
    results = []
    stats = {}
    errors = 0
    start_time = _time.time()
    TIME_LIMIT = 200  # seconds — return before MCP 4-min timeout

    for i, (file_path, media_type) in enumerate(batch):
        elapsed = _time.time() - start_time
        if elapsed > TIME_LIMIT:
            remaining = len(batch) - i
            print(f"[taster] Time limit reached ({elapsed:.0f}s). {remaining} files remaining.", file=sys.stderr, flush=True)
            return {
                "status": "partial",
                "dry_run": dry_run,
                "total_files_in_folder": total,
                "batch_offset": offset,
                "batch_requested": len(batch),
                "processed": len(results),
                "remaining_in_batch": remaining,
                "next_offset": offset + i,
                "stats": stats,
                "results": results,
                "message": f"Processed {len(results)} of {len(batch)} files before time limit. Call again with offset={offset + i} to continue.",
            }

        try:
            pct = ((offset + i + 1) / total) * 100
            print(f"[taster] [{offset+i+1}/{total}] ({pct:.0f}%) {file_path.name}", file=sys.stderr, flush=True)

            if media_type == "image":
                classification = classifier.classify_singleton(file_path)
                destination = router.route_singleton(classification)
            elif media_type == "video":
                classification = classifier.classify_video(file_path)
                destination = router.route_video(classification)
            else:
                classification = classifier.classify_document(file_path)
                destination = router.route_document(classification)

            results.append({
                "file": str(file_path),
                "name": file_path.name,
                "type": media_type,
                "classification": classification.get("classification"),
                "confidence": classification.get("confidence"),
                "reasoning": classification.get("reasoning", ""),
                "destination": destination,
            })
            stats[destination] = stats.get(destination, 0) + 1
        except Exception as e:
            print(f"[taster] ERROR classifying {file_path.name}: {e}", file=sys.stderr, flush=True)
            results.append({"file": str(file_path), "name": file_path.name, "error": str(e)})
            errors += 1

    elapsed = _time.time() - start_time
    next_offset = offset + len(batch)
    has_more = next_offset < total

    print(f"[taster] Batch complete in {elapsed:.0f}s. {len(results)} classified, {errors} errors. Stats: {stats}", file=sys.stderr, flush=True)

    # Build summary with percentages and top results
    processed_count = len(results)
    summary_lines = []
    if stats:
        for dest, count in sorted(stats.items(), key=lambda x: -x[1]):
            pct = (count / processed_count * 100) if processed_count else 0
            summary_lines.append(f"{dest}: {count} ({pct:.0f}%)")
    top_confident = sorted(
        [r for r in results if r.get("confidence") and "error" not in r],
        key=lambda x: x.get("confidence", 0),
        reverse=True,
    )[:5]

    return {
        "status": "completed" if not has_more else "partial",
        "dry_run": dry_run,
        "total_files_in_folder": total,
        "batch_offset": offset,
        "processed": processed_count,
        "errors": errors,
        "elapsed_seconds": round(elapsed, 1),
        "has_more": has_more,
        "next_offset": next_offset if has_more else None,
        "stats": stats,
        "summary": " | ".join(summary_lines) if summary_lines else "",
        "top_results": [
            {"name": r["name"], "classification": r["classification"], "confidence": r["confidence"], "reasoning": r.get("reasoning", "")}
            for r in top_confident
        ],
        "results": results,
        "message": f"Processed {processed_count} files in {elapsed:.0f}s." + (f" Call again with offset={next_offset} to continue ({total - next_offset} remaining)." if has_more else ""),
    }


def _handle_classify_files(pm: ProfileManager, arguments: dict) -> Any:
    """Classify specific files by path."""
    api_err = _require_api_key()
    if api_err:
        return api_err

    from ..core.cache import CacheManager
    from ..core.provider_factory import create_ai_client
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
    ai_client = create_ai_client(config)
    prompt_builder = PromptBuilder(config, profile=profile)
    classifier = MediaClassifier(config, ai_client, prompt_builder, cache_manager, profile=profile)
    router = Router(config, ai_client, profile=profile)

    results = []
    stats = {}
    errors = 0
    for fp in arguments["file_paths"]:
        path = Path(fp)
        if not path.exists():
            results.append({"file": fp, "name": Path(fp).name, "error": "File not found"})
            errors += 1
            continue

        try:
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
                results.append({"file": fp, "name": path.name, "error": "Unsupported file type"})
                errors += 1
                continue

            results.append({
                "file": fp,
                "name": path.name,
                "classification": classification.get("classification"),
                "confidence": classification.get("confidence"),
                "reasoning": classification.get("reasoning", ""),
                "destination": destination,
            })
            stats[destination] = stats.get(destination, 0) + 1
        except Exception as e:
            results.append({"file": fp, "name": path.name, "error": str(e)})
            errors += 1

    return {
        "status": "completed",
        "processed": len(results),
        "errors": errors,
        "stats": stats,
        "results": results,
        "message": f"Classified {len(results)} file(s). {errors} error(s).",
    }


def _handle_submit_feedback(arguments: dict) -> Any:
    """Submit a correction on a classification result."""
    from ..api.services.training_service import TrainingService
    config = _get_config()
    ts = TrainingService(config.profiles.profiles_dir)
    return ts.submit_feedback(
        file_path=arguments["file_path"],
        correct_category=arguments["correct_category"],
        reasoning=arguments.get("reasoning", ""),
    )


def _handle_view_feedback() -> Any:
    """View all submitted feedback and aggregate statistics."""
    from ..api.services.training_service import TrainingService
    config = _get_config()
    ts = TrainingService(config.profiles.profiles_dir)

    stats = ts.get_stats()
    feedback = ts._load_feedback()

    if not feedback:
        return {
            "total_feedback": 0,
            "message": "No feedback submitted yet. Use taster_submit_feedback to correct misclassifications.",
            "next_steps": [
                "Classify some files with taster_classify_folder",
                "Submit corrections with taster_submit_feedback",
                "Once you have enough feedback, generate a profile with taster_generate_profile",
            ],
        }

    return {
        "total_feedback": stats["total_feedback"],
        "by_category": stats["by_category"],
        "recent_feedback": feedback[-20:],  # last 20 entries
        "message": f"{stats['total_feedback']} feedback entries across {len(stats['by_category'])} categories.",
        "next_steps": [
            "Submit more corrections with taster_submit_feedback",
            "Generate a profile from this feedback with taster_generate_profile",
        ],
    }


def _handle_generate_profile(pm: ProfileManager, arguments: dict) -> Any:
    """Generate a taste profile by analyzing example files in folders."""
    api_err = _require_api_key()
    if api_err:
        return api_err

    from ..core.provider_factory import create_ai_client
    from ..core.file_utils import FileTypeRegistry

    config = _get_config()
    profile_name = arguments["profile_name"]
    good_folder = Path(arguments["good_examples_folder"])
    bad_folder = Path(arguments.get("bad_examples_folder", "")) if arguments.get("bad_examples_folder") else None

    if pm.profile_exists(profile_name):
        return {
            "error": f"Profile '{profile_name}' already exists.",
            "hint": "Use taster_delete_profile to remove it first, "
                    "taster_update_profile to modify it, "
                    "or choose a different name.",
        }

    if not good_folder.is_dir():
        return {"error": f"Good examples folder not found: {good_folder}"}

    if bad_folder and not bad_folder.is_dir():
        return {"error": f"Bad examples folder not found: {bad_folder}"}

    print(f"[taster] Generating profile from examples in {good_folder}...", file=sys.stderr, flush=True)

    gemini_client = create_ai_client(config)

    # Collect sample files (up to 10 good, 10 bad)
    good_files = FileTypeRegistry.list_all_media(good_folder)
    good_images = good_files.get("images", [])[:10]
    good_docs = good_files.get("documents", [])[:5]

    bad_images = []
    bad_docs = []
    if bad_folder:
        bad_files = FileTypeRegistry.list_all_media(bad_folder)
        bad_images = bad_files.get("images", [])[:10]
        bad_docs = bad_files.get("documents", [])[:5]

    # Detect media type from examples
    has_images = len(good_images) > 0 or len(bad_images) > 0
    has_docs = len(good_docs) > 0 or len(bad_docs) > 0
    if has_images and has_docs:
        media_type = "mixed"
    elif has_docs:
        media_type = "document"
    else:
        media_type = "image"

    # Analyze good examples with AI
    analysis_parts = []

    if good_images:
        sample_images = good_images[:5]
        prompt_parts = [
            "Analyze these GOOD example images. What do they have in common? "
            "What qualities make them good? Be specific about visual qualities, "
            "content, composition, and emotional impact."
        ]
        for img_path in sample_images:
            prompt_parts.append(img_path)
        prompt_parts.append(
            "Summarize what makes these examples GOOD. List specific, observable qualities."
        )
        good_analysis = gemini_client.generate(prompt_parts)
        analysis_parts.append(f"GOOD image examples analysis:\n{good_analysis.text}")

    if good_docs:
        from ..features.document_features import DocumentFeatureExtractor
        extractor = DocumentFeatureExtractor(config)
        doc_texts = []
        for doc_path in good_docs[:5]:
            text = extractor.extract_text(doc_path)
            if text:
                # Truncate long files for the prompt
                preview = text[:3000] + ("..." if len(text) > 3000 else "")
                doc_texts.append(f"--- {doc_path.name} ---\n{preview}")
        if doc_texts:
            docs_prompt = (
                "Analyze these GOOD example documents/files. What do they have in common? "
                "What qualities, patterns, conventions, or practices make them good? "
                "Be specific about structure, style, clarity, and any recurring patterns.\n\n"
                + "\n\n".join(doc_texts)
                + "\n\nSummarize what makes these examples GOOD. List specific, observable qualities and patterns."
            )
            good_doc_analysis = gemini_client.generate(docs_prompt)
            analysis_parts.append(f"GOOD document examples analysis:\n{good_doc_analysis.text}")

    if bad_images:
        sample_bad = bad_images[:5]
        prompt_parts = [
            "Analyze these BAD example images. What do they have in common? "
            "What qualities make them bad? Be specific."
        ]
        for img_path in sample_bad:
            prompt_parts.append(img_path)
        prompt_parts.append(
            "Summarize what makes these examples BAD. List specific, observable qualities."
        )
        bad_analysis = gemini_client.generate(prompt_parts)
        analysis_parts.append(f"BAD image examples analysis:\n{bad_analysis.text}")

    if bad_docs:
        from ..features.document_features import DocumentFeatureExtractor
        extractor = DocumentFeatureExtractor(config)
        doc_texts = []
        for doc_path in bad_docs[:5]:
            text = extractor.extract_text(doc_path)
            if text:
                preview = text[:3000] + ("..." if len(text) > 3000 else "")
                doc_texts.append(f"--- {doc_path.name} ---\n{preview}")
        if doc_texts:
            docs_prompt = (
                "Analyze these BAD example documents/files. What do they have in common? "
                "What qualities make them bad? Be specific about problems, anti-patterns, "
                "or missing qualities.\n\n"
                + "\n\n".join(doc_texts)
                + "\n\nSummarize what makes these examples BAD. List specific, observable issues."
            )
            bad_doc_analysis = gemini_client.generate(docs_prompt)
            analysis_parts.append(f"BAD document examples analysis:\n{bad_doc_analysis.text}")

    if not analysis_parts:
        return {"error": "No analyzable media files found in the example folders."}

    # Generate profile from analysis
    analysis_text = "\n\n".join(analysis_parts)

    generation_prompt = f"""\
Based on this analysis of example files, generate a taste profile for classifying {media_type} files.

{analysis_text}

Generate a JSON taste profile:

{{
  "description": "One sentence describing what this profile sorts",
  "media_types": ["{media_type}"],
  "categories": [
    {{"name": "CategoryName", "description": "When to put files here"}}
  ],
  "default_category": "Fallback category name",
  "top_priorities": ["Priority 1", ...],
  "positive_criteria": {{
    "must_have": [...],
    "highly_valued": [...],
    "bonus_points": [...]
  }},
  "negative_criteria": {{
    "deal_breakers": [...],
    "negative_factors": [...]
  }},
  "specific_guidance": ["Guidance 1", ...],
  "philosophy": "One sentence philosophy"
}}

Create 3-5 categories ordered best to worst. Be specific — use the actual qualities \
you observed in the examples, not generic platitudes. Only output valid JSON."""

    result = gemini_client.generate_json(
        prompt=generation_prompt,
        fallback=None,
    )

    if result is None:
        return {"error": "Failed to generate profile from examples."}

    profile = pm.create_profile(
        name=profile_name,
        description=result.get("description", f"Generated from examples in {good_folder.name}"),
        media_types=result.get("media_types", [media_type]),
        categories=result.get("categories", []),
        default_category=result.get("default_category", "Review"),
        top_priorities=result.get("top_priorities", []),
        positive_criteria=result.get("positive_criteria", {}),
        negative_criteria=result.get("negative_criteria", {}),
        specific_guidance=result.get("specific_guidance", []),
        philosophy=result.get("philosophy", ""),
    )

    return {
        "status": "created",
        "message": (
            f"Profile '{profile_name}' generated from "
            f"{len(good_images)} good images, {len(good_docs)} good docs, "
            f"{len(bad_images)} bad images, {len(bad_docs)} bad docs. "
            f"Categories: {', '.join(c.name for c in profile.categories)}"
        ),
        "analyzed": {
            "good_images": len(good_images),
            "bad_images": len(bad_images),
            "good_docs": len(good_docs),
            "bad_docs": len(bad_docs),
        },
        "profile": profile.to_dict(),
    }


def _handle_refine_profile(pm: ProfileManager, arguments: dict) -> Any:
    """Refine a profile based on post-classification corrections."""
    api_err = _require_api_key()
    if api_err:
        return api_err

    from ..core.provider_factory import create_ai_client
    from ..training.synthesizer import ProfileSynthesizer

    config = _get_config()
    profile_name = arguments["profile_name"]
    corrections = arguments["corrections"]

    if not pm.profile_exists(profile_name):
        return {
            "error": f"Profile '{profile_name}' not found.",
            "hint": "Use taster_list_profiles to see available profiles.",
        }

    if not corrections:
        return {"error": "No corrections provided."}

    ai_client = create_ai_client(config)
    synthesizer = ProfileSynthesizer(ai_client, pm)

    updated = synthesizer.refine_from_corrections(profile_name, corrections)

    changes = getattr(updated, "_refinement_changes", [])

    return {
        "status": "refined",
        "corrections_applied": len(corrections),
        "changes_made": changes,
        "profile": updated.to_dict(),
        "message": (
            f"Profile '{profile_name}' refined based on {len(corrections)} corrections. "
            f"{len(changes)} criteria changes applied."
        ),
    }
