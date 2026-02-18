"""Platform-aware user configuration directories.

Resolves the user config directory for Taster:
  - Windows:  %APPDATA%\\taster
  - macOS:    ~/Library/Application Support/taster
  - Linux:    $XDG_CONFIG_HOME/taster  (default ~/.config/taster)
"""

import platform
from pathlib import Path


def get_config_dir() -> Path:
    """Return the platform-appropriate user config directory."""
    system = platform.system()
    if system == "Windows":
        base = Path.home() / "AppData" / "Roaming"
    elif system == "Darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        # Linux / other Unix — respect XDG
        import os
        xdg = os.environ.get("XDG_CONFIG_HOME")
        base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "taster"


def get_profiles_dir() -> Path:
    """Return the user-level profiles directory."""
    return get_config_dir() / "profiles"


def get_env_path() -> Path:
    """Return the user-level .env file path."""
    return get_config_dir() / ".env"


def ensure_dirs() -> Path:
    """Create the config directory and profiles subdirectory. Returns config dir."""
    config_dir = get_config_dir()
    config_dir.mkdir(parents=True, exist_ok=True)
    get_profiles_dir().mkdir(parents=True, exist_ok=True)
    return config_dir


def find_config(explicit_path: Path | None = None) -> Path | None:
    """Find the config file using resolution order.

    1. Explicit path (if provided — raises FileNotFoundError if missing)
    2. ./config.yaml (project-local, for development)
    3. User config dir config.yaml
    4. None (caller should use defaults)
    """
    if explicit_path is not None:
        if not explicit_path.exists():
            raise FileNotFoundError(f"Config file not found: {explicit_path}")
        return explicit_path

    local = Path("config.yaml")
    if local.exists():
        return local

    user_cfg = get_config_dir() / "config.yaml"
    if user_cfg.exists():
        return user_cfg

    return None


def find_profiles_dir() -> Path:
    """Find the profiles directory using resolution order.

    1. ./profiles/ (project-local, for development)
    2. User config dir profiles/
    """
    local = Path("profiles")
    if local.is_dir():
        return local
    return get_profiles_dir()
