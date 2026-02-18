# Changelog

## 3.0.0 (2026-02-17)

**Sommelier is now a distributable PyPI package.**

### Breaking Changes

- **Package renamed:** `src/` is now `sommelier/`. All imports change from `from src.*` to `from sommelier.*`.
- **`requirements.txt` removed.** Dependencies are now declared in `pyproject.toml` with optional groups.
- **`setup_sommelier.py` removed.** Replaced by `sommelier init`.

### Added

- **`pip install sommelier`** -- proper Python package with `pyproject.toml` (hatchling build).
- **CLI entry point** -- `sommelier classify`, `sommelier train`, `sommelier serve`, `sommelier status`, `sommelier init`.
- **Platform-aware config directory** -- `%APPDATA%\sommelier` (Windows), `~/Library/Application Support/sommelier` (macOS), `$XDG_CONFIG_HOME/sommelier` (Linux).
- **Config resolution order** -- explicit path > `./config.yaml` > user config dir > built-in defaults. Sommelier works out of the box with zero config.
- **Optional dependency groups** -- `pip install sommelier[gemini]`, `sommelier[ml]`, `sommelier[all]`, etc. Clear error messages when an optional dep is missing.
- **`sommelier/compat.py`** -- `require()` helper for graceful optional dependency handling.
- **`sommelier/dirs.py`** -- platform-aware directory resolution for config, profiles, and .env.
- **New tests** -- `test_dirs.py`, `test_cli.py`, `test_compat.py`, `test_config_resolution.py`.

### Changed

- `load_config()` returns defaults when no config file is found (previously raised `FileNotFoundError`).
- Standalone scripts (`taste_classify.py`, `mcp_server.py`, etc.) still work as thin wrappers for repo-based usage.
- README updated for cross-platform paths and `pip install` workflow.

### Migration

If you were using Sommelier from a git clone:

1. Pull the latest code.
2. Run `pip install -e ".[gemini]"` (or whichever provider you use).
3. Replace `py -3.12 taste_classify.py` with `sommelier classify`.
4. Replace `py -3.12 setup_sommelier.py` with `sommelier init`.
5. If you import from Sommelier in your own code, change `from src.*` to `from sommelier.*`.
