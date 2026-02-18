"""Tests for sommelier.dirs — platform-aware config directory resolution."""

import os
import pytest
from pathlib import Path
from unittest.mock import patch

from sommelier.dirs import (
    get_config_dir,
    get_profiles_dir,
    get_env_path,
    ensure_dirs,
    find_config,
    find_profiles_dir,
)


# ── get_config_dir ──────────────────────────────────────────────────


class TestGetConfigDir:
    def test_windows(self):
        with patch("sommelier.dirs.platform.system", return_value="Windows"):
            result = get_config_dir()
            assert result == Path.home() / "AppData" / "Roaming" / "sommelier"

    def test_macos(self):
        with patch("sommelier.dirs.platform.system", return_value="Darwin"):
            result = get_config_dir()
            assert result == Path.home() / "Library" / "Application Support" / "sommelier"

    def test_linux_default(self):
        env = dict(os.environ)
        env.pop("XDG_CONFIG_HOME", None)
        # Keep HOME so Path.home() works
        with (
            patch("sommelier.dirs.platform.system", return_value="Linux"),
            patch.dict(os.environ, env, clear=True),
        ):
            result = get_config_dir()
            assert result == Path.home() / ".config" / "sommelier"

    def test_linux_xdg_set(self, tmp_path):
        with (
            patch("sommelier.dirs.platform.system", return_value="Linux"),
            patch.dict(os.environ, {"XDG_CONFIG_HOME": str(tmp_path)}),
        ):
            result = get_config_dir()
            assert result == tmp_path / "sommelier"

    def test_linux_xdg_empty_falls_back(self):
        with (
            patch("sommelier.dirs.platform.system", return_value="Linux"),
            patch.dict(os.environ, {"XDG_CONFIG_HOME": ""}, clear=False),
        ):
            # Empty string → falsy → fallback to ~/.config
            result = get_config_dir()
            assert result == Path.home() / ".config" / "sommelier"

    def test_returns_path_object(self):
        result = get_config_dir()
        assert isinstance(result, Path)


# ── get_profiles_dir / get_env_path ─────────────────────────────────


class TestHelperPaths:
    def test_profiles_dir_is_subdir_of_config(self):
        with patch("sommelier.dirs.platform.system", return_value="Windows"):
            assert get_profiles_dir() == get_config_dir() / "profiles"

    def test_env_path_is_in_config_dir(self):
        with patch("sommelier.dirs.platform.system", return_value="Windows"):
            assert get_env_path() == get_config_dir() / ".env"


# ── ensure_dirs ─────────────────────────────────────────────────────


class TestEnsureDirs:
    def test_creates_config_dir(self, tmp_path):
        config_dir = tmp_path / "cfg" / "sommelier"
        with patch("sommelier.dirs.get_config_dir", return_value=config_dir):
            with patch("sommelier.dirs.get_profiles_dir", return_value=config_dir / "profiles"):
                result = ensure_dirs()
                assert result == config_dir
                assert config_dir.is_dir()
                assert (config_dir / "profiles").is_dir()

    def test_idempotent(self, tmp_path):
        config_dir = tmp_path / "sommelier"
        config_dir.mkdir()
        (config_dir / "profiles").mkdir()
        with patch("sommelier.dirs.get_config_dir", return_value=config_dir):
            with patch("sommelier.dirs.get_profiles_dir", return_value=config_dir / "profiles"):
                result = ensure_dirs()
                assert result == config_dir


# ── find_config ─────────────────────────────────────────────────────


class TestFindConfig:
    def test_explicit_path_exists(self, tmp_path):
        cfg = tmp_path / "my.yaml"
        cfg.write_text("model: {}")
        assert find_config(cfg) == cfg

    def test_explicit_path_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            find_config(tmp_path / "nonexistent.yaml")

    def test_local_config_found(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        local = tmp_path / "config.yaml"
        local.write_text("model: {}")
        result = find_config()
        assert result is not None
        assert result.resolve() == local.resolve()

    def test_user_dir_fallback(self, tmp_path, monkeypatch):
        # No local config.yaml
        monkeypatch.chdir(tmp_path)
        user_cfg = tmp_path / "user_config" / "config.yaml"
        user_cfg.parent.mkdir()
        user_cfg.write_text("model: {}")
        with patch("sommelier.dirs.get_config_dir", return_value=tmp_path / "user_config"):
            assert find_config() == user_cfg

    def test_returns_none_when_nothing_found(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with patch("sommelier.dirs.get_config_dir", return_value=tmp_path / "empty"):
            assert find_config() is None

    def test_local_takes_priority_over_user_dir(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        local = tmp_path / "config.yaml"
        local.write_text("local: true")
        user_cfg = tmp_path / "user" / "config.yaml"
        user_cfg.parent.mkdir()
        user_cfg.write_text("user: true")
        with patch("sommelier.dirs.get_config_dir", return_value=tmp_path / "user"):
            result = find_config()
            assert result is not None
            assert result.resolve() == local.resolve()


# ── find_profiles_dir ───────────────────────────────────────────────


class TestFindProfilesDir:
    def test_local_profiles_found(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "profiles").mkdir()
        result = find_profiles_dir()
        assert result == Path("profiles")

    def test_falls_back_to_user_dir(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        # No local profiles/ directory
        user_profiles = tmp_path / "user" / "profiles"
        with patch("sommelier.dirs.get_profiles_dir", return_value=user_profiles):
            result = find_profiles_dir()
            assert result == user_profiles
