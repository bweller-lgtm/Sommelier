"""Tests for taster.cli — CLI entry point."""

import pytest
import sys
from unittest.mock import patch, MagicMock
from pathlib import Path

from taster.cli import _build_parser, main


class TestParser:
    """Test argument parsing."""

    def test_help_flag(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "taster" in captured.out

    def test_version_flag(self, capsys):
        from taster import __version__
        with pytest.raises(SystemExit) as exc_info:
            main(["--version"])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert __version__ in captured.out

    def test_no_command_shows_help(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "classify" in captured.out
        assert "train" in captured.out

    def test_unknown_command(self, capsys):
        with pytest.raises(SystemExit):
            main(["nonexistent"])


class TestClassifyArgs:
    """Test classify subcommand argument parsing."""

    def test_classify_requires_folder(self):
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["classify"])

    def test_classify_parses_folder(self):
        parser = _build_parser()
        args = parser.parse_args(["classify", "/some/folder"])
        assert args.command == "classify"
        assert args.folder == "/some/folder"

    def test_classify_config_flag(self):
        parser = _build_parser()
        args = parser.parse_args(["classify", "/f", "--config", "my.yaml"])
        assert args.config == "my.yaml"

    def test_classify_profile_flag(self):
        parser = _build_parser()
        args = parser.parse_args(["classify", "/f", "--profile", "photos"])
        assert args.profile == "photos"

    def test_classify_provider_flag(self):
        parser = _build_parser()
        args = parser.parse_args(["classify", "/f", "--provider", "gemini"])
        assert args.provider == "gemini"

    def test_classify_invalid_provider(self):
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["classify", "/f", "--provider", "invalid"])

    def test_classify_dry_run(self):
        parser = _build_parser()
        args = parser.parse_args(["classify", "/f", "--dry-run"])
        assert args.dry_run is True

    def test_classify_output_flag(self):
        parser = _build_parser()
        args = parser.parse_args(["classify", "/f", "--output", "/out"])
        assert args.output == "/out"

    def test_classify_no_classify_videos(self):
        parser = _build_parser()
        args = parser.parse_args(["classify", "/f", "--no-classify-videos"])
        assert args.no_classify_videos is True

    def test_classify_parallel_videos(self):
        parser = _build_parser()
        args = parser.parse_args(["classify", "/f", "--parallel-videos", "4"])
        assert args.parallel_videos == 4

    def test_classify_cache_dir(self):
        parser = _build_parser()
        args = parser.parse_args(["classify", "/f", "--cache-dir", "/cache"])
        assert args.cache_dir == "/cache"

    def test_classify_provider_local(self):
        parser = _build_parser()
        args = parser.parse_args(["classify", "/f", "--provider", "local"])
        assert args.provider == "local"


class TestTrainArgs:
    """Test train subcommand parsing."""

    def test_train_requires_folder(self):
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["train"])

    def test_train_parses_folder(self):
        parser = _build_parser()
        args = parser.parse_args(["train", "/photos"])
        assert args.command == "train"
        assert args.folder == "/photos"


class TestServeArgs:
    def test_serve_no_args(self):
        parser = _build_parser()
        args = parser.parse_args(["serve"])
        assert args.command == "serve"


class TestStatusArgs:
    def test_status_no_args(self):
        parser = _build_parser()
        args = parser.parse_args(["status"])
        assert args.command == "status"


class TestStatusCommand:
    """Test that status command runs without error."""

    def test_status_runs(self, capsys):
        with patch.dict("os.environ", {}, clear=False):
            main(["status"])
        captured = capsys.readouterr()
        assert "Taster v" in captured.out
        assert "Config directory" in captured.out


class TestInitCommand:
    """Test init command (mocked I/O)."""

    def test_init_creates_dirs(self, tmp_path):
        config_dir = tmp_path / "taster"
        with (
            patch("builtins.input", side_effect=["", "", "", "", "n"]),
            patch("taster.dirs.get_config_dir", return_value=config_dir),
            patch("taster.dirs.get_profiles_dir", return_value=config_dir / "profiles"),
            patch("taster.dirs.get_env_path", return_value=config_dir / ".env"),
            patch("taster.dirs.ensure_dirs", return_value=config_dir),
        ):
            main(["init"])
