"""Tests for sommelier.compat — optional dependency helpers."""

import pytest
from types import ModuleType
from unittest.mock import patch

from sommelier.compat import require


class TestRequire:
    def test_returns_module_when_installed(self):
        mod = require("json", "core")
        assert isinstance(mod, ModuleType)
        assert hasattr(mod, "dumps")

    def test_raises_import_error_when_missing(self):
        with pytest.raises(ImportError, match="pip install sommelier\\[fakeextra\\]"):
            require("nonexistent_package_xyz_123", "fakeextra")

    def test_error_message_includes_package_name(self):
        with pytest.raises(ImportError, match="'nonexistent_pkg'"):
            require("nonexistent_pkg", "test")

    def test_error_message_includes_extra_name(self):
        with pytest.raises(ImportError, match="sommelier\\[gemini\\]"):
            require("nonexistent_gemini_pkg", "gemini")

    def test_idempotent_calls(self):
        mod1 = require("os", "core")
        mod2 = require("os", "core")
        assert mod1 is mod2

    def test_nested_package(self):
        mod = require("os.path", "core")
        assert isinstance(mod, ModuleType)
        assert hasattr(mod, "join")

    def test_returns_correct_module(self):
        mod = require("pathlib", "core")
        assert hasattr(mod, "Path")

    def test_no_chained_exception(self):
        """Verify from None suppresses the original ImportError chain."""
        with pytest.raises(ImportError) as exc_info:
            require("totally_fake_pkg", "test")
        assert exc_info.value.__cause__ is None

    def test_works_for_yaml(self):
        mod = require("yaml", "core")
        assert hasattr(mod, "safe_load")

    def test_works_for_pydantic(self):
        mod = require("pydantic", "core")
        assert hasattr(mod, "BaseModel")

    def test_missing_package_different_extras(self):
        """Different extra names produce different error messages."""
        with pytest.raises(ImportError, match="sommelier\\[ml\\]"):
            require("fake_ml_pkg", "ml")
        with pytest.raises(ImportError, match="sommelier\\[api\\]"):
            require("fake_api_pkg", "api")

    def test_error_is_import_error_subclass(self):
        with pytest.raises(ImportError):
            require("nonexistent_xyz", "test")
