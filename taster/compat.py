"""Optional dependency helpers.

Provides a single ``require()`` function that imports a package or raises
an ``ImportError`` with a clear install hint pointing at the correct
``pip install taster[extra]`` command.
"""

import importlib
from types import ModuleType


def require(package: str, extra: str) -> ModuleType:
    """Import *package* or raise a clear install hint.

    >>> torch = require("torch", "ml")
    """
    try:
        return importlib.import_module(package)
    except ImportError:
        raise ImportError(
            f"'{package}' is required for this feature. "
            f"Install it with:  pip install taster[{extra}]"
        ) from None
