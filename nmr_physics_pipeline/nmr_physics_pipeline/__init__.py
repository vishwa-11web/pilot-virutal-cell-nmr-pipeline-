"""NMR-to-Physics AI/ML Pipeline package."""

from __future__ import annotations

import sys
from importlib import import_module

__version__ = "0.1.0"

data = import_module("data")
models = import_module("models")

sys.modules[__name__ + ".data"] = data
sys.modules[__name__ + ".models"] = models

__all__ = ["data", "models"]
