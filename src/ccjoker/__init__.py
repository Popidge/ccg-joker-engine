"""
CC Group Joker Engine package.
"""

from importlib.metadata import version as _version

try:
    __version__ = _version("cc-joker-engine")
except Exception:
    __version__ = "0.1.0"

__all__ = [
    "utils",
    "dataset",
    "model",
    "train",
    "eval",
]