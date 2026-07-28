from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .pipeline import VegetationPipeline

__all__ = ["VegetationPipeline"]


def __getattr__(name: str) -> Any:
    """Load the TensorFlow pipeline only when callers request it."""
    if name == "VegetationPipeline":
        from .pipeline import VegetationPipeline

        return VegetationPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
