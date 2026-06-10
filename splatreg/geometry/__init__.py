"""Geometry primitives, the Gaussian-derived SDF query (splatreg's differentiator)."""

__all__ = []

try:
    from .gaussian_sdf import gaussian_sdf  # noqa: F401

    __all__.append("gaussian_sdf")
except ImportError:
    pass
