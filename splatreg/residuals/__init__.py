"""Residual plugins. The base ABC always imports; concrete residuals load if their module exists
(so the package imports cleanly at every stage of the carve)."""
from .base import Residual

__all__ = ["Residual"]

try:
    from .sdf import SDF  # noqa: F401
    __all__.append("SDF")
except ImportError:
    pass
try:
    from .icp import ICP  # noqa: F401
    __all__.append("ICP")
except ImportError:
    pass
try:
    from .prior import Prior  # noqa: F401
    __all__.append("Prior")
except ImportError:
    pass
try:
    from .photometric import Photometric  # noqa: F401
    __all__.append("Photometric")
except ImportError:
    pass
