"""splatreg — composable geometry-first SE(3)/Sim(3) registration for 3D Gaussian Splatting.

*gsplat renders your Gaussians; splatreg registers against them.*

Public surface (filled in by the carve):
    register(target, source, residuals=[...], transform="sim3", backend="builtin") -> RegisterResult
    merge([a, b, ...], ref=0) -> Gaussians
    Tracker(target, residuals=[...]).track(frame) -> RegisterResult
    Residual, Solver  (extension points)
"""

from .core.types import Gaussians, Frame, RegisterResult, LinearizedProblem, SE3Update
from .residuals.base import Residual
from .solvers.base import Solver
from .quality import QualityConfig, resolve_quality

# The high-level pipeline (splatreg.api) is added by the carve; tolerate its absence pre-build.
try:
    from .api import register, merge, Tracker  # noqa: F401
except ImportError:
    register = merge = Tracker = None  # type: ignore

__version__ = "0.0.1"
__all__ = [
    "register",
    "merge",
    "Tracker",
    "Residual",
    "Solver",
    "QualityConfig",
    "resolve_quality",
    "Gaussians",
    "Frame",
    "RegisterResult",
    "LinearizedProblem",
    "SE3Update",
]
