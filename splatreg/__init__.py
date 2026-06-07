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

# v0.2: 6-DoF object-pose mode (pure torch — reuses register/track, always available).
try:
    from .object_pose import (  # noqa: F401
        ObjectPose,
        ObjectPoseEstimator,
        estimate_object_pose,
        add_metric,
        adds_metric,
        add_auc,
    )
except ImportError:  # pragma: no cover
    ObjectPose = ObjectPoseEstimator = estimate_object_pose = None  # type: ignore
    add_metric = adds_metric = add_auc = None  # type: ignore

# v0.2: camera localization in a splat (needs the gsplat [render] extra — guarded).
try:
    from .camera_loc import CameraPhotometric, localize_camera  # noqa: F401
except ImportError:  # gsplat not installed
    CameraPhotometric = localize_camera = None  # type: ignore

# v0.3: multi-splat joint/bundle registration + scene-scale spatial index (pure torch).
try:
    from .bundle import bundle_register, pairwise_consistency, BundleResult  # noqa: F401
except ImportError:  # pragma: no cover
    bundle_register = pairwise_consistency = BundleResult = None  # type: ignore
try:
    from .spatial_index import SpatialIndex, build_index  # noqa: F401
except ImportError:  # pragma: no cover
    SpatialIndex = build_index = None  # type: ignore

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
    # v0.2 object-pose
    "ObjectPose",
    "ObjectPoseEstimator",
    "estimate_object_pose",
    "add_metric",
    "adds_metric",
    "add_auc",
    # v0.2 camera localization
    "CameraPhotometric",
    "localize_camera",
    # v0.3 multi-splat bundle registration
    "bundle_register",
    "pairwise_consistency",
    "BundleResult",
    # v0.3 scene-scale spatial index
    "SpatialIndex",
    "build_index",
]
