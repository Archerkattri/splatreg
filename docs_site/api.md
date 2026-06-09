# API reference

The import surface is flat: everything below is available from `splatreg` directly (or the
named submodule). Optional-dependency features (`localize_camera` needs `splatreg[render]`)
degrade to `None` at import time rather than breaking the package.

## Registration

::: splatreg.api.register

::: splatreg.api.merge

::: splatreg.api.Tracker

## Multi-splat bundle registration

::: splatreg.bundle.bundle_register

::: splatreg.bundle.pairwise_consistency

## Object pose

::: splatreg.object_pose.estimate_object_pose

::: splatreg.object_pose.ObjectPoseEstimator

::: splatreg.object_pose.add_metric

::: splatreg.object_pose.adds_metric

::: splatreg.object_pose.add_auc

## Camera localization

::: splatreg.camera_loc.localize_camera

::: splatreg.camera_loc.coarse_localize_camera

## Core types

::: splatreg.core.types.Gaussians

::: splatreg.core.types.Frame

::: splatreg.core.types.RegisterResult

## PLY + gsplat I/O

::: splatreg.io.load_ply

::: splatreg.io.save_ply

::: splatreg.io.from_gsplat

::: splatreg.io.to_gsplat

## Gaussian-SDF field

::: splatreg.geometry.gaussian_sdf.gaussian_sdf

::: splatreg.geometry.gaussian_sdf.gaussian_sdf_grad

## Fusion / dedupe

::: splatreg.fuse.voxel_dedupe

::: splatreg.fuse.knn_dedupe

::: splatreg.fuse.auto_voxel_size

::: splatreg.fuse.auto_knn_radius

## Spatial index

::: splatreg.spatial_index.build_index

::: splatreg.spatial_index.SpatialIndex

## Quality policy

::: splatreg.quality.resolve_quality

::: splatreg.quality.QualityConfig

## Extension points

::: splatreg.residuals.base.Residual

::: splatreg.solvers.base.Solver

::: splatreg.testing.assert_residual_jacobian

## Command line

::: splatreg.cli.main

::: splatreg.cli.build_parser
