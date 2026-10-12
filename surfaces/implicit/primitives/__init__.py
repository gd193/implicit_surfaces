"""SDF primitives split by dimension: sdfnd (any dimension), sdf2d, sdf3d, and
combinators. Everything is re-exported here, so
`from surfaces.implicit.primitives import sphere_sdf` keeps working."""
from .combinators import intersection, smooth_union, subtraction, union
from .sdf2d import arc2d, circle_wave2d_sdf, circle_wave2d_udf, star2d_sdf
from .sdf3d import torus_sdf
from .sdfnd import bbox_sdf, capsule_sdf, rounded_box_sdf, safe_norm, sphere_sdf

__all__ = ["safe_norm", "sphere_sdf", "bbox_sdf", "rounded_box_sdf", "capsule_sdf",
           "arc2d", "star2d_sdf", "circle_wave2d_udf", "circle_wave2d_sdf", "torus_sdf",
           "union", "intersection", "subtraction", "smooth_union"]
