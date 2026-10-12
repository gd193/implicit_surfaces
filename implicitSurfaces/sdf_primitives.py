"""Back-compat shim — moved to `surfaces.implicit.primitives`."""
from surfaces.implicit.primitives import *  # noqa: F401, F403
from surfaces.implicit.primitives import (  # noqa: F401
    safe_norm,
    sphere_sdf,
    bbox_sdf,
    arc2d,
    torus_sdf,
    rounded_box_sdf,
    capsule_sdf,
    star2d_sdf,
    circle_wave2d_udf,
    circle_wave2d_sdf,
    union,
    intersection,
    subtraction,
    smooth_union,
)
