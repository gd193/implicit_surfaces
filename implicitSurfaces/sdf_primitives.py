"""Back-compat shim — moved to `surfaces.implicit.primitives`."""
from surfaces.implicit.primitives import *  # noqa: F401, F403
from surfaces.implicit.primitives import (  # noqa: F401
    safe_norm,
    sphere_sdf,
    bbox_sdf,
    arc2d,
)
