"""Discovery + skip helpers for backend-parametrized tests.

Tests parametrize over `list_backends(flavor=X)` so newly-registered
backends auto-extend the matrix without test edits. When the
primitive-side math (`closest_point_on_primitive`,
`ray_primitive_intersect`) is not yet implemented, query tests
gracefully skip rather than fail.
"""
from __future__ import annotations

from typing import Tuple

import pytest

from surfaces.explicit.accelerators import (
    list_backends, BVHAccelerator, KdAccelerator, OctreeAccelerator,
)
from surfaces.explicit.accelerators.protocol import Config
from surfaces.explicit.primitives import Primitives


_FLAVOR_TO_ACCEL = {
    "bvh": BVHAccelerator,
    "kd": KdAccelerator,
    "octree": OctreeAccelerator,
}


def discover(flavor: str):
    """List of backend names for `flavor`. Empty list → tests skip."""
    return list_backends(flavor=flavor)


def parametrize_backends(flavor: str):
    """Pytest decorator: parametrize a test over every registered
    backend of the given flavor. Skip the test entirely if none exist.
    """
    names = discover(flavor)
    if not names:
        return pytest.mark.skip(reason=f"no {flavor} backends registered")
    return pytest.mark.parametrize("backend_name", names, ids=names)


def build_accelerator(flavor: str, backend_name: str,
                      primitives: Primitives,
                      config: Config | None = None):
    """Build an accelerator for `(flavor, backend_name)` over `primitives`.

    Some primitive callbacks raise `NotImplementedError` during build
    (currently `TriMeshPrimitives.closest_point_on_primitive` and
    `ray_primitive_intersect`). We let the build proceed and only catch
    `NotImplementedError` at the query call sites — it's the query
    that's blocked, not the build.
    """
    accel_cls = _FLAVOR_TO_ACCEL[flavor]
    return accel_cls.build(primitives, backend_name, config)


def skip_if_primitive_math_missing(callback_name: str,
                                   primitives: Primitives) -> None:
    """Probe a primitive callback and `pytest.skip` if it raises
    NotImplementedError. Avoids spamming failures while triangle math
    is still pending.
    """
    import jax.numpy as jnp
    try:
        if callback_name == "closest_point_on_primitive":
            primitives.closest_point_on_primitive(
                jnp.zeros((3,), dtype=jnp.float32), jnp.int32(0))
        elif callback_name == "ray_primitive_intersect":
            primitives.ray_primitive_intersect(
                jnp.zeros((3,), dtype=jnp.float32),
                jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32),
                jnp.float32(1.0), jnp.int32(0))
        elif callback_name == "primitive_aabb":
            primitives.primitive_aabb(jnp.int32(0))
        else:
            raise ValueError(f"unknown callback {callback_name!r}")
    except NotImplementedError as e:
        pytest.skip(f"{callback_name} not implemented: {e}")
    except Exception:
        # Other exceptions (shape errors, etc.) fall through; let the
        # test surface them.
        pass


# =====================================================================
# Tolerance bundle.
# =====================================================================


# Distances/positions in float32 with vmap, accumulated FMA error.
DIST_RTOL = 1e-4
DIST_ATOL = 1e-5
POINT_ATOL = 1e-4
T_RTOL = 1e-4
T_ATOL = 1e-5
