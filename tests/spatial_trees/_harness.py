"""Backend parametrization and shared read-only builds."""
from __future__ import annotations

from functools import lru_cache

import pytest

from surfaces.explicit.accelerators import (
    list_backends, BVHAccelerator, KdAccelerator, OctreeAccelerator,
)
from surfaces.explicit.accelerators.protocol import Config
from surfaces.explicit.accelerators.registry import get_backend
from surfaces.explicit.primitives import Primitives


_FLAVOR_TO_ACCEL = {
    "bvh": BVHAccelerator,
    "kd": KdAccelerator,
    "octree": OctreeAccelerator,
}


def parametrize_backends(flavor=None, *, capability=None):
    names = list_backends(flavor=flavor)
    if capability:
        names = [name for name in names if getattr(get_backend(name), capability, False)]
    if not names:
        return pytest.mark.skip(reason=f"no {flavor} backends registered")
    return pytest.mark.parametrize("backend_name", names, ids=names)


@lru_cache(maxsize=None)
def build_accelerator(flavor: str, backend_name: str,
                      primitives: Primitives,
                      config: Config | None = None):
    """Reuse read-only accelerators across query checks."""
    accel_cls = _FLAVOR_TO_ACCEL[flavor]
    return accel_cls.build(primitives, backend_name, config)


def skip_if_primitive_math_missing(callback_name: str,
                                   primitives: Primitives) -> None:
    """Skip unsupported primitive callbacks."""
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


DIST_RTOL = 1e-4
DIST_ATOL = 1e-5
T_RTOL = 1e-4
T_ATOL = 1e-5
