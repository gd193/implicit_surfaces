"""Backend parametrization and shared read-only builds."""
from __future__ import annotations

from functools import lru_cache

import pytest

from sengi import list_backends, BVHAccelerator
from sengi.protocol import Config
from sengi.registry import get_backend
from sengi.primitives import Primitives


_FLAVOR_TO_ACCEL = {
    "bvh": BVHAccelerator,
}


def parametrize_backends(flavor=None, *, capability=None):
    """Exercise supported query runtimes; builders have separate coverage."""
    names = list_backends(flavor=flavor, runtime="warp")
    if capability:
        names = [name for name in names if getattr(get_backend(name), capability, False)]
    return pytest.mark.parametrize("backend_name", names, ids=names)


@lru_cache(maxsize=None)
def build_accelerator(flavor: str, backend_name: str,
                      primitives: Primitives,
                      config: Config | None = None):
    """Reuse read-only accelerators across query checks."""
    accel_cls = _FLAVOR_TO_ACCEL[flavor]
    return accel_cls.build(primitives, backend_name, config)


DIST_RTOL = 1e-4
DIST_ATOL = 1e-5
T_RTOL = 1e-4
T_ATOL = 1e-5
