"""Implicit surface family.

Level-set surfaces of `fn(params, x) - level_set`. Bound machinery
(Lipschitz / autobound / NIQ-style polynomial / Harnack) provides the
query methods (`empty_sphere`, `ray_step`, `classify_box`) consumed by
integrators and renderers.
"""
from .base import ImplicitSurface  # noqa: F401
from . import primitives  # noqa: F401
