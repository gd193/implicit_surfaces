"""Implicit surface family.

Level-set surfaces of `fn(params, x) - level_set`. Bound machinery
(Lipschitz / autobound / NIQ-style polynomial) provides the universal
`Surface` query methods (`empty_sphere`, `ray_step`, `classify_box`).

Public names re-exported here for ergonomic import. The lowercase
`implicitSurface` is a back-compat alias for `ImplicitSurface`; new code
should use the latter.
"""
from .base import ImplicitSurface, implicitSurface  # noqa: F401
from . import primitives  # noqa: F401
