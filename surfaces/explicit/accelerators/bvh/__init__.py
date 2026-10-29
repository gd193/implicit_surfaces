"""BVH-flavored mesh accelerator.

Binary tree over primitive AABBs, built bottom-up. Distinct from
kd/octree: no implicit-tree reuse path (no implicit BVH), built fresh
from primitives.

Layout:
  payload.py        — BVHPayload NamedTuple (SoA jnp arrays).
  accelerator.py    — BVHAccelerator(MeshAccelerator) shell.
  backends/         — concrete (algorithm, runtime) impls.
"""
from .accelerator import BVHAccelerator  # noqa: F401
from .payload import BVHPayload  # noqa: F401
from . import backends  # noqa: F401  (registers backends on import)
