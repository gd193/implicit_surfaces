"""Function-backed boundary data.

One class covers constant and spatially varying data: the user function
receives the region id and chooses how to dispatch. A gather such as
`params['values'][r]` costs the same for any region count; `from_region_fns`
builds a lax.switch over per-region formulas, which traces every branch.
"""
from typing import Any, Callable, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from boundaryConditions.base import BoundaryData, BoundaryKind, BoundaryValues, float_dtype


class FunctionBoundaryData(BoundaryData):
    """fn(params, region_id, x, outward_normal) -> c of shape value_shape.

    region_id arrives as an int32 scalar in [0, n_regions); x and the normal
    are already cast to float_dtype(x). The result is cast to that dtype; its
    shape must equal value_shape exactly (no broadcasting).
    """

    def __init__(self, kinds: Sequence[int],
                 fn: Callable[[Any, jax.Array, jax.Array, jax.Array], Any],
                 value_shape: Tuple[int, ...] = ()):
        kinds_np = np.asarray(kinds)
        if kinds_np.ndim != 1 or not np.issubdtype(kinds_np.dtype, np.integer):
            raise ValueError("kinds must be a 1-D integer sequence")
        supported = {int(BoundaryKind.DIRICHLET), int(BoundaryKind.NEUMANN)}
        if not set(kinds_np.tolist()) <= supported:
            raise ValueError("only DIRICHLET and NEUMANN kinds are supported")
        super().__init__(len(kinds_np), value_shape)
        self._kinds = kinds_np.astype(np.int32)
        self.fn = fn

    @classmethod
    def from_region_fns(cls, kinds: Sequence[int],
                        fns: Sequence[Callable[[Any, jax.Array, jax.Array], Any]],
                        value_shape: Tuple[int, ...] = ()):
        """One formula per region: fns[r](params, x, outward_normal) -> c."""
        if len(fns) != len(kinds):
            raise ValueError("need one function per region")

        def fn(params, region_id, x, normal):
            ftype = x.dtype
            # Every branch must return the same dtype and shape for lax.switch.
            branches = [lambda p, x, n, f=f: _checked(f(p, x, n), ftype, value_shape)
                        for f in fns]
            return jax.lax.switch(region_id, branches, params, x, normal)
        return cls(kinds, fn, value_shape)

    def kinds(self) -> jax.Array:
        return jnp.asarray(self._kinds, jnp.int32)

    def evaluate(self, params, region_id, x, outward_normal) -> BoundaryValues:
        ftype = float_dtype(x)
        x = jnp.asarray(x, ftype)
        normal = jnp.asarray(outward_normal, ftype)
        # Valid ids are a precondition; clipping makes out-of-range behavior
        # explicit rather than relying on gather/switch clamping.
        region = jnp.clip(jnp.asarray(region_id).astype(jnp.int32), 0, self.n_regions - 1)
        c = _checked(self.fn(params, region, x, normal), ftype, self.value_shape)
        dirichlet = self.kinds()[region] == BoundaryKind.DIRICHLET
        one, zero = jnp.asarray(1, ftype), jnp.asarray(0, ftype)
        return BoundaryValues(a=jnp.where(dirichlet, one, zero),
                              b=jnp.where(dirichlet, zero, one),
                              c=c)


def _checked(value, ftype, value_shape):
    """Cast boundary data to ftype; its shape must match value_shape exactly."""
    value = jnp.asarray(value, ftype)
    if value.shape != tuple(value_shape):
        raise ValueError(f"boundary data has shape {value.shape}, expected {tuple(value_shape)}")
    return value
