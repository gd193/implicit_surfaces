"""Two regions split by the plane w.x + d = 0. Exact box queries; test reference."""
import jax.numpy as jnp
import numpy as np

from boundaryConditions.base import MIXED, BoundaryPartition, float_dtype, storage_float_dtype


class HalfSpacePartition(BoundaryPartition):
    """w.x + d < 0 -> region_below, otherwise region_above (plane included)."""

    def __init__(self, normal, offset, region_below: int = 0, region_above: int = 1,
                 n_regions: int | None = None):
        normal_np = np.asarray(normal, dtype=np.float64)
        if normal_np.ndim != 1 or not np.all(np.isfinite(normal_np)) or not np.any(normal_np):
            raise ValueError("normal must be a finite, nonzero vector")
        if not np.isfinite(offset):
            raise ValueError("offset must be finite")
        regions = (int(region_below), int(region_above))
        if min(regions) < 0:
            raise ValueError("region ids must be nonnegative")
        super().__init__(len(normal_np), max(regions) + 1 if n_regions is None else n_regions)
        if max(regions) >= self.n_regions:
            raise ValueError("region ids must be below n_regions")
        self.normal = jnp.asarray(normal_np, storage_float_dtype())
        self.offset = jnp.asarray(offset, storage_float_dtype())
        self.regions = regions

    def _below(self, value):
        return jnp.where(value < 0, jnp.int32(self.regions[0]), jnp.int32(self.regions[1]))

    def classify_point(self, params, x):
        ftype = float_dtype(x)
        x = jnp.asarray(x, ftype)
        return self._below(x @ self.normal.astype(ftype) + self.offset.astype(ftype))

    def classify_box(self, params, lower, upper):
        ftype = float_dtype(lower, upper)
        lower, upper = jnp.asarray(lower, ftype), jnp.asarray(upper, ftype)
        w, d = self.normal.astype(ftype), self.offset.astype(ftype)
        center, half = 0.5 * (lower + upper), 0.5 * (upper - lower)
        value, spread = center @ w + d, jnp.abs(w) @ half
        # Rounding of the dot products, widened generously.
        pad = 8 * jnp.finfo(ftype).eps * (jnp.abs(w) @ (jnp.abs(center) + half) + jnp.abs(d))
        valid = jnp.all(jnp.isfinite(lower) & jnp.isfinite(upper) & (lower <= upper))
        label = jnp.where(value + spread + pad < 0, jnp.int32(self.regions[0]),
                          jnp.where(value - spread - pad >= 0, jnp.int32(self.regions[1]),
                                    jnp.int32(MIXED)))
        return jnp.where(valid, label, jnp.int32(MIXED))
