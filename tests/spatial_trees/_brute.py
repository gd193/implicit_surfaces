"""Layer-3 reference: brute-force spatial queries via project's `Primitives`.

Exhaustive O(N) reductions over `primitives.<callback>(query, idx)` for
every primitive index. Trivially correct given the callbacks are
correct; any mismatch between brute and a backend therefore localises
to the backend (build or traversal). The independent oracle in
`_oracle.py` exists to catch bugs that fool both brute and backends
because they share the same `Primitives` callbacks.

Hot-path performance is not a concern — these run in tests only.
"""
from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp

from surfaces.explicit.primitives import Primitives


# =====================================================================
# closest_point — exhaustive `argmin` over per-primitive distance.
# =====================================================================


def brute_closest_point(primitives: Primitives,
                        x: jnp.ndarray
                        ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """`(cp, dist, prim_idx)` from exhaustive enumeration."""
    idx = jnp.arange(primitives.n_primitives, dtype=jnp.int32)

    def per_prim(i):
        cp, d = primitives.closest_point_on_primitive(x, i)
        return cp, d

    cps, dists = jax.vmap(per_prim)(idx)
    j = jnp.argmin(dists)
    return cps[j], dists[j], j.astype(jnp.int32)


def brute_closest_point_batch(primitives: Primitives,
                              xs: jnp.ndarray
                              ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Vectorised over a `(M, 3)` query batch."""
    return jax.vmap(lambda x: brute_closest_point(primitives, x))(xs)


# =====================================================================
# first_hit — exhaustive `argmin` over per-primitive `t`.
# =====================================================================


def brute_first_hit(primitives: Primitives,
                    ray_origin: jnp.ndarray,
                    ray_direction: jnp.ndarray,
                    max_t: jnp.ndarray
                    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """`(t, prim_idx, hit_flag)`. Returns the smallest non-negative `t`
    among all primitives that report a hit within `[0, max_t]`.
    """
    idx = jnp.arange(primitives.n_primitives, dtype=jnp.int32)

    def per_prim(i):
        t, h = primitives.ray_primitive_intersect(ray_origin, ray_direction,
                                                  max_t, i)
        return t, h

    ts, hits = jax.vmap(per_prim)(idx)
    # Mask non-hits to +inf so argmin selects a real hit.
    inf = jnp.float32(jnp.inf)
    masked = jnp.where(hits, ts, inf)
    j = jnp.argmin(masked)
    any_hit = jnp.any(hits)
    return masked[j], j.astype(jnp.int32), any_hit


def brute_first_hit_batch(primitives: Primitives,
                          ray_origins: jnp.ndarray,
                          ray_directions: jnp.ndarray,
                          max_t: jnp.ndarray
                          ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    return jax.vmap(brute_first_hit, in_axes=(None, 0, 0, 0))(
        primitives, ray_origins, ray_directions, max_t)


# =====================================================================
# aabb_overlap — any-overlap reduction over per-primitive AABB.
# =====================================================================


def brute_aabb_overlap(primitives: Primitives,
                       q_lo: jnp.ndarray, q_hi: jnp.ndarray) -> jnp.ndarray:
    idx = jnp.arange(primitives.n_primitives, dtype=jnp.int32)

    def per_prim(i):
        p_lo, p_hi = primitives.primitive_aabb(i)
        return jnp.all(p_lo <= q_hi) & jnp.all(q_lo <= p_hi)

    overlaps = jax.vmap(per_prim)(idx)
    return jnp.any(overlaps)


def brute_aabb_overlap_batch(primitives: Primitives,
                             q_los: jnp.ndarray, q_his: jnp.ndarray
                             ) -> jnp.ndarray:
    return jax.vmap(brute_aabb_overlap, in_axes=(None, 0, 0))(
        primitives, q_los, q_his)
