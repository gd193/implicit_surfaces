"""Layer-4 self-consistency invariants.

Properties that must hold for any backend's output regardless of input
or oracle. Catches plausible-looking but inconsistent results (e.g.
correct distance + wrong primitive index from a leaf-list bug).
"""
from __future__ import annotations

import jax.numpy as jnp

from surfaces.explicit.primitives import Primitives


_DIST_RTOL = 1e-4
_DIST_ATOL = 1e-5
_POINT_ATOL = 1e-4


def assert_closest_point_consistent(primitives: Primitives,
                                    x: jnp.ndarray,
                                    cp: jnp.ndarray,
                                    dist: jnp.ndarray,
                                    prim_idx: jnp.ndarray) -> None:
    """`‖cp − x‖ ≈ dist`, `prim_idx ∈ [0, n)`,
    re-evaluating closest_point on the reported primitive collapses to
    a near-zero distance from `cp`.
    """
    n = primitives.n_primitives
    assert int(prim_idx) >= 0 and int(prim_idx) < n, (
        f"prim_idx {int(prim_idx)} out of [0,{n})")
    norm = float(jnp.linalg.norm(cp - x))
    assert abs(norm - float(dist)) <= _DIST_ATOL + _DIST_RTOL * float(dist), (
        f"||cp - x|| = {norm} disagrees with reported dist {float(dist)}")
    cp_re, d_re = primitives.closest_point_on_primitive(cp, prim_idx)
    assert float(d_re) <= _POINT_ATOL, (
        f"reported cp not on primitive {int(prim_idx)}: "
        f"dist-from-cp = {float(d_re)}")


def assert_first_hit_consistent(primitives: Primitives,
                                ray_origin: jnp.ndarray,
                                ray_direction: jnp.ndarray,
                                max_t: jnp.ndarray,
                                t: jnp.ndarray,
                                prim_idx: jnp.ndarray,
                                hit: jnp.ndarray) -> None:
    """If `hit`: `0 ≤ t ≤ max_t`, hit point lies on `prim_idx`."""
    if not bool(hit):
        return
    t_f = float(t)
    assert t_f >= 0.0, f"hit t={t_f} negative"
    assert t_f <= float(max_t) + _DIST_ATOL, (
        f"hit t={t_f} exceeds max_t={float(max_t)}")
    assert int(prim_idx) >= 0, f"hit but prim_idx={int(prim_idx)}"
    hit_pt = ray_origin + t * ray_direction
    _cp, d_re = primitives.closest_point_on_primitive(hit_pt, prim_idx)
    assert float(d_re) <= _POINT_ATOL, (
        f"hit point not on primitive {int(prim_idx)}: "
        f"dist = {float(d_re)}")


def assert_aabb_overlap_consistent(primitives: Primitives,
                                   q_lo: jnp.ndarray, q_hi: jnp.ndarray,
                                   reported: jnp.ndarray) -> None:
    """Cross-checks `reported` (any/no overlap) against an O(N) sweep
    over `primitive_aabb`. Bool-exact equality.
    """
    import jax
    idx = jnp.arange(primitives.n_primitives, dtype=jnp.int32)

    def _check(i):
        p_lo, p_hi = primitives.primitive_aabb(i)
        return jnp.all(p_lo <= q_hi) & jnp.all(q_lo <= p_hi)

    truth = bool(jnp.any(jax.vmap(_check)(idx)))
    assert bool(reported) == truth, (
        f"aabb_overlap reported {bool(reported)}, ground truth {truth}")
