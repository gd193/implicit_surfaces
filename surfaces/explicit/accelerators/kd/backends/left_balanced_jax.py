"""Left-balanced kd-tree backend, pure JAX.

Build follows the cudaKDTree (Wald 2022) note:
  algo_infos_crawled/kd.left_balanced.wald_cudaKDTree.md

Adaptation: cudaKDTree builds over points; we build over triangle-AABB
centroids and propagate per-subtree AABBs so the tree can serve our
extent-having primitive flavor (every node both splits AND owns one
primitive). Round-robin split axis (`level % 3`); no SAH or widest-axis
variant.

Build is host-side NumPy: an O(N log N) sort-based recursive median
selection mirrors the left-balanced "tag-update cascade" output without
the GPU-cascade machinery. Traversal is pure JAX (jax.jit /
lax.while_loop) using a fixed-depth explicit stack, vmapped over the
query batch — same idiom as `bvh.lbvh.jax`.
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from surfaces.explicit.accelerators.protocol import Backend, Config
from surfaces.explicit.accelerators.registry import register_backend
from surfaces.explicit.accelerators.kd.payload import KdPayload
from surfaces.explicit.primitives import Primitives


_INDEX_DTYPE = jnp.int32
_NDIM = 3


# =====================================================================
# Host-side build helpers.
# =====================================================================


def _left_subtree_size(n: int) -> int:
    """Number of nodes in the LEFT subtree of a left-balanced tree with `n`
    nodes, root at 0. Mirrors cudaKDTree's `ArbitraryBinaryTree` math
    (helpers.h:162-178). For `n` total nodes, the last level holds
    `n - (2^(L-1) - 1)` nodes (where `L = ceil(log2(n+1))` levels). The
    full left subtree size is `2^(L-1) - 1`; the lowest level adds up to
    `2^(L-2)` extras, capped by what is actually present.
    """
    if n <= 1:
        return 0
    levels = int(math.floor(math.log2(n))) + 1   # numLevelsFor
    last_level_capacity = 1 << (levels - 1)
    full_above = (1 << (levels - 1)) - 1         # everything above the last
    last_level_count = n - full_above
    left_full_above = (full_above - 1) // 2 if full_above > 0 else 0
    left_last_capacity = last_level_capacity // 2
    left_last_actual = min(last_level_count, left_last_capacity)
    return left_full_above + left_last_actual


def _build_left_balanced_np(centroids: np.ndarray) -> Tuple[np.ndarray,
                                                            np.ndarray]:
    """Permute primitive indices into a left-balanced kd-tree layout.

    Mirrors the *output* of cudaKDTree's tag-update cascade
    (note Build pipeline, builder_thrust.h): node `i` holds the median
    primitive of the i-th subtree along the level's split axis. Round-
    robin axis = `level(i) % 3`.

    Returns:
      prim_ids (N,)    : permutation of [0, N) — `prim_ids[i]` is the
                         primitive seated at node `i`.
      split_axis (N,)  : `level(i) % 3` for each node.
    """
    n = int(centroids.shape[0])
    prim_ids = np.empty((n,), dtype=np.int32)
    split_axis = np.empty((n,), dtype=np.int32)

    # Iterative recursion over (segment of original-id slice, dest node id,
    # depth). Each step picks the median along the level's axis, writes
    # it at `node_id`, recurses on left/right slices.
    work_idx = np.arange(n, dtype=np.int32)
    stack = [(work_idx, 0, 0)]
    while stack:
        ids, node_id, depth = stack.pop()
        m = ids.shape[0]
        if m == 0:
            continue
        axis = depth % _NDIM
        # Pivot position inside this segment so the LEFT child gets exactly
        # `_left_subtree_size(m)` items — reproduces cudaKDTree's
        # `ArrayLayoutInStep::pivotPosOf` choice.
        left_size = _left_subtree_size(m)
        coords = centroids[ids, axis]
        # `np.argpartition` is O(N); the K-th smallest goes to position K,
        # everything below is < pivot, everything above is >= pivot.
        order = np.argpartition(coords, left_size)
        left_part = ids[order[:left_size]]
        pivot = ids[order[left_size]]
        right_part = ids[order[left_size + 1:]]

        prim_ids[node_id] = pivot
        split_axis[node_id] = axis

        # Children of node `i`: 2i+1 (left), 2i+2 (right).
        left_child = 2 * node_id + 1
        right_child = 2 * node_id + 2
        if left_child < n:
            stack.append((left_part, left_child, depth + 1))
        if right_child < n:
            stack.append((right_part, right_child, depth + 1))

    return prim_ids, split_axis


def _refit_aabbs_np(prim_lo: np.ndarray, prim_hi: np.ndarray,
                    prim_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Bottom-up AABB propagation over the implicit left-balanced tree.

    Each node owns one primitive (its own AABB) and the union of its
    children's bounds. Iterates nodes in reverse index order — for a
    left-balanced layout that visits all descendants of node `i` before
    `i` itself (children `2i+1`, `2i+2` always have larger indices).
    """
    n = int(prim_ids.shape[0])
    INF = np.float32(np.inf)
    node_lo = np.full((n, 3), INF, dtype=np.float32)
    node_hi = np.full((n, 3), -INF, dtype=np.float32)
    for i in range(n - 1, -1, -1):
        pid = int(prim_ids[i])
        lo = prim_lo[pid]
        hi = prim_hi[pid]
        l = 2 * i + 1
        r = 2 * i + 2
        if l < n:
            lo = np.minimum(lo, node_lo[l])
            hi = np.maximum(hi, node_hi[l])
        if r < n:
            lo = np.minimum(lo, node_lo[r])
            hi = np.maximum(hi, node_hi[r])
        node_lo[i] = lo
        node_hi[i] = hi
    return node_lo, node_hi


def _build_payload(prim_lo_np: np.ndarray, prim_hi_np: np.ndarray
                   ) -> KdPayload:
    """Host-side kd-tree build. Returns a `KdPayload` with jnp arrays."""
    n = int(prim_lo_np.shape[0])
    assert n >= 1, "Need at least one primitive"

    # Build over centroids (note: adaptation step 1).
    centroids = (0.5 * (prim_lo_np + prim_hi_np)).astype(np.float32)

    # Permute primitives into left-balanced order (note: adaptation step 1).
    prim_ids, split_axis = _build_left_balanced_np(centroids)

    # Per-node split coordinate is the centroid component on the chosen axis
    # of the primitive seated there (note: adaptation step 4).
    node_centroids = centroids[prim_ids]  # (N, 3)
    node_arange = np.arange(n, dtype=np.int32)
    split_pos = node_centroids[node_arange, split_axis].astype(np.float32)

    # Implicit topology: children of i are 2i+1, 2i+2; -1 if out of range
    # (note: adaptation step 2).
    left = 2 * node_arange + 1
    right = 2 * node_arange + 2
    node_left = np.where(left < n, left, -1).astype(np.int32)
    node_right = np.where(right < n, right, -1).astype(np.int32)

    # Each node owns exactly one primitive (note: adaptation step 4).
    node_leaf_offset = node_arange.astype(np.int32)
    node_leaf_count = np.ones((n,), dtype=np.int32)

    # Subtree AABBs (note: adaptation step 3).
    node_lo, node_hi = _refit_aabbs_np(
        prim_lo_np.astype(np.float32),
        prim_hi_np.astype(np.float32),
        prim_ids,
    )

    return KdPayload(
        node_split_axis=jnp.asarray(split_axis),
        node_split_pos=jnp.asarray(split_pos),
        node_left=jnp.asarray(node_left),
        node_right=jnp.asarray(node_right),
        node_lower=jnp.asarray(node_lo),
        node_upper=jnp.asarray(node_hi),
        node_leaf_offset=jnp.asarray(node_leaf_offset),
        node_leaf_count=jnp.asarray(node_leaf_count),
        prim_ids=jnp.asarray(prim_ids),
        root=jnp.asarray(np.int32(0)),
    )


# =====================================================================
# Traversal helpers.
# =====================================================================


def _sqr_dist_point_to_box(p: jnp.ndarray, lo: jnp.ndarray,
                           hi: jnp.ndarray) -> jnp.ndarray:
    cl = jnp.clip(p, lo, hi)
    d = p - cl
    return jnp.sum(d * d)


def _ray_box_t(origin: jnp.ndarray, rcp_dir: jnp.ndarray,
               lo: jnp.ndarray, hi: jnp.ndarray,
               tmin: jnp.ndarray, tmax: jnp.ndarray
               ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    t1 = (lo - origin) * rcp_dir
    t2 = (hi - origin) * rcp_dir
    t_lo = jnp.minimum(t1, t2)
    t_hi = jnp.maximum(t1, t2)
    t_enter = jnp.maximum(jnp.max(t_lo), tmin)
    t_exit = jnp.minimum(jnp.min(t_hi), tmax)
    hit = t_enter <= t_exit
    return t_enter, hit


def _box_overlap(a_lo, a_hi, b_lo, b_hi) -> jnp.ndarray:
    return jnp.all(a_lo <= b_hi) & jnp.all(b_lo <= a_hi)


def _box_box_dist_bounds(a_lo: jnp.ndarray, a_hi: jnp.ndarray,
                         b_lo: jnp.ndarray, b_hi: jnp.ndarray
                         ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Unsigned distance bounds between two AABBs.

    Lower bound: per-axis non-negative separation, Euclidean-norm'd.
        sep_i = max(0, max(a_lo_i - b_hi_i, b_lo_i - a_hi_i))
        d_lo = sqrt(sum(sep_i^2))
    Upper bound: maximum corner-to-corner distance — bounded above by
        d_hi = sqrt(sum(max((a_hi_i - b_lo_i)^2, (b_hi_i - a_lo_i)^2)))
    Identical to the bvh.sah.jax helper; ported verbatim.
    """
    sep = jnp.maximum(jnp.maximum(a_lo - b_hi, b_lo - a_hi),
                      jnp.float32(0.0))
    d_lo2 = jnp.sum(sep * sep)
    span = jnp.maximum((a_hi - b_lo) ** 2, (b_hi - a_lo) ** 2)
    d_hi2 = jnp.sum(span)
    eps = jnp.float32(0.0)
    return jnp.sqrt(jnp.maximum(d_lo2, eps)), jnp.sqrt(jnp.maximum(d_hi2, eps))


# Sentinel meaning "no node" inside the traversal stack & cursor.
_NO_NODE = jnp.int32(-1)


# =====================================================================
# closest_point — shrinking-radius descent over the left-balanced tree.
# Translates the note's stack-based traversal (traverse-default-stack-
# based.h) but with AABB-distance pruning instead of 1-D plane culling,
# since each node also owns a primitive and carries a subtree AABB.
# =====================================================================


def _closest_point_single(payload: KdPayload, primitives: Primitives,
                          x: jnp.ndarray, max_stack_depth: int):
    INF = jnp.float32(jnp.inf)

    stack_idx_init = jnp.full((max_stack_depth,), -1, dtype=_INDEX_DTYPE)
    stack_d_init = jnp.full((max_stack_depth,), INF, dtype=jnp.float32)

    # state: (cur, stack_idx, stack_d, sp, best_d2, best_cp, best_pid, done)
    init = (payload.root.astype(_INDEX_DTYPE),
            stack_idx_init, stack_d_init,
            jnp.asarray(0, dtype=_INDEX_DTYPE),
            INF,
            jnp.zeros((3,), dtype=jnp.float32),
            jnp.asarray(-1, dtype=_INDEX_DTYPE),
            jnp.bool_(False))

    def cond_fn(state):
        return jnp.logical_not(state[7])

    def _pop(si, sd, sp, best_d, best_cp, best_pid):
        # Pop entries whose AABB-distance bound is no longer beaten by
        # `best_d`. Mirrors traverse-default-stack-based.h:98-106.
        def pop_cond(s):
            _ssp, _ssi, _ssd, found, empty = s
            return jnp.logical_not(found | empty)

        def pop_body(s):
            ssp, ssi, ssd, _f, _e = s
            empty = ssp == 0
            new_sp = jnp.where(empty, ssp, ssp - 1)
            top_d = ssd[new_sp]
            in_range = (top_d < best_d) & jnp.logical_not(empty)
            return new_sp, ssi, ssd, in_range, empty

        new_sp, _, _, found, _empty = lax.while_loop(
            pop_cond, pop_body,
            (sp, si, sd, jnp.bool_(False), jnp.bool_(False)))
        next_node = jnp.where(found, si[new_sp], jnp.asarray(0, _INDEX_DTYPE))
        done = jnp.logical_not(found)
        return next_node, si, sd, new_sp, best_d, best_cp, best_pid, done

    def body_fn(state):
        cur, si, sd, sp, best_d, best_cp, best_pid, _done = state

        # (a) Test the primitive seated at this node.
        pid = payload.prim_ids[cur]
        cp_k, dist_k = primitives.closest_point_on_primitive(x, pid)
        d2_k = dist_k * dist_k
        better = d2_k < best_d
        best_d = jnp.where(better, d2_k, best_d)
        best_cp = jnp.where(better, cp_k, best_cp)
        best_pid = jnp.where(better, pid, best_pid)

        # (b) Children + AABB-distance prune.
        l = payload.node_left[cur]
        r = payload.node_right[cur]
        has_l = l >= 0
        has_r = r >= 0
        # Replace -1 with cur for safe gather; pruning will mask via `has_*`.
        safe_l = jnp.where(has_l, l, cur)
        safe_r = jnp.where(has_r, r, cur)
        d_l = _sqr_dist_point_to_box(x, payload.node_lower[safe_l],
                                     payload.node_upper[safe_l])
        d_r = _sqr_dist_point_to_box(x, payload.node_lower[safe_r],
                                     payload.node_upper[safe_r])
        # If the child does not exist or its AABB is already farther than
        # the current best, mark its distance as +inf to drop it.
        d_l = jnp.where(has_l & (d_l < best_d), d_l, INF)
        d_r = jnp.where(has_r & (d_r < best_d), d_r, INF)

        any_child = (d_l < INF) | (d_r < INF)
        # Order: descend near-first, push far if it survives.
        l_first = d_l <= d_r
        near = jnp.where(l_first, safe_l, safe_r)
        far = jnp.where(l_first, safe_r, safe_l)
        d_near = jnp.where(l_first, d_l, d_r)
        d_far = jnp.where(l_first, d_r, d_l)

        push_far = d_far < INF
        do_descend = d_near < INF

        def descend_path(_):
            # Optionally push far child onto the stack.
            def push_then_go(_):
                overflow = sp >= max_stack_depth
                nsi = jnp.where(overflow, si, si.at[sp].set(far))
                nsd = jnp.where(overflow, sd, sd.at[sp].set(d_far))
                nsp = jnp.where(overflow, sp, sp + 1)
                return (near, nsi, nsd, nsp, best_d, best_cp, best_pid,
                        jnp.bool_(overflow))

            def no_push_go(_):
                return (near, si, sd, sp, best_d, best_cp, best_pid,
                        jnp.bool_(False))

            return lax.cond(push_far, push_then_go, no_push_go, operand=None)

        def pop_path(_):
            return _pop(si, sd, sp, best_d, best_cp, best_pid)

        return lax.cond(do_descend & any_child, descend_path, pop_path,
                        operand=None)

    final = lax.while_loop(cond_fn, body_fn, init)
    _cur, _si, _sd, _sp, best_d2, best_cp, best_pid, _done = final
    best_dist = jnp.sqrt(jnp.maximum(best_d2, jnp.float32(0.0)))
    return best_cp, best_dist, best_pid


# =====================================================================
# first_hit — ray descent. AABB slab test prunes subtrees; per-node ray-
# primitive test updates the running `tmax`. Mirrors the note's stack-
# based traversal but with ray-AABB culling on the subtree box.
# =====================================================================


def _first_hit_single(payload: KdPayload, primitives: Primitives,
                      ray_origin: jnp.ndarray, ray_direction: jnp.ndarray,
                      max_t: jnp.ndarray, max_stack_depth: int):
    INF = jnp.float32(jnp.inf)
    # Sign-preserving guard — same fix as `bvh.lbvh.jax` to avoid flipping
    # near-zero ray components.
    safe_dir = jnp.where(jnp.abs(ray_direction) < 1e-20,
                         jnp.copysign(jnp.float32(1e-20), ray_direction),
                         ray_direction)
    rcp_dir = jnp.float32(1.0) / safe_dir
    tmin = jnp.float32(0.0)

    stack_idx_init = jnp.full((max_stack_depth,), -1, dtype=_INDEX_DTYPE)
    stack_t_init = jnp.full((max_stack_depth,), INF, dtype=jnp.float32)

    init = (payload.root.astype(_INDEX_DTYPE),
            stack_idx_init, stack_t_init,
            jnp.asarray(0, dtype=_INDEX_DTYPE),
            max_t.astype(jnp.float32),
            jnp.asarray(-1, dtype=_INDEX_DTYPE),
            jnp.bool_(False),
            jnp.bool_(False))

    def cond_fn(s):
        return jnp.logical_not(s[7])

    def _pop_ray(si, st, sp, cur_tmax, best_pid, hit):
        def pop_cond(s):
            _ssp, _ssi, _sst, found, empty = s
            return jnp.logical_not(found | empty)

        def pop_body(s):
            ssp, ssi, sst, _f, _e = s
            empty = ssp == 0
            new_sp = jnp.where(empty, ssp, ssp - 1)
            top_t = sst[new_sp]
            in_range = (top_t < cur_tmax) & jnp.logical_not(empty)
            return new_sp, ssi, sst, in_range, empty

        new_sp, _, _, found, _empty = lax.while_loop(
            pop_cond, pop_body,
            (sp, si, st, jnp.bool_(False), jnp.bool_(False)))
        next_node = jnp.where(found, si[new_sp], jnp.asarray(0, _INDEX_DTYPE))
        done = jnp.logical_not(found)
        return next_node, si, st, new_sp, cur_tmax, best_pid, hit, done

    def body_fn(s):
        cur, si, st, sp, cur_tmax, best_pid, hit, _done = s

        # (a) Test this node's primitive.
        pid = payload.prim_ids[cur]
        t_k, hit_k = primitives.ray_primitive_intersect(
            ray_origin, ray_direction, cur_tmax, pid)
        accept = hit_k & (t_k < cur_tmax) & (t_k >= tmin)
        cur_tmax = jnp.where(accept, t_k, cur_tmax)
        best_pid = jnp.where(accept, pid, best_pid)
        hit = hit | accept

        # (b) Slab-test children, descend near-first.
        l = payload.node_left[cur]
        r = payload.node_right[cur]
        has_l = l >= 0
        has_r = r >= 0
        safe_l = jnp.where(has_l, l, cur)
        safe_r = jnp.where(has_r, r, cur)
        t_l, h_l = _ray_box_t(ray_origin, rcp_dir,
                              payload.node_lower[safe_l],
                              payload.node_upper[safe_l],
                              tmin, cur_tmax)
        t_r, h_r = _ray_box_t(ray_origin, rcp_dir,
                              payload.node_lower[safe_r],
                              payload.node_upper[safe_r],
                              tmin, cur_tmax)
        live_l = has_l & h_l
        live_r = has_r & h_r

        def both_miss(_):
            return _pop_ray(si, st, sp, cur_tmax, best_pid, hit)

        def at_least_one(_):
            l_first = jnp.where(live_l & live_r, t_l <= t_r,
                                live_l)  # if only one is live, descend it.
            near = jnp.where(l_first, safe_l, safe_r)
            far = jnp.where(l_first, safe_r, safe_l)
            far_t = jnp.where(l_first, t_r, t_l)
            push_far = live_l & live_r

            def do_push(_):
                overflow = sp >= max_stack_depth
                nsi = jnp.where(overflow, si, si.at[sp].set(far))
                nst = jnp.where(overflow, st, st.at[sp].set(far_t))
                nsp = jnp.where(overflow, sp, sp + 1)
                return (near, nsi, nst, nsp, cur_tmax, best_pid, hit,
                        jnp.bool_(overflow))

            def no_push(_):
                return (near, si, st, sp, cur_tmax, best_pid, hit,
                        jnp.bool_(False))

            return lax.cond(push_far, do_push, no_push, operand=None)

        return lax.cond(live_l | live_r, at_least_one, both_miss,
                        operand=None)

    final = lax.while_loop(cond_fn, body_fn, init)
    _cur, _si, _st, _sp, cur_tmax, best_pid, hit, _done = final
    return cur_tmax, best_pid, hit


# =====================================================================
# aabb_overlap — descend prunes by subtree-AABB vs query-AABB; per-node
# leaf test = primitive AABB vs query AABB.
# =====================================================================


def _aabb_overlap_single(payload: KdPayload, primitives: Primitives,
                         q_lo: jnp.ndarray, q_hi: jnp.ndarray,
                         max_stack_depth: int) -> jnp.ndarray:
    stack_init = jnp.full((max_stack_depth,), -1, dtype=_INDEX_DTYPE)

    init = (payload.root.astype(_INDEX_DTYPE),
            stack_init,
            jnp.asarray(0, dtype=_INDEX_DTYPE),
            jnp.bool_(False),
            jnp.bool_(False))

    def cond_fn(s):
        return jnp.logical_not(s[3] | s[4])

    def _pop_box(st, sp):
        empty = sp == 0
        new_sp = jnp.where(empty, sp, sp - 1)
        next_node = jnp.where(empty, jnp.asarray(0, _INDEX_DTYPE),
                              st[new_sp])
        return next_node, st, new_sp, jnp.bool_(False), empty

    def body_fn(s):
        cur, st, sp, _found, _done = s

        # Subtree-AABB cull comes from `node_lower/upper`. If we got here
        # the parent's overlap test already passed; still test self-AABB
        # (cheap) plus this node's primitive AABB.
        # Test this node's primitive AABB vs query.
        pid = payload.prim_ids[cur]
        p_lo, p_hi = primitives.primitive_aabb(pid)
        own_hit = _box_overlap(p_lo, p_hi, q_lo, q_hi)

        l = payload.node_left[cur]
        r = payload.node_right[cur]
        has_l = l >= 0
        has_r = r >= 0
        safe_l = jnp.where(has_l, l, cur)
        safe_r = jnp.where(has_r, r, cur)
        o_l = has_l & _box_overlap(payload.node_lower[safe_l],
                                   payload.node_upper[safe_l],
                                   q_lo, q_hi)
        o_r = has_r & _box_overlap(payload.node_lower[safe_r],
                                   payload.node_upper[safe_r],
                                   q_lo, q_hi)

        def short_done(_):
            return (jnp.asarray(0, _INDEX_DTYPE), st, sp,
                    jnp.bool_(True), jnp.bool_(True))

        def keep_going(_):
            def both_miss(_):
                return _pop_box(st, sp)

            def at_least(_):
                next_node = jnp.where(o_l, safe_l, safe_r)

                def push_r(_):
                    overflow = sp >= max_stack_depth
                    nst = jnp.where(overflow, st, st.at[sp].set(safe_r))
                    nsp = jnp.where(overflow, sp, sp + 1)
                    return (next_node, nst, nsp, jnp.bool_(False),
                            jnp.bool_(overflow))

                def no_push(_):
                    return (next_node, st, sp, jnp.bool_(False),
                            jnp.bool_(False))

                return lax.cond(o_l & o_r, push_r, no_push, operand=None)

            return lax.cond(o_l | o_r, at_least, both_miss, operand=None)

        return lax.cond(own_hit, short_done, keep_going, operand=None)

    final = lax.while_loop(cond_fn, body_fn, init)
    return final[3]


# =====================================================================
# distance_bounds — query AABB to mesh distance bounds.
# Wald left-balanced topology: every node owns exactly one primitive
# (no leaf-vs-interior branch, no empty-leaf path). Children of node i
# are stored at node_left[i]/node_right[i] (-1 == out of range).
# Stack-descent: at each visit, tighten best_hi from the node's own
# AABB-vs-query bounds, prune subtree if d_lo_node > best_hi, then test
# the node's primitive AABB and finally push existing children.
# =====================================================================


def _distance_bounds_single(payload: KdPayload, primitives: Primitives,
                            q_lo: jnp.ndarray, q_hi: jnp.ndarray,
                            max_stack_depth: int
                            ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    INF = jnp.float32(jnp.inf)
    stack_init = jnp.full((max_stack_depth,), -1, dtype=_INDEX_DTYPE)

    # Initial best_hi from root box; best_lo is INF (running MIN over
    # per-prim lower bounds).
    root = payload.root.astype(_INDEX_DTYPE)
    _root_lo, root_hi = _box_box_dist_bounds(
        q_lo, q_hi, payload.node_lower[root], payload.node_upper[root])

    # state: (cur, stack, sp, best_lo, best_hi, done)
    init = (root,
            stack_init,
            jnp.asarray(0, dtype=_INDEX_DTYPE),
            INF,
            root_hi,
            jnp.bool_(False))

    def cond_fn(s):
        return jnp.logical_not(s[5])

    def _pop(st, sp, best_lo, best_hi):
        empty = sp == 0
        new_sp = jnp.where(empty, sp, sp - 1)
        next_node = jnp.where(empty, jnp.asarray(0, _INDEX_DTYPE),
                              st[new_sp])
        done = empty
        return next_node, st, new_sp, best_lo, best_hi, done

    def body_fn(s):
        cur, st, sp, best_lo, best_hi, _done = s
        # (a) Subtree-AABB bounds: tighten best_hi, prune if d_lo > best_hi.
        d_lo_cur, d_hi_cur = _box_box_dist_bounds(
            q_lo, q_hi, payload.node_lower[cur], payload.node_upper[cur])
        best_hi = jnp.minimum(best_hi, d_hi_cur)
        skip = d_lo_cur > best_hi

        def skip_branch(_):
            return _pop(st, sp, best_lo, best_hi)

        def visit(_):
            # (c) Test the node's own primitive AABB — every node owns one
            # in Wald left-balanced layout (no empty-leaf path).
            pid = payload.prim_ids[cur]
            p_lo, p_hi = primitives.primitive_aabb(pid)
            p_d_lo, p_d_hi = _box_box_dist_bounds(q_lo, q_hi, p_lo, p_hi)
            new_best_lo = jnp.minimum(best_lo, p_d_lo)
            new_best_hi = jnp.minimum(best_hi, p_d_hi)

            # (d) Push children (if not -1).
            l = payload.node_left[cur]
            r = payload.node_right[cur]
            has_l = l >= 0
            has_r = r >= 0
            safe_l = jnp.where(has_l, l, cur)
            safe_r = jnp.where(has_r, r, cur)
            d0_lo, d0_hi = _box_box_dist_bounds(
                q_lo, q_hi,
                payload.node_lower[safe_l], payload.node_upper[safe_l])
            d1_lo, d1_hi = _box_box_dist_bounds(
                q_lo, q_hi,
                payload.node_lower[safe_r], payload.node_upper[safe_r])
            tighter_hi = jnp.minimum(
                jnp.minimum(new_best_hi,
                            jnp.where(has_l, d0_hi, INF)),
                jnp.where(has_r, d1_hi, INF))
            hit0 = has_l & (d0_lo <= tighter_hi)
            hit1 = has_r & (d1_lo <= tighter_hi)

            def both(_):
                left_first = d0_lo <= d1_lo
                near = jnp.where(left_first, safe_l, safe_r)
                far = jnp.where(left_first, safe_r, safe_l)
                overflow = sp >= max_stack_depth
                nst = jnp.where(overflow, st, st.at[sp].set(far))
                nsp = jnp.where(overflow, sp, sp + 1)
                return (near, nst, nsp, new_best_lo, tighter_hi,
                        jnp.bool_(overflow))

            def only_l(_):
                return (safe_l, st, sp, new_best_lo, tighter_hi,
                        jnp.bool_(False))

            def only_r(_):
                return (safe_r, st, sp, new_best_lo, tighter_hi,
                        jnp.bool_(False))

            def none(_):
                return _pop(st, sp, new_best_lo, tighter_hi)

            return lax.cond(
                hit0 & hit1, both,
                lambda _: lax.cond(
                    hit0, only_l,
                    lambda __: lax.cond(hit1, only_r, none,
                                        operand=None),
                    operand=None),
                operand=None)

        return lax.cond(skip, skip_branch, visit, operand=None)

    final = lax.while_loop(cond_fn, body_fn, init)
    _cur, _st, _sp, best_lo, best_hi, _done = final
    # If the descent never visited any node's primitive (degenerate edge
    # case — single-node tree fully pruned, impossible since root is
    # always visited), clamp INF to 0.
    best_lo = jnp.where(jnp.isinf(best_lo), jnp.float32(0.0), best_lo)
    return best_lo.astype(jnp.float32), best_hi.astype(jnp.float32)


# =====================================================================
# Backend class.
# =====================================================================


@register_backend
class KdLeftBalancedJaxBackend(Backend[KdPayload]):
    """Wald-style left-balanced kd-tree, adapted to extent-having
    primitives: every node owns one primitive AND splits, with a
    propagated subtree AABB for slab/distance pruning. Round-robin
    split axis (level % 3); pure-JAX traversal kernels.
    """

    name = "kd.left_balanced.jax"
    flavor = "kd"
    algorithm = "left_balanced"
    runtime = "jax"
    payload_type = KdPayload
    is_differentiable = False
    has_distance_bounds = True

    def __init__(self):
        self._primitives: Primitives | None = None
        self._max_stack_depth: int = 32

    def build(self, primitives: Primitives, config: Config) -> KdPayload:
        self._primitives = primitives
        n = int(primitives.n_primitives)
        idx = jnp.arange(n, dtype=_INDEX_DTYPE)
        lo, hi = primitives.primitive_aabb(idx)
        prim_lo_np = np.asarray(lo, dtype=np.float32)
        prim_hi_np = np.asarray(hi, dtype=np.float32)

        # Stack depth: a left-balanced tree of N nodes has depth
        # `floor(log2(N))`; `2 * depth + 4` gives slack for far-child
        # pushes plus a couple of extra slots, capped against the
        # user's `max_stack_depth` request.
        depth = max(1, int(math.floor(math.log2(max(n, 1)))) if n > 0 else 1)
        needed = 2 * depth + 4
        self._max_stack_depth = max(int(config.max_stack_depth), needed)
        return _build_payload(prim_lo_np, prim_hi_np)

    def closest_point(self, payload: KdPayload, x: jnp.ndarray
                      ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        prims = self._primitives
        max_stack = self._max_stack_depth

        def single(xq):
            return _closest_point_single(payload, prims, xq, max_stack)

        if x.ndim == 1:
            return single(x)
        return jax.vmap(single)(x)

    def first_hit(self, payload: KdPayload, ray_origin: jnp.ndarray,
                  ray_direction: jnp.ndarray, max_t: jnp.ndarray
                  ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        prims = self._primitives
        max_stack = self._max_stack_depth

        def single(o, d, mt):
            return _first_hit_single(payload, prims, o, d, mt, max_stack)

        if ray_origin.ndim == 1:
            return single(ray_origin, ray_direction, max_t)
        return jax.vmap(single)(ray_origin, ray_direction, max_t)

    def aabb_overlap(self, payload: KdPayload,
                     lower: jnp.ndarray, upper: jnp.ndarray) -> jnp.ndarray:
        prims = self._primitives
        max_stack = self._max_stack_depth

        def single(lo, hi):
            return _aabb_overlap_single(payload, prims, lo, hi, max_stack)

        if lower.ndim == 1:
            return single(lower, upper)
        return jax.vmap(single)(lower, upper)

    def distance_bounds(self, payload: KdPayload,
                        lower: jnp.ndarray, upper: jnp.ndarray
                        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        prims = self._primitives
        max_stack = self._max_stack_depth

        def single(lo, hi):
            return _distance_bounds_single(payload, prims, lo, hi, max_stack)

        if lower.ndim == 1:
            return single(lower, upper)
        return jax.vmap(single)(lower, upper)
