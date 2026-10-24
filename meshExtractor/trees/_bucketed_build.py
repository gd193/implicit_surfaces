"""Shared scaffolding for bucketed-batch tree construction.

Two driver paths, picked per surface:

  * **classify_box path** (`build_bucketed_tree_classify_box`) — NIQ's structure.
    Each iteration classifies the current frontier cells via `surface.classify_box`,
    then splits UNKNOWN cells. The last iteration is no-split. Used for surfaces that
    don't gain anything from the parent-polynomial trick (every classify still costs
    one full forward pass): NIQ neural, harmonic, plain SDFs.

  * **classify_subcubes path** (`build_bucketed_tree_classify_subcubes`) — Option A.
    Each iteration takes UNKNOWN parents and classifies all `split_factor` children of
    each parent via `surface.classify_subcubes` — one parent-level forward pass yields
    N child classifications. Used for surfaces that override `classify_subcubes` with
    a polynomial-derived path (autobound surfaces).

Both share the JIT-cache discipline (static bucket sizes, donated output buffers) and
the same outer/inner Python loop structure; they differ only in what the per-batch
kernel does and the loop's iteration count (`max_depth + 1` vs `max_depth`).

Topology recording (`record_topology=True`):
    Drivers preallocate tree arrays sized via `estimate_max_nodes` and thread a
    `_TopologyRecorder` alongside the existing frontier state. Each iter:
    classifications scatter into `leaf_classification`; splits append children
    and link them via `parent` / `children`. Frontier carries a parallel
    `node_ids` array tracked through compaction. Host-side `n_used` counter
    rides the existing per-iter sync used by `get_next_bucket_size`. No trim
    on output — consumers slice with `n_nodes`. Returns SpatialBuildResult.
"""
from __future__ import annotations

import functools
import math
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp

from surfaces.implicit.base import implicitSurface
from meshExtractor.base import Leaves
from surfaces.spatial_tree import SpatialTree, SpatialBuildResult, estimate_max_nodes
from utils.jnp_array_compaction import compact_tree


# Bucket sizes: powers of 2 from 2^6 (64) up to 2^25 (~33M).
BUCKET_SIZES = [2 ** k for k in range(6, 26)]


def get_next_bucket_size(s: int) -> int:
    for b in BUCKET_SIZES:
        if s <= b:
            return b
    raise ValueError(f"max bucket size exceeded: {s}")


@functools.partial(jax.jit, static_argnames=("new_size",))
def compact_to_bucket(state, is_active, new_size: int):
    """JIT'd `compact_tree` wrapper; `new_size` is static so cache keys by bucket size."""
    new_state, valid_mask, _ = compact_tree(state, is_active, new_size)
    return new_state, valid_mask


# ===========================================================================
# Topology recording (opt-in via record_topology=True).
#
# Tree arrays preallocated up front to `estimate_max_nodes(...)`. Two helpers
# below scatter per-iter records into them; both are pure functions of the
# current tree arrays + frontier batch + `n_used` *write base offset*. The
# driver tracks `n_used` host-side, increments it by `int(jnp.sum(out_valid))`
# per split iter (rides the existing per-iter bucket-size sync). Helpers JIT-
# cache by frontier shape only.
# ===========================================================================


def _child_layout_indices(node_ids_flat: jnp.ndarray, K: int, child_layout: str):
    """Per-child (parent_id, k_slot) given the iter_fn's child packing.

    Two layouts in use:
      "grouped"  — `[p0_c0..p0_c{K-1}, p1_c0..p1_c{K-1}, ...]` (octree, all
                   `classify_subcubes` paths). parent_ids_per_child = repeat;
                   k_slots = tile(arange(K), P).
      "concat"   — `[c0_p0..c0_p{P-1}, c1_p0..c1_p{P-1}, ...]` (kd_tree
                   classify_box). parent_ids_per_child = tile(., K); k_slots =
                   repeat(arange(K), P).
    """
    P = node_ids_flat.shape[0]
    if child_layout == "grouped":
        parent_ids_per_child = jnp.repeat(node_ids_flat, K)
        k_slots = jnp.tile(jnp.arange(K, dtype=jnp.int32), P)
    elif child_layout == "concat":
        parent_ids_per_child = jnp.tile(node_ids_flat, K)
        k_slots = jnp.repeat(jnp.arange(K, dtype=jnp.int32), P)
    else:
        raise ValueError(f"unknown child_layout: {child_layout!r}")
    return parent_ids_per_child, k_slots


@functools.partial(jax.jit,
                   static_argnames=("classify_fn", "do_split", "K", "N_max", "child_layout"))
def _record_classify_box(tree_aabb_lo, tree_aabb_hi, tree_parent, tree_children,
                          tree_is_leaf, tree_classification,
                          classify_fn, params, do_split: bool, K: int, N_max: int,
                          child_layout: str,
                          n_used_base: jnp.ndarray,
                          node_ids_flat, node_valid_flat, node_lower_flat, node_upper_flat,
                          out_valid_flat, out_lower_flat, out_upper_flat):
    """Record one classify_box iter's worth of tree state.

    Reclassifies the frontier (the iter_fn already did this internally; the
    extra vmap is the price we pay for keeping iter_fn untouched — build is
    one-shot, not on the hot path).

    `child_layout` selects how the iter_fn packed its children — see
    `_child_layout_indices`.

    Returns the updated tree arrays and a `child_global_ids_flat` array
    aligned with `out_valid_flat` (sentinel `-1` where invalid). When
    `do_split=False` (terminal iter), child outputs are the surviving
    parents themselves and we just propagate node_ids.
    """
    signs = jax.vmap(lambda lo, hi: classify_fn(params, lo, hi))(node_lower_flat, node_upper_flat)
    signs_i8 = signs.astype(jnp.int8)

    # 1) write classifications at frontier entries
    write_idx = jnp.where(node_valid_flat, node_ids_flat, N_max + 1)
    tree_classification = tree_classification.at[write_idx].set(signs_i8, mode='drop')

    if do_split:
        # cumsum gives compact 0..n_valid-1 indices for valid children; invalids alias OOB
        cumsum = jnp.cumsum(out_valid_flat.astype(jnp.int32)) - 1
        child_global_ids = n_used_base + cumsum
        write_child_idx = jnp.where(out_valid_flat, child_global_ids, N_max + 1)

        tree_aabb_lo = tree_aabb_lo.at[write_child_idx].set(out_lower_flat, mode='drop')
        tree_aabb_hi = tree_aabb_hi.at[write_child_idx].set(out_upper_flat, mode='drop')

        parent_ids_per_child, k_slots = _child_layout_indices(node_ids_flat, K, child_layout)
        tree_parent = tree_parent.at[write_child_idx].set(parent_ids_per_child, mode='drop')

        # Gate parent index too — padded frontier slots carry valid-but-bogus
        # ids (from compaction's gather-from-0), so an ungated scatter would
        # overwrite real children of node-id-0 with -1.
        parent_idx_for_write = jnp.where(
            out_valid_flat, parent_ids_per_child, jnp.int32(N_max + 1)
        )
        tree_children = tree_children.at[parent_idx_for_write, k_slots].set(
            child_global_ids, mode='drop')

        # parents that got split: valid AND classified UNKNOWN
        parents_split_mask = node_valid_flat & (signs_i8 == 0)
        flip_idx = jnp.where(parents_split_mask, node_ids_flat, N_max + 1)
        tree_is_leaf = tree_is_leaf.at[flip_idx].set(False, mode='drop')

        out_node_ids = jnp.where(out_valid_flat, child_global_ids, jnp.int32(-1))
    else:
        # terminal iter — out arrays are the surviving (UNKNOWN-classified) parents
        # themselves, in the same order. Their node ids carry through unchanged.
        out_node_ids = jnp.where(out_valid_flat, node_ids_flat, jnp.int32(-1))

    return (tree_aabb_lo, tree_aabb_hi, tree_parent, tree_children,
            tree_is_leaf, tree_classification, out_node_ids)


@functools.partial(jax.jit,
                   static_argnames=("classify_subcubes_fn", "K", "N_max", "child_layout"))
def _record_classify_subcubes(tree_aabb_lo, tree_aabb_hi, tree_parent, tree_children,
                               tree_is_leaf, tree_classification,
                               classify_subcubes_fn, params, K: int, N_max: int,
                               child_layout: str,
                               n_used_base: jnp.ndarray,
                               node_ids_flat, node_valid_flat, node_lower_flat, node_upper_flat,
                               out_valid_flat, out_lower_flat, out_upper_flat):
    """Record one classify_subcubes iter. Recomputes per-child signs by
    re-running `classify_subcubes_fn` per parent — same parent-polynomial
    cost the iter_fn already paid.

    Tree-node membership for children = `parent_valid` (all children of a
    valid parent are real tree nodes). `out_valid_flat` is the *next-
    frontier* mask (subset of tree nodes whose sign is UNKNOWN); we use it
    only for the propagated `out_node_ids`.
    """
    sub_lo = out_lower_flat.reshape(node_ids_flat.shape[0], K, 3)
    sub_hi = out_upper_flat.reshape(node_ids_flat.shape[0], K, 3)

    def per_parent(parent_lo, parent_hi, slo, shi):
        return classify_subcubes_fn(params, parent_lo, parent_hi, slo, shi)

    child_signs = jax.vmap(per_parent)(node_lower_flat, node_upper_flat, sub_lo, sub_hi)
    child_signs_flat = child_signs.reshape(-1).astype(jnp.int8)

    # Real-child mask depends on layout (which dimension repeats parents).
    if child_layout == "grouped":
        real_child_mask = jnp.repeat(node_valid_flat, K)
    else:  # "concat"
        real_child_mask = jnp.tile(node_valid_flat, K)

    cumsum = jnp.cumsum(real_child_mask.astype(jnp.int32)) - 1
    child_global_ids = n_used_base + cumsum
    write_child_idx = jnp.where(real_child_mask, child_global_ids, N_max + 1)

    tree_aabb_lo = tree_aabb_lo.at[write_child_idx].set(out_lower_flat, mode='drop')
    tree_aabb_hi = tree_aabb_hi.at[write_child_idx].set(out_upper_flat, mode='drop')
    tree_classification = tree_classification.at[write_child_idx].set(child_signs_flat, mode='drop')

    parent_ids_per_child, k_slots = _child_layout_indices(node_ids_flat, K, child_layout)
    tree_parent = tree_parent.at[write_child_idx].set(parent_ids_per_child, mode='drop')

    parent_idx_for_write = jnp.where(
        real_child_mask, parent_ids_per_child, jnp.int32(N_max + 1)
    )
    tree_children = tree_children.at[parent_idx_for_write, k_slots].set(
        child_global_ids, mode='drop')

    # Every valid parent in this batch is split (its children are now in tree).
    flip_idx = jnp.where(node_valid_flat, node_ids_flat, N_max + 1)
    tree_is_leaf = tree_is_leaf.at[flip_idx].set(False, mode='drop')

    # Propagate ids to the next frontier — only UNKNOWN children stay active.
    out_node_ids = jnp.where(out_valid_flat, child_global_ids, jnp.int32(-1))

    return (tree_aabb_lo, tree_aabb_hi, tree_parent, tree_children,
            tree_is_leaf, tree_classification, out_node_ids)


def _allocate_tree_arrays(N_max: int, K: int):
    return dict(
        aabb_lo=jnp.zeros((N_max, 3), dtype=jnp.float32),
        aabb_hi=jnp.zeros((N_max, 3), dtype=jnp.float32),
        parent=jnp.full((N_max,), -1, dtype=jnp.int32),
        children=jnp.full((N_max, K), -1, dtype=jnp.int32),
        is_leaf=jnp.ones((N_max,), dtype=jnp.bool_),
        classification=jnp.zeros((N_max,), dtype=jnp.int8),
    )


def _finalize_tree(arrs: dict, n_nodes: int, flavor: str, K: int) -> SpatialTree:
    return SpatialTree(
        aabb_lo=arrs['aabb_lo'], aabb_hi=arrs['aabb_hi'],
        parent=arrs['parent'], children=arrs['children'],
        is_leaf=arrs['is_leaf'], leaf_classification=arrs['classification'],
        n_nodes=n_nodes, flavor=flavor, K=K,
    )


def uses_classify_subcubes_override(surface) -> bool:
    """Pick the parent-classifies-children driver for this surface?

    Reads `surface.use_classify_subcubes` (class- or instance-level attribute, default False
    on the ABC). Set True only when the surface's `classify_subcubes` override is meaningfully
    tighter than N independent `classify_box` calls — i.e. it derives child signs from a
    cached parent model instead of running the model N times.
    """
    return bool(getattr(surface, "use_classify_subcubes", False))


def build_bucketed_tree_classify_box(iter_fn: Callable,
                                     split_factor: int,
                                     surface,
                                     params: Any,
                                     root_lower: jnp.ndarray,
                                     root_upper: jnp.ndarray,
                                     max_depth: int,
                                     batch_process_size: int,
                                     *,
                                     record_topology: bool = False,
                                     flavor: str = "octree",
                                     child_layout: str = "grouped",
                                     max_nodes: Optional[int] = None):
    """NIQ-style driver — classify-then-split. `max_depth + 1` iterations.

    Returns (leaves, stats) when `record_topology=False`. Returns
    (SpatialBuildResult, stats) when `record_topology=True` — the
    `leaves` field on the result matches the no-record return verbatim.

    `stats` is a list of per-iter dicts:
        {iter, this_b, n_occ, mlp_calls, n_unknown_after}.
    `mlp_calls` = n_occ * this_b is the number of `classify_box` invocations the JIT'd
    vmap actually ran at this iter — i.e. the number of MLP forward passes for this iter.
    """
    classify_fn = surface.classify_box
    B = batch_process_size
    stats = []

    node_lower = jnp.asarray(root_lower, dtype=jnp.float32)[None, :]
    node_upper = jnp.asarray(root_upper, dtype=jnp.float32)[None, :]
    node_valid = jnp.ones((1,), dtype=bool)
    N_curr = 1

    # Topology-recording state (only used when record_topology=True).
    if record_topology:
        N_max = max_nodes or estimate_max_nodes(max_depth, B, split_factor)
        tree_arrs = _allocate_tree_arrays(N_max, split_factor)
        # Root: id 0. Write its AABB.
        tree_arrs['aabb_lo'] = tree_arrs['aabb_lo'].at[0].set(node_lower[0])
        tree_arrs['aabb_hi'] = tree_arrs['aabb_hi'].at[0].set(node_upper[0])
        n_used = 1
        node_ids = jnp.zeros((1,), dtype=jnp.int32)  # parallel to frontier
    else:
        N_max = 0
        tree_arrs = None
        n_used = 0
        node_ids = None

    n_iters = max_depth + 1
    for i_split in range(n_iters):
        init_size = node_lower.shape[0]
        this_b = min(B, init_size)
        node_valid = node_valid.reshape(-1, this_b)
        node_lower = node_lower.reshape(-1, this_b, 3)
        node_upper = node_upper.reshape(-1, this_b, 3)
        if record_topology:
            node_ids = node_ids.reshape(-1, this_b)
        nb = node_lower.shape[0]
        n_occ = int(math.ceil(N_curr / this_b)) if N_curr > 0 else 0

        quit_next = (i_split + 1 == n_iters) or N_curr == 0
        do_split = not quit_next

        outL = split_factor * this_b if do_split else this_b
        out_valid = jnp.zeros((nb, outL), dtype=bool)
        out_lower = jnp.zeros((nb, outL, 3), dtype=jnp.float32)
        out_upper = jnp.zeros((nb, outL, 3), dtype=jnp.float32)
        total_n_valid = jnp.zeros((), dtype=jnp.int32)
        if record_topology:
            out_node_ids = jnp.full((nb, outL), -1, dtype=jnp.int32)
            # device-side running counter, threaded through the batch loop;
            # synced to host once at end of iter (rides bucket-size sync).
            running_n_used = jnp.asarray(n_used, dtype=jnp.int32)

        for ib in range(n_occ):
            out_valid, out_lower, out_upper, total_n_valid = iter_fn(
                classify_fn, params, do_split,
                node_valid[ib], node_lower[ib], node_upper[ib],
                ib, out_valid, out_lower, out_upper, total_n_valid,
            )
            if record_topology:
                rec_out = _record_classify_box(
                    tree_arrs['aabb_lo'], tree_arrs['aabb_hi'], tree_arrs['parent'],
                    tree_arrs['children'], tree_arrs['is_leaf'], tree_arrs['classification'],
                    classify_fn, params, do_split, split_factor, N_max, child_layout,
                    running_n_used,
                    node_ids[ib], node_valid[ib], node_lower[ib], node_upper[ib],
                    out_valid[ib], out_lower[ib], out_upper[ib],
                )
                (tree_arrs['aabb_lo'], tree_arrs['aabb_hi'], tree_arrs['parent'],
                 tree_arrs['children'], tree_arrs['is_leaf'],
                 tree_arrs['classification'], batch_out_node_ids) = rec_out
                out_node_ids = out_node_ids.at[ib].set(batch_out_node_ids)
                if do_split:
                    running_n_used = running_n_used + jnp.sum(out_valid[ib]).astype(jnp.int32)

        node_valid = out_valid.reshape(-1)
        node_lower = out_lower.reshape(-1, 3)
        node_upper = out_upper.reshape(-1, 3)
        if record_topology:
            node_ids = out_node_ids.reshape(-1)
            n_used = int(running_n_used)  # piggybacks the bucket-size sync below
        N_curr = int(total_n_valid)

        # 1 classify_box (= 1 MLP forward pass) per cell processed (incl. padding inside
        # an occupied batch). Padded batches (ib >= n_occ) are skipped, so the cost is
        # exactly n_occ * this_b.
        stats.append({
            "iter": i_split,
            "this_b": this_b,
            "n_occ": n_occ,
            "mlp_calls": n_occ * this_b,
            "n_unknown_after": N_curr,
        })

        if N_curr == 0:
            empty_leaves = Leaves(
                valid=jnp.zeros((0,), dtype=bool),
                lower=jnp.zeros((0, 3), dtype=jnp.float32),
                upper=jnp.zeros((0, 3), dtype=jnp.float32),
                uniformity="uniform",
            )
            if record_topology:
                tree = _finalize_tree(tree_arrs, n_used, flavor, split_factor)
                return SpatialBuildResult(
                    tree=tree, leaves=empty_leaves,
                    leaf_node_ids=jnp.zeros((0,), dtype=jnp.int32),
                ), stats
            return empty_leaves, stats

        target_size = get_next_bucket_size(N_curr)
        if record_topology:
            (node_lower, node_upper, node_ids), node_valid = compact_to_bucket(
                (node_lower, node_upper, node_ids), node_valid, target_size
            )
        else:
            (node_lower, node_upper), node_valid = compact_to_bucket(
                (node_lower, node_upper), node_valid, target_size
            )

        if quit_next:
            break

    final_leaves = Leaves(valid=node_valid, lower=node_lower, upper=node_upper, uniformity="uniform")
    if record_topology:
        # `node_ids` aligns with the surviving frontier 1:1; entries corresponding
        # to invalid frontier slots carry -1. Consumers slice by `Leaves.valid`.
        tree = _finalize_tree(tree_arrs, n_used, flavor, split_factor)
        return SpatialBuildResult(tree=tree, leaves=final_leaves, leaf_node_ids=node_ids), stats
    return final_leaves, stats


def build_bucketed_tree_classify_subcubes(iter_fn: Callable,
                                          split_factor: int,
                                          surface,
                                          params: Any,
                                          root_lower: jnp.ndarray,
                                          root_upper: jnp.ndarray,
                                          max_depth: int,
                                          batch_process_size: int,
                                          *,
                                          record_topology: bool = False,
                                          flavor: str = "octree",
                                          child_layout: str = "grouped",
                                          max_nodes: Optional[int] = None):
    """Option-A driver — per parent, classify all children via parent polynomial. `max_depth` iterations.

    Returns (leaves, stats) when `record_topology=False`. Returns
    (SpatialBuildResult, stats) when `record_topology=True`.

    One Taylor pass per parent ⇒ `mlp_calls = n_occ * this_b` (one
    MLP forward pass per parent, regardless of the `split_factor` children classified from it).
    """
    classify_subcubes_fn = surface.classify_subcubes
    B = batch_process_size
    stats = []

    node_lower = jnp.asarray(root_lower, dtype=jnp.float32)[None, :]
    node_upper = jnp.asarray(root_upper, dtype=jnp.float32)[None, :]
    node_valid = jnp.ones((1,), dtype=bool)
    N_curr = 1

    if record_topology:
        N_max = max_nodes or estimate_max_nodes(max_depth, B, split_factor)
        tree_arrs = _allocate_tree_arrays(N_max, split_factor)
        tree_arrs['aabb_lo'] = tree_arrs['aabb_lo'].at[0].set(node_lower[0])
        tree_arrs['aabb_hi'] = tree_arrs['aabb_hi'].at[0].set(node_upper[0])
        n_used = 1
        node_ids = jnp.zeros((1,), dtype=jnp.int32)
    else:
        N_max = 0
        tree_arrs = None
        n_used = 0
        node_ids = None

    for i_split in range(max_depth):
        if N_curr == 0:
            break

        init_size = node_lower.shape[0]
        this_b = min(B, init_size)
        node_valid = node_valid.reshape(-1, this_b)
        node_lower = node_lower.reshape(-1, this_b, 3)
        node_upper = node_upper.reshape(-1, this_b, 3)
        if record_topology:
            node_ids = node_ids.reshape(-1, this_b)
        nb = node_lower.shape[0]
        n_occ = int(math.ceil(N_curr / this_b))

        outL = split_factor * this_b
        out_valid = jnp.zeros((nb, outL), dtype=bool)
        out_lower = jnp.zeros((nb, outL, 3), dtype=jnp.float32)
        out_upper = jnp.zeros((nb, outL, 3), dtype=jnp.float32)
        total_n_valid = jnp.zeros((), dtype=jnp.int32)
        if record_topology:
            out_node_ids = jnp.full((nb, outL), -1, dtype=jnp.int32)
            running_n_used = jnp.asarray(n_used, dtype=jnp.int32)

        for ib in range(n_occ):
            out_valid, out_lower, out_upper, total_n_valid = iter_fn(
                classify_subcubes_fn, params,
                node_valid[ib], node_lower[ib], node_upper[ib],
                ib, out_valid, out_lower, out_upper, total_n_valid,
            )
            if record_topology:
                rec_out = _record_classify_subcubes(
                    tree_arrs['aabb_lo'], tree_arrs['aabb_hi'], tree_arrs['parent'],
                    tree_arrs['children'], tree_arrs['is_leaf'], tree_arrs['classification'],
                    classify_subcubes_fn, params, split_factor, N_max, child_layout,
                    running_n_used,
                    node_ids[ib], node_valid[ib], node_lower[ib], node_upper[ib],
                    out_valid[ib], out_lower[ib], out_upper[ib],
                )
                (tree_arrs['aabb_lo'], tree_arrs['aabb_hi'], tree_arrs['parent'],
                 tree_arrs['children'], tree_arrs['is_leaf'],
                 tree_arrs['classification'], batch_out_node_ids) = rec_out
                out_node_ids = out_node_ids.at[ib].set(batch_out_node_ids)
                # All children of valid parents are tree nodes (regardless of
                # child sign). Recorder uses `node_valid` repeated K times.
                running_n_used = running_n_used + (
                    split_factor * jnp.sum(node_valid[ib]).astype(jnp.int32)
                )

        node_valid = out_valid.reshape(-1)
        node_lower = out_lower.reshape(-1, 3)
        node_upper = out_upper.reshape(-1, 3)
        if record_topology:
            node_ids = out_node_ids.reshape(-1)
            n_used = int(running_n_used)
        N_curr = int(total_n_valid)

        stats.append({
            "iter": i_split,
            "this_b": this_b,
            "n_occ": n_occ,
            "mlp_calls": n_occ * this_b,
            "n_unknown_after": N_curr,
        })

        if N_curr == 0:
            empty_leaves = Leaves(
                valid=jnp.zeros((0,), dtype=bool),
                lower=jnp.zeros((0, 3), dtype=jnp.float32),
                upper=jnp.zeros((0, 3), dtype=jnp.float32),
                uniformity="uniform",
            )
            if record_topology:
                tree = _finalize_tree(tree_arrs, n_used, flavor, split_factor)
                return SpatialBuildResult(
                    tree=tree, leaves=empty_leaves,
                    leaf_node_ids=jnp.zeros((0,), dtype=jnp.int32),
                ), stats
            return empty_leaves, stats

        target_size = get_next_bucket_size(N_curr)
        if record_topology:
            (node_lower, node_upper, node_ids), node_valid = compact_to_bucket(
                (node_lower, node_upper, node_ids), node_valid, target_size
            )
        else:
            (node_lower, node_upper), node_valid = compact_to_bucket(
                (node_lower, node_upper), node_valid, target_size
            )

    final_leaves = Leaves(valid=node_valid, lower=node_lower, upper=node_upper, uniformity="uniform")
    if record_topology:
        tree = _finalize_tree(tree_arrs, n_used, flavor, split_factor)
        return SpatialBuildResult(tree=tree, leaves=final_leaves, leaf_node_ids=node_ids), stats
    return final_leaves, stats
