"""Octree builder — 8-way split, two driver paths.

Per surface, picks one:
  * `classify_box` path (NIQ default): classify cells, split UNKNOWN ones.
  * `classify_subcubes` path (Option A): per parent, classify all 8 children via parent polynomial.

Detection via `uses_classify_subcubes_override(surface)`. Only surfaces with a
non-default `classify_subcubes` (autobound) take the Option-A fast path.
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

from meshExtractor.base import ImplicitTree, Leaves
from meshExtractor.trees._bucketed_build import (
    BUCKET_SIZES,
    build_bucketed_tree_classify_box,
    build_bucketed_tree_classify_subcubes,
    uses_classify_subcubes_override,
)


# (i, j, k) ∈ {0, 1}^3 octant offsets.
_OCTANT_OFFSETS = jnp.asarray(
    np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], dtype=bool)
)


@functools.partial(jax.jit, static_argnames=("classify_fn", "do_split"),
                   donate_argnums=(7, 8, 9, 10))
def _iter_classify_box(classify_fn, params, do_split,
                       node_valid, node_lower, node_upper,
                       ib, out_valid, out_lower, out_upper, out_n_valid):
    """One batch: classify cells via classify_box, split UNKNOWN cells 8-ways if do_split."""
    signs = jax.vmap(lambda lo, hi: classify_fn(params, lo, hi))(node_lower, node_upper)
    split_mask = node_valid & (signs == 0)
    n_unknown = jnp.sum(split_mask)

    if do_split:
        mid = 0.5 * (node_lower + node_upper)
        child_lower = jnp.where(_OCTANT_OFFSETS[None, :, :], mid[:, None, :], node_lower[:, None, :])
        child_upper = jnp.where(_OCTANT_OFFSETS[None, :, :], node_upper[:, None, :], mid[:, None, :])
        new_lower = child_lower.reshape(-1, 3)
        new_upper = child_upper.reshape(-1, 3)
        new_valid = jnp.repeat(split_mask, 8)
        new_n_valid = 8 * n_unknown
        outL = out_valid.shape[1]  # 8 * B
    else:
        new_lower = node_lower
        new_upper = node_upper
        new_valid = split_mask
        new_n_valid = n_unknown
        outL = node_valid.shape[0]  # B

    out_valid = out_valid.at[ib, :outL].set(new_valid)
    out_lower = out_lower.at[ib, :outL, :].set(new_lower)
    out_upper = out_upper.at[ib, :outL, :].set(new_upper)
    return out_valid, out_lower, out_upper, out_n_valid + new_n_valid


@functools.partial(jax.jit, static_argnames=("classify_subcubes_fn",),
                   donate_argnums=(6, 7, 8, 9))
def _iter_classify_subcubes(classify_subcubes_fn, params,
                            node_valid, node_lower, node_upper,
                            ib, out_valid, out_lower, out_upper, out_n_valid):
    """One batch: per parent, build 8 octant children + classify via parent polynomial."""

    def per_parent(parent_valid, parent_lo, parent_hi):
        mid = 0.5 * (parent_lo + parent_hi)
        sub_lo = jnp.where(_OCTANT_OFFSETS, mid[None, :], parent_lo[None, :])
        sub_hi = jnp.where(_OCTANT_OFFSETS, parent_hi[None, :], mid[None, :])
        signs = classify_subcubes_fn(params, parent_lo, parent_hi, sub_lo, sub_hi)
        child_valid = parent_valid & (signs == 0)
        return sub_lo, sub_hi, child_valid

    sub_lo_batch, sub_hi_batch, child_valid_batch = jax.vmap(per_parent)(
        node_valid, node_lower, node_upper
    )
    new_lower = sub_lo_batch.reshape(-1, 3)
    new_upper = sub_hi_batch.reshape(-1, 3)
    new_valid = child_valid_batch.reshape(-1)
    new_n_valid = jnp.sum(new_valid)
    outL = out_valid.shape[1]

    out_valid = out_valid.at[ib, :outL].set(new_valid)
    out_lower = out_lower.at[ib, :outL, :].set(new_lower)
    out_upper = out_upper.at[ib, :outL, :].set(new_upper)
    return out_valid, out_lower, out_upper, out_n_valid + new_n_valid


class Octree(ImplicitTree):
    leaf_uniformity = "uniform"

    def __init__(self, max_depth: int, batch_process_size: int = 2048):
        if max_depth < 0:
            raise ValueError("max_depth must be non-negative")
        if batch_process_size not in BUCKET_SIZES:
            raise ValueError(f"batch_process_size must be one of {BUCKET_SIZES}")
        self.max_depth = max_depth
        self.batch_process_size = batch_process_size

    def build(self, surface, params, root_lower, root_upper) -> Leaves:
        if uses_classify_subcubes_override(surface):
            leaves, stats = build_bucketed_tree_classify_subcubes(
                iter_fn=_iter_classify_subcubes,
                split_factor=8,
                surface=surface, params=params,
                root_lower=root_lower, root_upper=root_upper,
                max_depth=self.max_depth,
                batch_process_size=self.batch_process_size,
            )
        else:
            leaves, stats = build_bucketed_tree_classify_box(
                iter_fn=_iter_classify_box,
                split_factor=8,
                surface=surface, params=params,
                root_lower=root_lower, root_upper=root_upper,
                max_depth=self.max_depth,
                batch_process_size=self.batch_process_size,
            )
        self.last_build_stats = stats
        return leaves
