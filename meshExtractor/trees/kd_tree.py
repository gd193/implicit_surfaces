"""Kd-tree builder — 2-way longest-axis split, two driver paths.

See `octree.py` for the dispatch rationale.

For a cubic root, KdTree(max_depth=3*N) is roughly equivalent to Octree(max_depth=N).
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

from meshExtractor.base import ImplicitTree, Leaves
from meshExtractor.trees._bucketed_build import (
    BUCKET_SIZES,
    build_bucketed_tree_classify_box,
    build_bucketed_tree_classify_subcubes,
    uses_classify_subcubes_override,
)
from surfaces.spatial_tree import SpatialBuildResult


@functools.partial(jax.jit, static_argnames=("classify_fn", "do_split"),
                   donate_argnums=(7, 8, 9, 10))
def _iter_classify_box(classify_fn, params, do_split,
                       node_valid, node_lower, node_upper,
                       ib, out_valid, out_lower, out_upper, out_n_valid):
    """One batch: classify_box + 2-way longest-axis split if do_split."""
    signs = jax.vmap(lambda lo, hi: classify_fn(params, lo, hi))(node_lower, node_upper)
    split_mask = node_valid & (signs == 0)
    n_unknown = jnp.sum(split_mask)

    if do_split:
        mid = 0.5 * (node_lower + node_upper)
        extent = node_upper - node_lower
        split_axis = jnp.argmax(extent, axis=-1)
        axis_mask = jnp.arange(3)[None, :] == split_axis[:, None]
        childA_upper = jnp.where(axis_mask, mid, node_upper)
        childB_lower = jnp.where(axis_mask, mid, node_lower)
        new_lower = jnp.concatenate([node_lower, childB_lower], axis=0)
        new_upper = jnp.concatenate([childA_upper, node_upper], axis=0)
        new_valid = jnp.concatenate([split_mask, split_mask], axis=0)
        new_n_valid = 2 * n_unknown
        outL = out_valid.shape[1]  # 2 * B
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
    """One batch: per parent, 2 longest-axis-split children classified via parent polynomial."""

    def per_parent(parent_valid, parent_lo, parent_hi):
        mid = 0.5 * (parent_lo + parent_hi)
        extent = parent_hi - parent_lo
        split_axis = jnp.argmax(extent, axis=-1)
        axis_mask = jnp.arange(3) == split_axis
        childA_upper = jnp.where(axis_mask, mid, parent_hi)
        childB_lower = jnp.where(axis_mask, mid, parent_lo)
        sub_lo = jnp.stack([parent_lo, childB_lower], axis=0)
        sub_hi = jnp.stack([childA_upper, parent_hi], axis=0)
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


class KdTree(ImplicitTree):
    leaf_uniformity = "uniform"

    def __init__(self, max_depth: int, batch_process_size: int = 2048):
        if max_depth < 0:
            raise ValueError("max_depth must be non-negative")
        if batch_process_size not in BUCKET_SIZES:
            raise ValueError(f"batch_process_size must be one of {BUCKET_SIZES}")
        self.max_depth = max_depth
        self.batch_process_size = batch_process_size

    def build(self, surface, params, root_lower, root_upper) -> Leaves:
        leaves, stats = self._dispatch(
            surface, params, root_lower, root_upper, record_topology=False,
        )
        self.last_build_stats = stats
        return leaves

    def build_with_topology(self, surface, params, root_lower, root_upper) -> SpatialBuildResult:
        """Build the leaf cover and the full subdivision tree topology.

        Same per-iter classifier cost as `build`, plus a recorder pass per
        batch. Build is one-shot, not on the hot query path.
        """
        result, stats = self._dispatch(
            surface, params, root_lower, root_upper, record_topology=True,
        )
        self.last_build_stats = stats
        return result

    def _dispatch(self, surface, params, root_lower, root_upper, *, record_topology: bool):
        if uses_classify_subcubes_override(surface):
            return build_bucketed_tree_classify_subcubes(
                iter_fn=_iter_classify_subcubes,
                split_factor=2,
                surface=surface, params=params,
                root_lower=root_lower, root_upper=root_upper,
                max_depth=self.max_depth,
                batch_process_size=self.batch_process_size,
                record_topology=record_topology, flavor="kd",
                child_layout="grouped",   # vmap+stack over (parent, K, ...)
            )
        return build_bucketed_tree_classify_box(
            iter_fn=_iter_classify_box,
            split_factor=2,
            surface=surface, params=params,
            root_lower=root_lower, root_upper=root_upper,
            max_depth=self.max_depth,
            batch_process_size=self.batch_process_size,
            record_topology=record_topology, flavor="kd",
            child_layout="concat",        # `concatenate([childA, childB])` packing
        )
