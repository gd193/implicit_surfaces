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
"""
from __future__ import annotations

import functools
import math
from typing import Any, Callable

import jax
import jax.numpy as jnp

from implicitSurfaces.implicit_surface import implicitSurface
from meshExtractor.base import Leaves
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
    new_state, valid_mask, _ = compact_tree(state, is_active, new_size, fill_index_value=0)
    return new_state, valid_mask


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
                                     batch_process_size: int):
    """NIQ-style driver — classify-then-split. `max_depth + 1` iterations.

    Returns (leaves, stats) where `stats` is a list of per-iter dicts:
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

    n_iters = max_depth + 1
    for i_split in range(n_iters):
        init_size = node_lower.shape[0]
        this_b = min(B, init_size)
        node_valid = node_valid.reshape(-1, this_b)
        node_lower = node_lower.reshape(-1, this_b, 3)
        node_upper = node_upper.reshape(-1, this_b, 3)
        nb = node_lower.shape[0]
        n_occ = int(math.ceil(N_curr / this_b)) if N_curr > 0 else 0

        quit_next = (i_split + 1 == n_iters) or N_curr == 0
        do_split = not quit_next

        outL = split_factor * this_b if do_split else this_b
        out_valid = jnp.zeros((nb, outL), dtype=bool)
        out_lower = jnp.zeros((nb, outL, 3), dtype=jnp.float32)
        out_upper = jnp.zeros((nb, outL, 3), dtype=jnp.float32)
        total_n_valid = jnp.zeros((), dtype=jnp.int32)

        for ib in range(n_occ):
            out_valid, out_lower, out_upper, total_n_valid = iter_fn(
                classify_fn, params, do_split,
                node_valid[ib], node_lower[ib], node_upper[ib],
                ib, out_valid, out_lower, out_upper, total_n_valid,
            )

        node_valid = out_valid.reshape(-1)
        node_lower = out_lower.reshape(-1, 3)
        node_upper = out_upper.reshape(-1, 3)
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
            return Leaves(
                valid=jnp.zeros((0,), dtype=bool),
                lower=jnp.zeros((0, 3), dtype=jnp.float32),
                upper=jnp.zeros((0, 3), dtype=jnp.float32),
                uniformity="uniform",
            ), stats

        target_size = get_next_bucket_size(N_curr)
        (node_lower, node_upper), node_valid = compact_to_bucket(
            (node_lower, node_upper), node_valid, target_size
        )

        if quit_next:
            break

    return Leaves(valid=node_valid, lower=node_lower, upper=node_upper, uniformity="uniform"), stats


def build_bucketed_tree_classify_subcubes(iter_fn: Callable,
                                          split_factor: int,
                                          surface,
                                          params: Any,
                                          root_lower: jnp.ndarray,
                                          root_upper: jnp.ndarray,
                                          max_depth: int,
                                          batch_process_size: int):
    """Option-A driver — per parent, classify all children via parent polynomial. `max_depth` iterations.

    Returns (leaves, stats). One Taylor pass per parent ⇒ `mlp_calls = n_occ * this_b` (one
    MLP forward pass per parent, regardless of the `split_factor` children classified from it).
    """
    classify_subcubes_fn = surface.classify_subcubes
    B = batch_process_size
    stats = []

    node_lower = jnp.asarray(root_lower, dtype=jnp.float32)[None, :]
    node_upper = jnp.asarray(root_upper, dtype=jnp.float32)[None, :]
    node_valid = jnp.ones((1,), dtype=bool)
    N_curr = 1

    for i_split in range(max_depth):
        if N_curr == 0:
            break

        init_size = node_lower.shape[0]
        this_b = min(B, init_size)
        node_valid = node_valid.reshape(-1, this_b)
        node_lower = node_lower.reshape(-1, this_b, 3)
        node_upper = node_upper.reshape(-1, this_b, 3)
        nb = node_lower.shape[0]
        n_occ = int(math.ceil(N_curr / this_b))

        outL = split_factor * this_b
        out_valid = jnp.zeros((nb, outL), dtype=bool)
        out_lower = jnp.zeros((nb, outL, 3), dtype=jnp.float32)
        out_upper = jnp.zeros((nb, outL, 3), dtype=jnp.float32)
        total_n_valid = jnp.zeros((), dtype=jnp.int32)

        for ib in range(n_occ):
            out_valid, out_lower, out_upper, total_n_valid = iter_fn(
                classify_subcubes_fn, params,
                node_valid[ib], node_lower[ib], node_upper[ib],
                ib, out_valid, out_lower, out_upper, total_n_valid,
            )

        node_valid = out_valid.reshape(-1)
        node_lower = out_lower.reshape(-1, 3)
        node_upper = out_upper.reshape(-1, 3)
        N_curr = int(total_n_valid)

        stats.append({
            "iter": i_split,
            "this_b": this_b,
            "n_occ": n_occ,
            "mlp_calls": n_occ * this_b,
            "n_unknown_after": N_curr,
        })

        if N_curr == 0:
            return Leaves(
                valid=jnp.zeros((0,), dtype=bool),
                lower=jnp.zeros((0, 3), dtype=jnp.float32),
                upper=jnp.zeros((0, 3), dtype=jnp.float32),
                uniformity="uniform",
            ), stats

        target_size = get_next_bucket_size(N_curr)
        (node_lower, node_upper), node_valid = compact_to_bucket(
            (node_lower, node_upper), node_valid, target_size
        )

    return Leaves(valid=node_valid, lower=node_lower, upper=node_upper, uniformity="uniform"), stats
