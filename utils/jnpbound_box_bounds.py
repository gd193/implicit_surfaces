"""Provably-valid box bounds extracted from a `jnp_bounds` Taylor enclosure.

A degree-k Taylor enclosure produced by `jb.taylor_bounds` over a trust region
encodes provably-valid bounds on the function value and on its first (k - 1)
derivatives across the entire trust region. These helpers extract those bounds.

The canonical primitive is `range_over_box(poly, lower, upper)` which returns a
provably-valid (lo, hi) of `poly` over the input box. `implementation="warp"`
uses the Warp range kernel where supported, else `eval` (also the "jax" path). For derivative
bounds we differentiate the polynomial symbolically (coefficient reshuffle)
and evaluate the resulting derivative polynomial over the same interval.

Convention: the polynomial is in *local* coordinates centred at the trust-region
origin; pass `Interval(lower=-half_extent, upper=+half_extent)` to bound over
the actual box.
"""
from __future__ import annotations

from typing import Tuple

import jax.numpy as jnp

from jnp_bounds.mixed_polynomials_arithmetic import range_over_box
from jnp_bounds.types import Interval, MixedIntervalPolynomial
from jnp_bounds.utils.dim_degree_coefficients import (
    dim_coeff_len_to_degree_cache,
    dim_degree_to_exponents_cache,
)


def _poly_input_degree(poly: MixedIntervalPolynomial, dim: int) -> int:
    return dim_coeff_len_to_degree_cache[(dim, poly.num_coeffs)]


def _interval_eval_to_pair(res) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """`mp_eval` may return an Interval or a scalar (constant poly); normalise."""
    if isinstance(res, Interval):
        return jnp.squeeze(res.lower), jnp.squeeze(res.upper)
    out = jnp.squeeze(res)
    return out, out


def box_value_bounds(poly: MixedIntervalPolynomial,
                     half_extent: jnp.ndarray,
                     implementation: str = "jax") -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Provably-valid (lo, hi) of f over the box [center - half_extent, center + half_extent].

    Polynomial must be in local coords centred at the box centre.
    """
    res = range_over_box(poly, -half_extent, half_extent, implementation=implementation)
    return _interval_eval_to_pair(res)


def polynomial_subbox_bounds(poly: MixedIntervalPolynomial,
                             delta_lower: jnp.ndarray,
                             delta_upper: jnp.ndarray,
                             implementation: str = "jax") -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Provably-valid (lo, hi) of `poly(δ)` for δ ∈ [delta_lower, delta_upper].

    Generalises `box_value_bounds` to off-centre sub-boxes inside the trust region.
    The polynomial's trust region is centred at δ = 0 with extent ±half_extent; this helper
    accepts any sub-box [delta_lower, delta_upper] inside that domain. Used to classify all
    N sub-boxes of a parent cell from one Taylor pass.
    """
    res = range_over_box(poly, delta_lower, delta_upper, implementation=implementation)
    return _interval_eval_to_pair(res)


def polynomial_subboxes_bounds(poly: MixedIntervalPolynomial,
                               delta_lowers: jnp.ndarray,
                               delta_uppers: jnp.ndarray,
                               implementation: str = "jax") -> Tuple[jnp.ndarray, jnp.ndarray]:
    """`polynomial_subbox_bounds` for N sub-boxes [N, dim] at once, as (lo[N], hi[N]).

    One range launch over all boxes; the polynomial is shared, not broadcast per box.
    """
    res = range_over_box(poly, delta_lowers, delta_uppers,
                         implementation=implementation, batched=True)
    n = delta_lowers.shape[0]
    if isinstance(res, Interval):
        return res.lower.reshape(n), res.upper.reshape(n)
    out = jnp.reshape(res, (n,))
    return out, out


def differentiate_poly(poly: MixedIntervalPolynomial,
                       dim: int,
                       axis: int) -> MixedIntervalPolynomial:
    """Symbolic derivative of `poly` w.r.t. its `axis`-th input.

    A degree-k input polynomial yields a degree-(k-1) output polynomial. The
    differentiated coefficient is `α_axis * c_α` placed at multi-index `α - e_axis`.
    Output real/interval split is determined by output multi-index degree, NOT by
    the input coef's real/interval status — but in practice an input real coef of
    degree d gives a derivative coef of degree d-1 ≤ in_deg - 2 < out_deg, so it
    lands in the output's real block; an input interval coef of degree in_deg
    gives a derivative coef of degree in_deg - 1 = out_deg, so it lands in the
    output's interval block. This lets the result remain a clean
    MixedIntervalPolynomial.
    """
    in_degree = _poly_input_degree(poly, dim)
    if in_degree < 1:
        # constant → zero gradient
        zero_real = jnp.zeros((1,) + poly.data_shape, dtype=poly.real_coeffs.dtype)
        empty_int_lo = jnp.zeros((0,) + poly.data_shape, dtype=poly.interval_coeffs.lower.dtype)
        empty_int_hi = jnp.zeros((0,) + poly.data_shape, dtype=poly.interval_coeffs.upper.dtype)
        return MixedIntervalPolynomial(
            real_coeffs=zero_real,
            interval_coeffs=Interval(lower=empty_int_lo, upper=empty_int_hi),
        )

    out_degree = in_degree - 1
    in_exps = list(dim_degree_to_exponents_cache.static[(dim, in_degree)][0])
    out_exps = list(dim_degree_to_exponents_cache.static[(dim, out_degree)][0])

    out_exp_to_idx = {tuple(e): i for i, e in enumerate(out_exps)}
    n_out = len(out_exps)
    n_out_real = sum(1 for e in out_exps if sum(e) < out_degree)
    n_out_int = n_out - n_out_real

    new_real = jnp.zeros((n_out_real,) + poly.data_shape, dtype=poly.real_coeffs.dtype)
    new_int_lo = jnp.zeros((n_out_int,) + poly.data_shape, dtype=poly.interval_coeffs.lower.dtype)
    new_int_hi = jnp.zeros((n_out_int,) + poly.data_shape, dtype=poly.interval_coeffs.upper.dtype)

    n_in_real = poly.num_real_coeffs
    for i, e in enumerate(in_exps):
        if e[axis] < 1:
            continue
        scale = float(e[axis])
        new_e = list(e)
        new_e[axis] -= 1
        out_idx = out_exp_to_idx[tuple(new_e)]
        out_is_interval = sum(new_e) == out_degree
        in_is_interval = i >= n_in_real

        if in_is_interval:
            ci = i - n_in_real
            lo = scale * poly.interval_coeffs.lower[ci]
            hi = scale * poly.interval_coeffs.upper[ci]
            if out_is_interval:
                local_idx = out_idx - n_out_real
                new_int_lo = new_int_lo.at[local_idx].add(lo)
                new_int_hi = new_int_hi.at[local_idx].add(hi)
            else:
                # input interval at degree d gives derivative degree d-1; if that
                # lands in out's real block, the derivative is interval-valued and
                # we widen — promote by adding to both halves of an interval entry.
                # For a clean MixedIntervalPolynomial, we represent this by adding
                # a midpoint to the real block and the symmetric half-width into
                # an extra interval term — but since this case can only occur when
                # in_degree > out_degree + 1 (impossible since out_degree = in_degree - 1),
                # we never hit it in practice. Guard with an assert.
                raise AssertionError("Unreachable: interval input at non-leading degree")
        else:
            v = scale * poly.real_coeffs[i]
            if out_is_interval:
                local_idx = out_idx - n_out_real
                new_int_lo = new_int_lo.at[local_idx].add(v)
                new_int_hi = new_int_hi.at[local_idx].add(v)
            else:
                new_real = new_real.at[out_idx].add(v)

    return MixedIntervalPolynomial(
        real_coeffs=new_real,
        interval_coeffs=Interval(lower=new_int_lo, upper=new_int_hi),
    )


def box_grad_bounds(poly: MixedIntervalPolynomial,
                    dim: int,
                    half_extent: jnp.ndarray,
                    implementation: str = "jax") -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Per-axis (lo[d], hi[d]) bound on ∇f over the box.

    Requires input polynomial degree >= 2 (otherwise gradient is constant or
    undefined).  Computed by symbolic-differentiating `poly` per axis and
    interval-evaluating each derivative polynomial over the trust region.
    """
    in_degree = _poly_input_degree(poly, dim)
    if in_degree < 2:
        raise ValueError(
            f"box_grad_bounds requires Taylor degree >= 2 (got {in_degree})"
        )
    los = []
    his = []
    for axis in range(dim):
        d_poly = differentiate_poly(poly, dim, axis)
        lo, hi = box_value_bounds(d_poly, half_extent, implementation)
        los.append(lo)
        his.append(hi)
    return jnp.stack(los), jnp.stack(his)


def box_hessian_bounds(poly: MixedIntervalPolynomial,
                       dim: int,
                       half_extent: jnp.ndarray,
                       implementation: str = "jax") -> Tuple[jnp.ndarray, jnp.ndarray]:
    """(lo[d, d], hi[d, d]) bound on Hessian over the box. Requires degree >= 3."""
    in_degree = _poly_input_degree(poly, dim)
    if in_degree < 3:
        raise ValueError(
            f"box_hessian_bounds requires Taylor degree >= 3 (got {in_degree})"
        )
    lo_rows = []
    hi_rows = []
    for axis_a in range(dim):
        d1 = differentiate_poly(poly, dim, axis_a)
        los = []
        his = []
        for axis_b in range(dim):
            d2 = differentiate_poly(d1, dim, axis_b)
            lo, hi = box_value_bounds(d2, half_extent, implementation)
            los.append(lo)
            his.append(hi)
        lo_rows.append(jnp.stack(los))
        hi_rows.append(jnp.stack(his))
    return jnp.stack(lo_rows), jnp.stack(hi_rows)
