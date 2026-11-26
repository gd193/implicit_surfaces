"""Tests for the sphereWalker driver, WoS/WoSt value and gradient solvers, and differential WoS."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from sphereWalker import (
    DifferentialWoSIntegrator,
    MCIntegrator,
    WoSGradientIntegrator,
    WoSIntegrator,
    WoStGradientIntegrator,
    WoStIntegrator,
    WoStParameters,
    average_draws,
    finish_step,
)
from sphereWalker.types import WalkerTermination as WT
from surfaces.implicit.lipschitz import sdfImplicit
from surfaces.implicit.primitives import sphere_sdf


def _unit_disk():
    surface = sdfImplicit(fn=sphere_sdf, dim=2)
    params = {"center": jnp.zeros(2, jnp.float32), "radius": jnp.float32(1.0)}
    return surface, params


def test_wos_integrator_vector_pde_with_vector_source():
    # u(x, y) = [x^2 + y^2, 2x + y^2] on the unit disk:
    # -Lap u = [-4, -2], u(0, 0) = [0, 0].
    surface, params = _unit_disk()

    def boundary_fn(x):
        return jnp.array([jnp.sum(x * x), 2.0 * x[0] + x[1] ** 2], dtype=x.dtype)

    def source_fn(x):
        return jnp.array([-4.0, -2.0], dtype=x.dtype)

    integrator = WoSIntegrator(
        empty_sphere_fn=surface.empty_sphere,
        boundary_condition_fn=boundary_fn,
        source_term_fn=source_fn,
        dim=2,
        max_steps=128,
        eps=1e-3,
    )
    points = jnp.array([[0.0, 0.0], [0.25, -0.25]], dtype=jnp.float32)
    out = integrator.integrate_batch(
        params,
        points,
        jr.PRNGKey(0),
        init_trust_region_size=1.0,
        n_samples=8192,
        steps_between_refill=16,
        max_active_states=1024,
        jit_buck_size=2,
    )
    expected = jax.vmap(boundary_fn)(points)
    assert out.w.shape == (2, 2)
    np.testing.assert_allclose(out.w, expected, atol=0.05)


def test_wos_integrator_scalar_shape_preserved_and_bad_source_rejected():
    surface, params = _unit_disk()
    scalar_wos = WoSIntegrator(
        empty_sphere_fn=surface.empty_sphere,
        boundary_condition_fn=lambda x: jnp.asarray(1.0, dtype=x.dtype),
        source_term_fn=lambda x: jnp.asarray(0.0, dtype=x.dtype),
        dim=2,
        max_steps=32,
        eps=1e-3,
    )
    out = scalar_wos.integrate_batch(
        params,
        jnp.zeros((1, 2), jnp.float32),
        jr.PRNGKey(1),
        init_trust_region_size=1.0,
        n_samples=16,
        jit_buck_size=1,
    )
    assert out.w.shape == (1, 1)
    np.testing.assert_allclose(out.w, [[1.0]], atol=1e-6)

    bad_source_wos = WoSIntegrator(
        empty_sphere_fn=surface.empty_sphere,
        boundary_condition_fn=lambda x: jnp.zeros(2, dtype=x.dtype),
        source_term_fn=lambda x: jnp.zeros(3, dtype=x.dtype),
        dim=2,
        max_steps=8,
        eps=1e-3,
    )
    with pytest.raises(ValueError, match="broadcast"):
        bad_source_wos.integrate_batch(
            params,
            jnp.zeros((1, 2), jnp.float32),
            jr.PRNGKey(2),
            init_trust_region_size=1.0,
            n_samples=4,
            jit_buck_size=1,
        )


def test_differential_wos_skips_active_lanes_and_matches_derivative():
    # u(x; R) = R^2 - ||x||^2 solves -Lap u = 4 with u = 0 on ||x|| = R.
    # Exact derivative: du/dR = 2R (= 2 at R = 1).
    surface, params = _unit_disk()
    eval_chunks = []

    def Df(p, x, g):
        jax.debug.callback(lambda: eval_chunks.append(1))
        return surface.dirichlet_boundary_differential(p, x, g)

    integrator = DifferentialWoSIntegrator(
        empty_sphere_fn=surface.empty_sphere,
        boundary_condition_fn=lambda x: jnp.asarray(0.0, dtype=x.dtype),
        source_term_fn=lambda x: jnp.asarray(4.0, dtype=x.dtype),
        normal_fn=surface.normal_at,
        Df=Df,
        V_n=surface.normal_surface_velocity,
        offset_l=0.05,
        transition_bucket_size=4,
        dim=2,
        max_steps=128,
        eps=1e-3,
    )

    # 1. With 0 completed lanes, _accumulate_completed must not call Df at all.
    buf = 16
    init_pos = jnp.zeros((buf, 2), jnp.float32)
    keys = jr.split(jr.PRNGKey(3), buf)
    states = integrator._get_initial_state(init_pos, keys, 1.0, buf)
    parents = jnp.zeros((buf,), jnp.int32)
    is_done = jnp.zeros((buf,), dtype=bool)
    sums = jax.tree.map(lambda leaf: jnp.zeros((1,) + leaf.shape, leaf.dtype), params)
    counts = jnp.zeros((1, len(WT)), jnp.int32)
    out_sums, out_counts = jax.jit(integrator._accumulate_completed, static_argnums=(6,))(
        params, states, parents, is_done, sums, counts, 1
    )
    jax.block_until_ready(out_counts)
    assert len(eval_chunks) == 0
    assert int(out_counts.sum()) == 0

    # 2. Full solve matches du/dR = 2.0 at origin (within finite-difference offset bias).
    out = integrator.integrate_batch(
        params,
        jnp.zeros((1, 2), jnp.float32),
        jr.PRNGKey(4),
        init_trust_region_size=1.0,
        n_samples=4096,
        steps_between_refill=16,
        max_active_states=512,
        jit_buck_size=1,
    )
    np.testing.assert_allclose(out.w["radius"], [2.0], rtol=0.15)


def test_differential_wos_marks_nonfinite_update_invalid():
    surface, params = _unit_disk()
    integrator = DifferentialWoSIntegrator(
        empty_sphere_fn=surface.empty_sphere,
        boundary_condition_fn=lambda x: jnp.asarray(0.0, dtype=x.dtype),
        source_term_fn=lambda x: jnp.asarray(0.0, dtype=x.dtype),
        normal_fn=surface.normal_at,
        Df=lambda p, x, g: jax.tree.map(lambda a: jnp.full_like(a, jnp.nan), p),
        V_n=surface.normal_surface_velocity,
        offset_l=0.05,
        transition_bucket_size=4,
        dim=2,
        max_steps=128,
        eps=1e-3,
    )
    details = integrator.integrate_batch(
        params,
        jnp.zeros((1, 2), jnp.float32),
        jr.PRNGKey(5),
        init_trust_region_size=1.0,
        n_samples=8,
        steps_between_refill=16,
        max_active_states=8,
        jit_buck_size=1,
        return_details=True,
    )
    assert int(details.successful_samples[0]) == 0
    assert int(details.termination_counts[0, WT.INVALID]) == 8



def test_wos_integrator_float32_positions_float64_vector_source_under_x64():
    with jax.enable_x64(True):
        surface, params = _unit_disk()

        def boundary_fn(x):
            return jnp.array([jnp.sum(x * x), 2.0 * x[0] + x[1] ** 2], dtype=x.dtype)

        integrator = WoSIntegrator(
            empty_sphere_fn=surface.empty_sphere,
            boundary_condition_fn=boundary_fn,
            source_term_fn=lambda x: jnp.array([-4.0, -2.0], dtype=jnp.float64),
            dim=2,
            max_steps=128,
            eps=1e-3,
        )
        out = integrator.integrate_batch(
            params,
            jnp.zeros((1, 2), jnp.float32),
            jr.PRNGKey(11),
            init_trust_region_size=1.0,
            n_samples=2048,
            steps_between_refill=16,
            max_active_states=512,
            jit_buck_size=1,
        )
        assert out.w.shape == (1, 2)
        assert out.w.dtype == jnp.float32
        np.testing.assert_allclose(out.w, [[0.0, 0.0]], atol=0.1)


def test_differential_accumulate_pads_tail_with_completed_walker():
    surface, params = _unit_disk()
    seen = []

    def record(x):
        seen.append(np.asarray(x).reshape(-1, 2))

    def poisoned(fn):
        def wrapped(p, x, *rest):
            jax.debug.callback(record, x)
            out = fn(p, x, *rest)
            at_origin = jnp.all(x == 0)
            return jax.tree.map(lambda a: jnp.where(at_origin, jnp.nan, a), out)
        return wrapped

    integrator = DifferentialWoSIntegrator(
        empty_sphere_fn=surface.empty_sphere,
        boundary_condition_fn=lambda x: jnp.asarray(0.0, dtype=x.dtype),
        source_term_fn=lambda x: jnp.asarray(0.0, dtype=x.dtype),
        normal_fn=surface.normal_at,
        Df=poisoned(surface.dirichlet_boundary_differential),
        V_n=poisoned(surface.normal_surface_velocity),
        offset_l=0.05,
        transition_bucket_size=4,
        dim=2,
        max_steps=128,
        eps=1e-3,
    )
    buf = 8
    states = integrator._get_initial_state(
        jnp.zeros((buf, 2), jnp.float32), jr.split(jr.PRNGKey(0), buf), 1.0, buf)
    k = 5
    states = states._replace(
        position=states.position.at[k].set(jnp.array([0.95, 0.0], jnp.float32)),
        boundary_hit_position=states.boundary_hit_position.at[k].set(jnp.array([1.0, 0.0], jnp.float32)),
        offset_length=states.offset_length.at[k].set(0.05),
        status=states.status.at[k].set(WT.BOUNDARY_HIT),
        phase=states.phase.at[k].set(1),
        done_flag=states.done_flag.at[k].set(True),
    )
    parents = jnp.zeros((buf,), jnp.int32)
    sums = jax.tree.map(lambda leaf: jnp.zeros((1,) + leaf.shape, leaf.dtype), params)
    counts = jnp.zeros((1, len(WT)), jnp.int32)
    out_sums, out_counts = jax.jit(integrator._accumulate_completed, static_argnums=(6,))(
        params, states, parents, states.done_flag, sums, counts, 1)
    jax.block_until_ready(out_counts)
    jax.effects_barrier()

    assert int(out_counts[0, WT.BOUNDARY_HIT]) == 1
    assert int(out_counts.sum()) == 1
    for leaf in jax.tree.leaves(out_sums):
        assert bool(jnp.all(jnp.isfinite(leaf)))
    evaluated = np.concatenate(seen)
    assert evaluated.size > 0
    assert not np.any(np.all(evaluated == 0.0, axis=1))

def test_pre_warm_compaction_buckets_warms_refill_rounds():
    surface, params = _unit_disk()
    integrator = WoSIntegrator(
        empty_sphere_fn=surface.empty_sphere,
        boundary_condition_fn=lambda x: jnp.asarray(1.0, dtype=x.dtype),
        source_term_fn=lambda x: jnp.asarray(0.0, dtype=x.dtype),
        dim=2,
        max_steps=128,
        eps=1e-3,
    )
    points = jnp.zeros((2, 2), jnp.float32)
    kwargs = dict(
        init_trust_region_size=1.0,
        n_samples=32,
        steps_between_refill=8,
        max_active_states=8,
        jit_buck_size=2,
        n_compactions=2,
        refill_rounds=3,
    )
    with pytest.raises(ValueError, match="refill_rounds"):
        integrator.pre_warm_compaction_buckets(
            params, points, jr.PRNGKey(6), **{**kwargs, "refill_rounds": 0}
        )

    integrator.pre_warm_compaction_buckets(params, points, jr.PRNGKey(6), **kwargs)
    warmed_cache = integrator._process_refill_rounds._cache_size()
    assert warmed_cache >= 1

    out = integrator.integrate_batch(params, points, jr.PRNGKey(7), **kwargs)
    assert integrator._process_refill_rounds._cache_size() == warmed_cache
    np.testing.assert_allclose(out.w, [[1.0], [1.0]], atol=1e-6)


# ---------------------------------------------------------------- gradients
def _quadratic(x):
    # -Lap u = -4 and grad u = (2x + 2, 2y + 1).
    return jnp.sum(x * x) + 2.0 * x[0] + x[1]


def _quadratic_gradient(points):
    points = np.asarray(points)
    return np.stack([2 * points[:, 0] + 2, 2 * points[:, 1] + 1], axis=-1)


_GRADIENT_POINTS = np.array([[0.3, -0.2], [-0.1, 0.4]], np.float32)


def _gradient_solve(integrator, params, key, n_samples):
    return integrator.integrate_batch(
        params, jnp.asarray(_GRADIENT_POINTS), key, init_trust_region_size=1.0,
        n_samples=n_samples, steps_between_refill=16, max_active_states=4096, jit_buck_size=2)


def test_wos_gradient_matches_analytic_gradient_on_unit_disk():
    surface, params = _unit_disk()
    integrator = WoSGradientIntegrator(
        surface.empty_sphere, _quadratic, lambda x: jnp.asarray(-4.0, x.dtype), 2,
        max_steps=256, eps=1e-3, control_fn=_quadratic)
    out = _gradient_solve(integrator, params, jr.PRNGKey(20), 8192)
    assert out.w.shape == (2, 1, 2)
    np.testing.assert_allclose(out.w[:, 0], _quadratic_gradient(_GRADIENT_POINTS), atol=0.15)


def test_wos_gradient_step_block_skips_first_ball_after_step_zero():
    surface, params = _unit_disk()
    first_steps = []

    class Counting(WoSGradientIntegrator):
        def _first_step_plan(self, params, state, query):
            jax.debug.callback(lambda: first_steps.append(1))
            return super()._first_step_plan(params, state, query)

    integrator = Counting(surface.empty_sphere, _quadratic, lambda x: jnp.asarray(-4.0, x.dtype), 2,
                          max_steps=256, eps=1e-3)
    buf = 8
    positions = jnp.broadcast_to(jnp.asarray(_GRADIENT_POINTS[:1]), (buf, 2))
    states = integrator._get_initial_state(positions, jr.split(jr.PRNGKey(21), buf), 1.0, buf)

    def block(steps):
        first_steps.clear()
        out = jax.jit(lambda s: integrator._step_block(params, s, steps, False))(states)
        jax.block_until_ready(out)
        jax.effects_barrier()
        return out, len(first_steps)

    _, one_step = block(1)
    fast, six_steps = block(6)
    assert one_step > 0
    # Interior starts leave the first ball in step 0, so the remaining steps
    # take the continuation-only branch and never run first-ball work.
    assert six_steps == one_step

    # The fast path gives the same states as six masked steps.
    masked_step = jax.vmap(integrator.step, in_axes=(None, 0))
    masked = jax.jit(lambda s: jax.lax.fori_loop(0, 6, lambda i, t: masked_step(params, t), s))(states)
    for a, b in zip(jax.tree.leaves(fast), jax.tree.leaves(masked)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-6, atol=1e-6)


def test_wos_gradient_float32_geometry_float64_source_under_x64():
    with jax.enable_x64(True):
        surface, params = _unit_disk()
        integrator = WoSGradientIntegrator(
            surface.empty_sphere, _quadratic, lambda x: jnp.asarray(-4.0, jnp.float64), 2,
            max_steps=256, eps=1e-3, control_fn=_quadratic)
        out = _gradient_solve(integrator, params, jr.PRNGKey(22), 4096)
        assert out.w.dtype == jnp.float32
        np.testing.assert_allclose(out.w[:, 0], _quadratic_gradient(_GRADIENT_POINTS), atol=0.2)


def test_wos_gradient_mirrored_pair_with_parameterized_control_and_source_draws():
    surface, params = _unit_disk()
    params = {**params, "shift": jnp.float32(0.5)}

    def make(mirror):
        return WoSGradientIntegrator(
            surface.empty_sphere, _quadratic, lambda x: jnp.asarray(-4.0, x.dtype), 2,
            max_steps=256, eps=1e-3, source_samples=2,
            control_fn=lambda p, x: _quadratic(x) + p["shift"], control_takes_params=True,
            mirror_first_sphere=mirror, gradient_source_samples=4)

    key = jr.PRNGKey(23)
    plain = _gradient_solve(make(False), params, key, 4096)
    mirrored = _gradient_solve(make(True), params, key, 4096)
    np.testing.assert_allclose((plain.w[:, 0] + mirrored.w[:, 0]) / 2,
                               _quadratic_gradient(_GRADIENT_POINTS), atol=0.15)

    with pytest.raises(ValueError, match="gradient_source_samples"):
        WoSGradientIntegrator(surface.empty_sphere, _quadratic, _quadratic, 2, gradient_source_samples=0)
    bad_control = WoSGradientIntegrator(
        surface.empty_sphere, _quadratic, lambda x: jnp.asarray(-4.0, x.dtype), 2,
        control_fn=lambda x: jnp.zeros(3, x.dtype))
    with pytest.raises(ValueError, match="broadcast"):
        _gradient_solve(bad_control, params, key, 4)


# --------------------------------------------------------------------- WoSt
_A, _B = 0.3, 1.0
_ANNULUS_POINTS = np.array([[0.6, 0.0], [0.0, -0.5]], np.float32)


def _annulus_truth(r):
    # -Lap u = 1, u(a) = 0, u'(b) = 0.
    return (_A * _A - r * r) / 4 + 0.5 * _B * _B * jnp.log(r / _A)


def _annulus_gradient(points):
    points = np.asarray(points, np.float64)
    r = np.linalg.norm(points, axis=-1, keepdims=True)
    return (-r / 2 + _B * _B / (2 * r)) * points / r


def _annulus_solvers(source):
    inner = sdfImplicit(fn=sphere_sdf, dim=2)
    outer = sdfImplicit(fn=sphere_sdf, dim=2)
    params = {"center": jnp.zeros(2, jnp.float32), "radius": jnp.float32(_A)}
    neumann = {"center": jnp.zeros(2, jnp.float32), "radius": jnp.float32(_B)}
    zero = lambda x: jnp.zeros((), x.dtype)
    common = dict(neumann_surface=outer, eps=1e-3, max_steps=4000)
    value = WoStIntegrator(inner.empty_sphere, zero, source, 2, **common)
    gradient = WoStGradientIntegrator(
        inner.empty_sphere, zero, source, 2, **common,
        control_fn=lambda x: _annulus_truth(jnp.linalg.norm(x)))
    return value, gradient, params, neumann


def _wost_solve(integrator, params, neumann, key, n_samples):
    return integrator.integrate_batch(
        params, jnp.asarray(_ANNULUS_POINTS), key, 1.0, n_samples, neumann_params=neumann,
        steps_between_refill=16, max_active_states=4096, jit_buck_size=2)


def test_wost_annulus_value_and_gradient_match_mixed_boundary_solution():
    value, gradient, params, neumann = _annulus_solvers(lambda x: jnp.ones((), x.dtype))
    radii = np.linalg.norm(_ANNULUS_POINTS, axis=-1)

    out = _wost_solve(value, params, neumann, jr.PRNGKey(30), 4096)
    assert out.w.shape == (2, 1)
    np.testing.assert_allclose(out.w[:, 0], _annulus_truth(radii), atol=0.03)

    grad = _wost_solve(gradient, params, neumann, jr.PRNGKey(31), 4096)
    assert grad.w.shape == (2, 1, 2)
    np.testing.assert_allclose(grad.w[:, 0], _annulus_gradient(_ANNULUS_POINTS), atol=0.1)


def test_wost_float32_geometry_float64_source_under_x64():
    with jax.enable_x64(True):
        value, gradient, params, neumann = _annulus_solvers(lambda x: jnp.ones((), jnp.float64))
        radii = np.linalg.norm(_ANNULUS_POINTS, axis=-1)
        out = _wost_solve(value, params, neumann, jr.PRNGKey(32), 2048)
        assert out.w.dtype == jnp.float32
        np.testing.assert_allclose(out.w[:, 0], _annulus_truth(radii), atol=0.04)
        grad = _wost_solve(gradient, params, neumann, jr.PRNGKey(33), 2048)
        assert grad.w.dtype == jnp.float32
        np.testing.assert_allclose(grad.w[:, 0], _annulus_gradient(_ANNULUS_POINTS), atol=0.15)


# ------------------------------------------------------------- differential
def _differential(**overrides):
    surface, params = _unit_disk()
    options = dict(
        empty_sphere_fn=surface.empty_sphere,
        boundary_condition_fn=lambda x: jnp.asarray(0.0, dtype=x.dtype),
        source_term_fn=lambda x: jnp.asarray(1.0, dtype=x.dtype),
        normal_fn=surface.normal_at,
        Df=surface.dirichlet_boundary_differential,
        V_n=surface.normal_surface_velocity,
        offset_l=0.05,
        transition_bucket_size=4,
        dim=2,
        max_steps=128,
        eps=1e-3,
    )
    options.update(overrides)
    return DifferentialWoSIntegrator(**options), surface, params


def test_differential_zero_radius_phase_one_step_stays_finite():
    def undecided(p, x, tr):
        return jnp.asarray(-0.5, x.dtype), jnp.asarray(0.0, x.dtype), tr

    with jax.enable_x64(True):
        integrator, _, params = _differential(
            empty_sphere_fn=undecided, source_term_fn=lambda x: jnp.asarray(1.0, jnp.float64))
        # Unguarded Green weight: r/R is NaN at r = R = 0.
        integrator.weighted_greens_fn = lambda r, R: r / R
        state = integrator._get_initial_state(
            jnp.array([[0.2, 0.1]], jnp.float32), jr.split(jr.PRNGKey(40), 1), 1.0, 1)
        state = state._replace(phase=jnp.ones(1, jnp.int32))
        out = jax.jit(jax.vmap(integrator.step, in_axes=(None, 0)))(params, state)
    assert int(out.status[0]) == WT.ACTIVE
    assert int(out.nSteps[0]) == 1
    assert out.w.dtype == jnp.float32
    assert float(out.w[0]) == 0.0
    np.testing.assert_array_equal(out.position, state.position)


def test_differential_transition_pads_tail_with_transitioning_walker():
    seen = []

    def record(x):
        seen.append(np.asarray(x).reshape(-1, 2))

    surface, _ = _unit_disk()

    def normal_fn(p, x):
        jax.debug.callback(record, x)
        return surface.normal_at(p, x)

    integrator, _, params = _differential(normal_fn=normal_fn)
    buf = 8
    states = integrator._get_initial_state(
        jnp.zeros((buf, 2), jnp.float32), jr.split(jr.PRNGKey(41), buf), 1.0, buf)
    transition = jax.jit(lambda s: integrator._transition_state_phases(params, s, 1.0))

    # Nothing selected: no chunk runs, so normal_fn is never evaluated.
    jax.block_until_ready(transition(states))
    jax.effects_barrier()
    assert seen == []

    # Five boundary hits with bucket 4: the second chunk has three padding slots.
    hits = np.array([1, 2, 3, 5, 6])
    angles = np.linspace(0.3, 2.5, len(hits))
    boundary = np.stack([np.cos(angles), np.sin(angles)], axis=-1).astype(np.float32)
    states = states._replace(
        position=states.position.at[hits].set(boundary),
        status=states.status.at[hits].set(WT.BOUNDARY_HIT),
        done_flag=states.done_flag.at[hits].set(True))
    out = transition(states)
    jax.block_until_ready(out)
    jax.effects_barrier()

    evaluated = np.concatenate(seen)
    assert len(evaluated) == 8  # two chunks of four
    assert not np.any(np.all(evaluated == 0.0, axis=1))
    np.testing.assert_array_equal(np.asarray(out.phase)[hits], 1)
    np.testing.assert_array_equal(np.asarray(out.status)[hits], WT.ACTIVE)
    np.testing.assert_allclose(np.asarray(out.boundary_hit_position)[hits], boundary)
    others = np.setdiff1d(np.arange(buf), hits)
    np.testing.assert_array_equal(np.asarray(out.phase)[others], 0)
    np.testing.assert_array_equal(np.asarray(out.position)[others], 0.0)


# ------------------------------------------------------------------- driver
class _CountState(NamedTuple):
    position: jnp.ndarray
    nSteps: jnp.ndarray
    w: jnp.ndarray
    rng_key: jnp.ndarray
    done_flag: jnp.ndarray
    status: jnp.ndarray


class _UniformSum(MCIntegrator[_CountState]):
    """Not a PDE: each sample is x[0] plus `length` uniform draws."""

    length = 3

    def _get_initial_state(self, positions, keys, init_tr, buffer_size):
        return _CountState(positions, jnp.zeros(buffer_size, jnp.int32),
                           jnp.zeros(buffer_size, positions.dtype), keys,
                           jnp.zeros(buffer_size, bool), jnp.zeros(buffer_size, jnp.int32))

    def step(self, params, state):
        key, draw = jr.split(state.rng_key)
        n = state.nSteps + 1
        done = n >= self.length
        updated = state._replace(
            nSteps=n, w=state.w + jr.uniform(draw, dtype=state.w.dtype), rng_key=key,
            done_flag=done, status=jnp.where(done, WT.BOUNDARY_HIT, WT.ACTIVE).astype(jnp.int32))
        return finish_step(state, updated)

    def _compute_accumulator_update(self, params, state):
        return state.position[0] + state.w


@pytest.mark.parametrize("make_key", [jr.PRNGKey, jr.key], ids=["raw_key", "typed_key"])
def test_mc_integrator_direct_subclass_refills_without_pde_callbacks(make_key):
    integrator = _UniformSum(2, max_steps=10)
    points = jnp.array([[0.0, 0.0], [1.0, 0.0], [-2.0, 0.5]], jnp.float32)
    n_samples = 256
    # Few lanes and a step block that does not divide the sample length, so
    # refills land mid-block and the generic _refill_state is exercised.
    options = dict(init_trust_region_size=1.0, n_samples=n_samples, steps_between_refill=2,
                   max_active_states=32, jit_buck_size=2, n_compactions=2, refill_rounds=3)
    details = integrator.integrate_batch(None, points, make_key(50), return_details=True, **options)
    np.testing.assert_array_equal(details.successful_samples, n_samples)
    np.testing.assert_array_equal(details.termination_counts[:, WT.BOUNDARY_HIT], n_samples)
    np.testing.assert_allclose(details.nSteps, _UniformSum.length)
    np.testing.assert_allclose(details.contribution_sum / n_samples,
                               np.asarray(points[:, 0]) + 1.5, atol=0.12)
    out = integrator.integrate_batch(None, points, make_key(50), **options)
    np.testing.assert_allclose(out.w, details.contribution_sum / n_samples)

    with pytest.raises(ValueError, match="dim"):
        _UniformSum(0)


# ------------------------------------------------- review regression tests
def _steps(integrator, params, states, n=1):
    step = jax.jit(jax.vmap(integrator.step, in_axes=(None, 0)))
    for _ in range(n):
        states = step(params, states)
    return states


def test_wos_gradient_masked_step_evaluates_source_once_per_lane():
    surface, params = _unit_disk()
    traces, calls = [], []

    def source(x):
        traces.append(1)
        # Count evaluated lanes, whether vmap loops or vectorizes the callback.
        jax.debug.callback(lambda v: calls.append(np.asarray(v).reshape(-1, 2).shape[0]), x)
        return jnp.asarray(-4.0, x.dtype)

    integrator = WoSGradientIntegrator(surface.empty_sphere, _quadratic, source, 2,
                                       max_steps=256, eps=1e-3)
    buf = 8
    positions = jnp.broadcast_to(jnp.asarray(_GRADIENT_POINTS[:1]), (buf, 2))
    states = integrator._get_initial_state(positions, jr.split(jr.PRNGKey(60), buf), 1.0, buf)
    # Half the lanes are first-ball steps, half are continuations.
    states = _steps(integrator, params, states)
    states = states._replace(grad_weight=states.grad_weight.at[: buf // 2].set(0.0))
    traces.clear()
    calls.clear()
    out = jax.jit(jax.vmap(integrator.step, in_axes=(None, 0)))(params, states)
    jax.block_until_ready(out)
    jax.effects_barrier()
    assert len(traces) == 1
    assert sum(calls) == buf

    # Same states as evaluating each branch with its own source call.
    def unfused(p, s):
        problem = integrator._problem_params(p)
        query = integrator.empty_sphere(problem, s.position, s.trust_region_size)
        first = integrator._run_plan(problem, s, *integrator._first_step_plan(p, s, query))
        walked = integrator._run_plan(problem, s, *integrator._continuation_plan(p, s, query))
        is_first = ~jnp.any(s.grad_weight != 0)
        return jax.tree.map(lambda a, b: jnp.where(is_first, a, b), first, walked)

    reference = jax.jit(jax.vmap(unfused, in_axes=(None, 0)))(params, states)
    for a, b in zip(jax.tree.leaves(out), jax.tree.leaves(reference)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_wos_gradient_step_block_leaves_masked_path_after_stall_recovers():
    surface, params = _unit_disk()
    first_steps = []

    def stalling(p, x, tr):
        # An undecided query at the initial trust region; decided afterwards.
        residual, radius, next_tr = surface.empty_sphere(p, x, tr)
        stall = tr >= 1.0
        return residual, jnp.where(stall, 0.0, radius), jnp.where(stall, 0.5, next_tr)

    class Counting(WoSGradientIntegrator):
        def _first_step_plan(self, params, state, query):
            jax.debug.callback(
                lambda x: first_steps.append(np.asarray(x).reshape(-1, 2).shape[0]), state.position)
            return super()._first_step_plan(params, state, query)

    integrator = Counting(stalling, _quadratic, lambda x: jnp.asarray(-4.0, x.dtype), 2,
                          max_steps=256, eps=1e-3)
    buf = 8
    positions = jnp.broadcast_to(jnp.asarray(_GRADIENT_POINTS[:1]), (buf, 2))
    states = integrator._get_initial_state(positions, jr.split(jr.PRNGKey(61), buf), 1.0, buf)
    # Only lane 0 stalls on step 0 and recovers on step 1.
    states = states._replace(trust_region_size=states.trust_region_size.at[1:].set(0.9))

    first_steps.clear()
    fast = jax.jit(lambda s: integrator._step_block(params, s, 8, False))(states)
    jax.block_until_ready(fast)
    jax.effects_barrier()
    # Masked steps 0 and 1 only (all lanes); steps 2..7 are continuation-only.
    assert sum(first_steps) == 2 * buf
    assert bool(jnp.all(jnp.any(fast.grad_weight != 0, axis=-1)))

    masked_step = jax.vmap(integrator.step, in_axes=(None, 0))
    masked = jax.jit(lambda s: jax.lax.fori_loop(0, 8, lambda i, t: masked_step(params, t), s))(states)
    for a, b in zip(jax.tree.leaves(fast), jax.tree.leaves(masked)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-6, atol=1e-6)


def _small_ball(p, x, tr):
    return jnp.asarray(-0.5, x.dtype), jnp.asarray(0.01, x.dtype), tr


_HUGE_SOURCE = 1e39  # overflows float32; source * R^2/(2 dim) = 2.5e34 does not


def test_wos_and_gradient_weight_huge_float64_source_before_cast_under_x64():
    with jax.enable_x64(True):
        _, params = _unit_disk()
        source = lambda x: jnp.asarray(_HUGE_SOURCE, jnp.float64)
        buf = 4096
        positions = jnp.broadcast_to(jnp.array([0.2, 0.1], jnp.float32), (buf, 2))
        keys = jr.split(jr.PRNGKey(62), buf)

        value = WoSIntegrator(_small_ball, lambda x: jnp.zeros((), x.dtype), source, 2, max_steps=8)
        out = _steps(value, params, value._get_initial_state(positions, keys, 1.0, buf))
        assert out.w.dtype == jnp.float32
        np.testing.assert_array_equal(out.status, WT.ACTIVE)
        assert bool(jnp.all(jnp.isfinite(out.w)))
        # E[weighted Green] = R^2 / (2 dim).
        np.testing.assert_allclose(float(jnp.mean(out.w.astype(jnp.float64))), 2.5e34, rtol=0.1)

        gradient = WoSGradientIntegrator(_small_ball, lambda x: jnp.zeros((), x.dtype), source, 2,
                                         max_steps=8)
        states = gradient._get_initial_state(positions[:64], keys[:64], 1.0, 64)
        first = _steps(gradient, params, states)
        np.testing.assert_array_equal(first.status, WT.ACTIVE)
        assert first.grad_source.dtype == jnp.float32
        assert bool(jnp.all(jnp.isfinite(first.grad_source)))
        assert float(jnp.max(jnp.abs(first.grad_source))) > 1e35
        walked = _steps(gradient, params, first)
        np.testing.assert_array_equal(walked.status, WT.ACTIVE)
        assert bool(jnp.all(jnp.isfinite(walked.w))) and float(jnp.min(walked.w)) > 0


@pytest.mark.parametrize("source_samples", [1, 3])
def test_wost_weights_huge_float64_source_before_cast_under_x64(source_samples):
    from sphereWalker import WoStParameters

    with jax.enable_x64(True):
        _, _, params, neumann = _annulus_solvers(lambda x: jnp.ones((), x.dtype))
        outer = sdfImplicit(fn=sphere_sdf, dim=2)
        integrator = WoStIntegrator(
            _small_ball, lambda x: jnp.zeros((), x.dtype),
            lambda x: jnp.asarray(_HUGE_SOURCE, jnp.float64), 2,
            neumann_surface=outer, max_steps=8, source_samples=source_samples)
        buf = 4
        positions = jnp.broadcast_to(jnp.array([0.6, 0.0], jnp.float32), (buf, 2))
        states = integrator._get_initial_state(positions, jr.split(jr.PRNGKey(63), buf), 1.0, buf)
        out = _steps(integrator, WoStParameters(params, neumann), states)
        np.testing.assert_array_equal(out.status, WT.ACTIVE)
        assert out.w.dtype == jnp.float32
        # Green-radius sampling: every interior draw weighs R^2 / (2 dim).
        np.testing.assert_allclose(np.asarray(out.w, np.float64), 2.5e34, rtol=1e-5)


def test_wost_gradient_first_ball_rejects_infinite_clearance_and_uses_neumann_retry():
    from sphereWalker import WoStParameters

    _, _, params, neumann = _annulus_solvers(lambda x: jnp.ones((), x.dtype))
    zero = lambda x: jnp.zeros((), x.dtype)
    one = lambda x: jnp.ones((), x.dtype)
    position = jnp.array([[0.6, 0.0]], jnp.float32)
    keys = jr.split(jr.PRNGKey(64), 1)

    infinite = lambda p, x, tr: (jnp.asarray(-0.5, x.dtype), jnp.asarray(jnp.inf, x.dtype), tr)
    integrator = WoStGradientIntegrator(infinite, zero, one, 2, max_steps=8,
                                        neumann_surface=sdfImplicit(fn=sphere_sdf, dim=2))
    out = _steps(integrator, WoStParameters(params, neumann),
                 integrator._get_initial_state(position, keys, 1.0, 1))
    assert int(out.status[0]) == WT.INVALID

    class StalledEnclosure(sdfImplicit):
        def empty_sphere(self, p, x, tr):
            value, _, _ = super().empty_sphere(p, x, tr)
            return value, jnp.zeros_like(value), 0.125 * tr

    inner = sdfImplicit(fn=sphere_sdf, dim=2)
    integrator = WoStGradientIntegrator(inner.empty_sphere, zero, one, 2, max_steps=8,
                                        neumann_surface=StalledEnclosure(fn=sphere_sdf, dim=2))
    states = integrator._get_initial_state(position, keys, 1.0, 1)
    out = _steps(integrator, WoStParameters(params, neumann), states)
    # The Dirichlet ball (0.3) is decided, so the stalled Neumann enclosure
    # limits the first ball and proposes the retry region.
    assert int(out.status[0]) == WT.ACTIVE
    assert int(out.nSteps[0]) == 1
    np.testing.assert_allclose(out.trust_region_size, [0.125])
    np.testing.assert_array_equal(out.grad_weight, 0.0)
    np.testing.assert_array_equal(out.position, position)


class _TypedKeySum(_UniformSum):
    def _get_initial_state(self, positions, keys, init_tr, buffer_size):
        if not jax.dtypes.issubdtype(keys.dtype, jax.dtypes.prng_key):
            raise TypeError("typed PRNG keys required")
        return super()._get_initial_state(positions, keys, init_tr, buffer_size)


def test_mc_integrator_passes_typed_keys_and_rejects_rbg():
    integrator = _TypedKeySum(2, max_steps=10)
    points = jnp.array([[0.0, 0.0], [1.0, 0.0]], jnp.float32)
    options = dict(init_trust_region_size=1.0, n_samples=8, steps_between_refill=2,
                   max_active_states=4, jit_buck_size=2)
    integrator.pre_warm_compaction_buckets(None, points, jr.key(70), **options)
    details = integrator.integrate_batch(None, points, jr.key(70), return_details=True, **options)
    np.testing.assert_array_equal(details.successful_samples, 8)
    empty = integrator.integrate_batch(None, points[:0], jr.key(70), return_details=True, **options)
    assert empty.contribution_sum.shape == (0,)

    for impl in ("rbg", "unsafe_rbg"):
        with pytest.raises(ValueError, match="rbg"):
            integrator.integrate_batch(None, points, jr.key(0, impl=impl), **options)
        with pytest.raises(ValueError, match="rbg"):
            integrator.pre_warm_compaction_buckets(None, points, jr.key(0, impl=impl), **options)


def test_run_chunked_and_average_draws_validate_sizes():
    from sphereWalker import average_draws, run_chunked

    def body(idx, safe_idx, active, carry):
        return carry + 1

    init = jnp.int32(7)
    assert run_chunked(jnp.zeros((0,), bool), 4, body, init) is init
    out = jax.jit(lambda m: run_chunked(m, 4, body, init))(jnp.array([True, False, True]))
    assert int(out) == 8
    for bad in (0, -1, True, 2.0):
        with pytest.raises(ValueError, match="bucket_size"):
            run_chunked(jnp.ones((3,), bool), bad, body, init)

    sample = lambda key: jr.uniform(key)
    np.testing.assert_allclose(average_draws(jr.PRNGKey(0), sample, count=5, chunk_size=2),
                               average_draws(jr.PRNGKey(0), sample, count=5, chunk_size=3), rtol=1e-6)
    for kwargs, name in (({"count": 0}, "count"), ({"count": -2}, "count"), ({"count": True}, "count"),
                         ({"count": 2, "chunk_size": 0}, "chunk_size"),
                         ({"count": 2, "chunk_size": False}, "chunk_size")):
        with pytest.raises(ValueError, match=name):
            average_draws(jr.PRNGKey(0), sample, **kwargs)


def test_process_kernel_reused_across_sample_counts_in_one_bucket():
    surface, params = _unit_disk()
    integrator = WoSIntegrator(surface.empty_sphere, lambda x: jnp.asarray(1.0, x.dtype),
                               lambda x: jnp.asarray(0.0, x.dtype), 2, max_steps=128, eps=1e-3)
    options = dict(init_trust_region_size=1.0, n_samples=16, steps_between_refill=8,
                   max_active_states=8, jit_buck_size=4)
    points = jnp.array([[0.0, 0.0], [0.2, 0.1], [-0.3, 0.2], [0.1, -0.4]], jnp.float32)
    integrator.pre_warm_compaction_buckets(params, points[:3], jr.PRNGKey(71), **options)
    warmed = integrator._process_active_states._cache_size()
    for n in (3, 4, 2):
        out = integrator.integrate_batch(params, points[:n], jr.PRNGKey(72), **options)
        np.testing.assert_allclose(out.w, 1.0, atol=1e-6)
        assert integrator._process_active_states._cache_size() == warmed


@pytest.mark.parametrize("bad_distance", ["zero", "beyond_radius"])
def test_wost_independent_source_ray_distance_out_of_bounds_is_invalid(bad_distance):
    from sphereWalker import WoStParameters

    class BadSourceRays(sdfImplicit):
        """The continuation ray is traced first; later rays are source rays."""

        traced = 0

        def intersect_ray(self, p, origin, direction, max_distance, *, on_boundary=False):
            hit = super().intersect_ray(p, origin, direction, max_distance, on_boundary=on_boundary)
            BadSourceRays.traced += 1
            if BadSourceRays.traced == 1:
                return hit
            distance = (jnp.zeros_like(hit.distance) if bad_distance == "zero"
                        else 2 * max_distance * jnp.ones_like(hit.distance))
            return hit._replace(distance=distance)

    _, _, params, neumann = _annulus_solvers(lambda x: jnp.ones((), x.dtype))
    inner = sdfImplicit(fn=sphere_sdf, dim=2)
    integrator = WoStIntegrator(inner.empty_sphere, lambda x: jnp.zeros((), x.dtype),
                                lambda x: jnp.ones((), x.dtype), 2, max_steps=8, source_samples=3,
                                neumann_surface=BadSourceRays(fn=sphere_sdf, dim=2))
    buf = 4
    positions = jnp.broadcast_to(jnp.array([0.6, 0.0], jnp.float32), (buf, 2))
    states = integrator._get_initial_state(positions, jr.split(jr.PRNGKey(73), buf), 1.0, buf)
    out = _steps(integrator, WoStParameters(params, neumann), states)
    assert BadSourceRays.traced >= 2
    np.testing.assert_array_equal(out.status, WT.INVALID)


def _fluid_solve(**overrides):
    from examples.fluids.dirichlet import make_dirichlet_gradient_solve

    sphere = lambda p, x, tr: (jnp.linalg.norm(x) - 1., 1. - jnp.linalg.norm(x), tr)
    solve = make_dirichlet_gradient_solve(
        sphere, lambda x: x * x, lambda x: jnp.zeros(3, x.dtype), dim=3,
        max_steps=64, gradient_source_samples=1, eps=1e-3)
    options = dict(return_details=True, max_active_states=8, steps_between_refill=4,
                   n_compactions=0, jit_buck_size=4)
    options.update(overrides)
    return solve, options


def test_fluid_pairs_invalid_starts_typed_keys_and_x64_refill_rounds():
    solve, options = _fluid_solve()
    # Interior, exact boundary, and a zero-radius (undecided) start.
    points = jnp.array([[0.1, 0.2, 0.3], [1.0, 0.0, 0.0], [0.0, 0.5, 0.0]], jnp.float32)
    walks = 8
    raw = solve(None, points[:2], jr.PRNGKey(74), 1., walks, **options)
    assert int(raw.successful_samples[0]) > 0
    assert int(raw.successful_samples[1]) == 0
    assert int(raw.termination_counts[1, WT.INVALID]) == walks
    assert int(raw.termination_counts[1].sum()) == walks
    assert float(raw.nSteps[1]) == 0.0
    np.testing.assert_array_equal(raw.contribution_sum[1], 0.0)

    from examples.fluids.dirichlet import make_dirichlet_gradient_solve
    undecided = lambda p, x, tr: (jnp.asarray(-0.5, x.dtype), jnp.zeros((), x.dtype), tr)
    stalled = make_dirichlet_gradient_solve(
        undecided, lambda x: x * x, lambda x: jnp.zeros(3, x.dtype), dim=3,
        max_steps=64, gradient_source_samples=1, eps=1e-3)
    details = stalled(None, points[2:], jr.PRNGKey(74), 1., walks, **options)
    assert int(details.successful_samples[0]) == 0
    assert int(details.termination_counts[0, WT.INVALID]) == walks

    # A typed Threefry key gives the same streams as the raw key.
    typed = solve(None, points[:2], jr.key(74), 1., walks, **options)
    for a, b in zip(jax.tree.leaves(typed), jax.tree.leaves(raw)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    with jax.enable_x64(True):
        solve, options = _fluid_solve(refill_rounds=2, max_active_states=4)
        out = solve(None, points[:1], jr.PRNGKey(75), 1., 16, **options)
        assert int(out.successful_samples[0]) > 0
        assert out.termination_counts.dtype == jnp.int32


def test_re_review_edge_cases_multidraw_python_scalar_simultaneous_stall_and_int_draws():
    # 1. average_draws handles integer and boolean samplers without carry dtype error.
    assert float(average_draws(jr.PRNGKey(80), lambda k: jnp.int32(2), count=3)) == pytest.approx(2.0)
    assert float(average_draws(jr.PRNGKey(81), lambda k: jnp.asarray(True), count=3)) == pytest.approx(1.0)

    # 2. Simultaneous first-ball stall (both clearances 0) takes the smaller retry TR.
    class SlowDirichlet:
        @staticmethod
        def empty_sphere(p, x, tr):
            cleared = tr <= 0.2
            return (jnp.asarray(-0.5, x.dtype),
                    jnp.where(cleared, jnp.asarray(0.15, x.dtype), jnp.asarray(0.0, x.dtype)),
                    0.9 * tr)

    class FastNeumann(sdfImplicit):
        def empty_sphere(self, p, x, tr):
            cleared = tr <= 0.2
            return (jnp.asarray(-0.5, x.dtype),
                    jnp.where(cleared, jnp.asarray(0.15, x.dtype), jnp.asarray(0.0, x.dtype)),
                    0.1 * tr)

    _, _, params, neumann = _annulus_solvers(lambda x: jnp.zeros((), x.dtype))
    grad = WoStGradientIntegrator(
        SlowDirichlet.empty_sphere, lambda x: jnp.zeros((), x.dtype),
        lambda x: jnp.zeros((), x.dtype), 2, max_steps=6, eps=1e-3,
        neumann_surface=FastNeumann(fn=sphere_sdf, dim=2))
    pos = jnp.array([[0.6, 0.0]], jnp.float32)
    state = grad._get_initial_state(pos, jr.split(jr.PRNGKey(82), 1), 1.0, 1)
    out = jax.jit(lambda s: grad._step_block(WoStParameters(params, neumann), s, 3, False))(state)
    assert bool(jnp.any(out.grad_weight[0] != 0))
    assert int(out.status[0]) != WT.MAX_STEPS

    # 3. Under x64: Python scalar source (1e39) and multi-draw mean near float32 max (8e42 * 0.01^2 / 4 = 2e38)
    #    must not overflow before dividing by count.
    with jax.enable_x64(True):
        small_ball = lambda p, x, tr: (jnp.asarray(-0.5, x.dtype), jnp.asarray(0.01, x.dtype), tr)
        wos_py = WoSIntegrator(small_ball, lambda x: jnp.zeros((), x.dtype), lambda x: 1e39, 2,
                               max_steps=4, eps=1e-3)
        st = wos_py._get_initial_state(jnp.array([[0.0, 0.0]], jnp.float32), jr.split(jr.PRNGKey(83), 1), 1.0, 1)
        out_py = _steps(wos_py, None, st)
        assert int(out_py.status[0]) == WT.ACTIVE
        assert bool(jnp.all(jnp.isfinite(out_py.w)))

        _, _, params, neumann = _annulus_solvers(lambda x: jnp.float64(8e42))
        wost_multi = WoStIntegrator(
            small_ball, lambda x: jnp.zeros((), x.dtype), lambda x: jnp.float64(8e42), 2,
            max_steps=4, eps=1e-3, source_samples=3,
            neumann_surface=sdfImplicit(fn=sphere_sdf, dim=2))
        st_w = wost_multi._get_initial_state(pos, jr.split(jr.PRNGKey(84), 1), 1.0, 1)
        out_w = _steps(wost_multi, WoStParameters(params, neumann), st_w)
        assert int(out_w.status[0]) == WT.ACTIVE
        assert out_w.w.dtype == jnp.float32
        assert bool(jnp.all(jnp.isfinite(out_w.w)))
