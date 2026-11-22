"""Tests for WoSIntegrator vector PDEs, DifferentialWoS chunked evaluation, and refill-round warmup."""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from sphereWalker.integrator import DifferentialWoSIntegrator, WoSIntegrator
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
