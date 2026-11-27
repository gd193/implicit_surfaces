"""Regression tests for implicit surface, ray sampler, and material fixes."""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from rayMarcher.camera import Camera
from rayMarcher.materials import Material, scatter
from rayMarcher.sampler import generate_antialiased_ray
from sphereWalker import WoSIntegrator
from sphereWalker.types import WalkerTermination as WT
from surfaces.implicit.harmonic import poissonSurface
from surfaces.implicit.lipschitz import lipschitzImplicit
from surfaces.implicit.primitives import sphere_sdf
from utils.sampling_primitives import (
    sample_uniform_ray_bbox_2d,
    sample_uniform_ray_bbox_3d,
)


def test_classify_box_with_nonzero_level_set():
    # f(x) = ||x|| - 1.0, level_set = 0.5 -> surface is at ||x|| = 1.5.
    surface = lipschitzImplicit(lipschitz_bound=1.0, fn=sphere_sdf, level_set=0.5, dim=3)
    params = {"center": jnp.zeros(3, jnp.float32), "radius": jnp.float32(1.0)}

    # ||x|| = 1.1: f = 0.1, f - 0.5 = -0.4 (inside). Half-diag ~0.087 < 0.4 -> -1.
    lo_in = jnp.array([1.05, -0.05, -0.05], jnp.float32)
    hi_in = jnp.array([1.15, 0.05, 0.05], jnp.float32)
    assert int(surface.classify_box(params, lo_in, hi_in)) == -1

    # ||x|| = 2.0: f = 1.0, f - 0.5 = +0.5 (outside). Half-diag ~0.087 < 0.5 -> +1.
    # (Previously subtracted level_set twice -> f - 1.0 = 0.0 -> sign 0!)
    lo_out = jnp.array([1.95, -0.05, -0.05], jnp.float32)
    hi_out = jnp.array([2.05, 0.05, 0.05], jnp.float32)
    assert int(surface.classify_box(params, lo_out, hi_out)) == 1


def test_poisson_surface_harnack_bound_finite_and_zero_on_level_set():
    # Single monopole at origin: f(x) = 1 / ||x||, level_set = 1.0 -> unit sphere.
    surface = poissonSurface(scale=1.0, offset=0.0, level_set=1.0, dim=3)
    params = {"p": jnp.zeros((1, 3), jnp.float32)}

    # Outside level set (f < level_set): ||x|| = 2 -> f = 0.5; must be finite & > 0.
    fn_out, r_out, _ = surface.empty_sphere(params, jnp.array([2.0, 0.0, 0.0], jnp.float32), jnp.float32(10.0))
    assert np.isfinite(fn_out) and float(fn_out) < 0.0
    assert np.isfinite(r_out) and 0.0 < float(r_out) <= 1.0

    # Inside level set (f > level_set): ||x|| = 0.5 -> f = 2.0; bounded by R_sing = 0.5.
    fn_in, r_in, _ = surface.empty_sphere(params, jnp.array([0.5, 0.0, 0.0], jnp.float32), jnp.float32(10.0))
    assert np.isfinite(fn_in) and float(fn_in) > 0.0
    assert np.isfinite(r_in) and 0.0 < float(r_in) <= 0.5

    # On level set (f == level_set): ||x|| = 1 -> radius must be ~0.
    fn_on, r_on, _ = surface.empty_sphere(params, jnp.array([1.0, 0.0, 0.0], jnp.float32), jnp.float32(10.0))
    assert abs(float(fn_on)) < 1e-6
    assert abs(float(r_on)) < 1e-6



def test_poisson_surface_negative_offset_stays_inside_gap():
    # f(x) = 1/||x|| - 100, level_set = 1 -> surface at ||x|| = 1/101.
    surface = poissonSurface(scale=1.0, offset=-100.0, level_set=1.0)
    params = {"p": jnp.zeros((1, 3), jnp.float32)}
    fn_val, r, next_tr = surface.empty_sphere(params, jnp.array([0.01, 0.0, 0.0], jnp.float32), jnp.float32(10.0))
    gap = 0.01 - 1.0 / 101.0
    assert float(fn_val) < 0.0
    assert 0.0 < float(r) < gap
    assert float(next_tr) == float(r)


def test_poisson_surface_unreachable_level_set_returns_full_ball():
    # level_set below the ball's lower bound: the whole Harnack ball is empty.
    surface = poissonSurface(scale=1.0, offset=0.0, level_set=0.1, alpha=0.5)
    params = {"p": jnp.zeros((1, 3), jnp.float32)}
    _, r, _ = surface.empty_sphere(params, jnp.array([1.0, 0.0, 0.0], jnp.float32), jnp.float32(1e-3))
    np.testing.assert_allclose(float(r), 0.5, rtol=1e-6)


@pytest.mark.parametrize("start", [[0.5, 0.0, 0.0], [0.0, 0.0, -0.3]])
def test_poisson_surface_multi_step_wos_reaches_level_set(start):
    # Unit sphere around one monopole; u = 1 on the boundary, no source.
    surface = poissonSurface(scale=1.0, offset=0.0, level_set=1.0)
    params = {"p": jnp.zeros((1, 3), jnp.float32)}
    integrator = WoSIntegrator(
        empty_sphere_fn=surface.empty_sphere,
        boundary_condition_fn=lambda x: jnp.asarray(1.0, dtype=x.dtype),
        source_term_fn=lambda x: jnp.asarray(0.0, dtype=x.dtype),
        dim=3,
        max_steps=512,
        eps=1e-3,
    )
    details = integrator.integrate_batch(
        params,
        jnp.array([start], jnp.float32),
        jr.PRNGKey(7),
        init_trust_region_size=0.1,
        n_samples=64,
        steps_between_refill=32,
        max_active_states=64,
        jit_buck_size=1,
        return_details=True,
    )
    assert int(details.successful_samples[0]) == 64
    assert int(details.termination_counts[0, WT.MAX_STEPS]) == 0
    assert int(details.termination_counts[0, WT.INVALID]) == 0
    assert float(details.nSteps[0]) > 1.0  # genuinely multi-step


@pytest.mark.parametrize("kwargs", [
    {"dim": 2},
    {"scale": 0.0},
    {"scale": -1.0},
    {"alpha": 1.25},
    {"alpha": 0.0},
])
def test_poisson_surface_rejects_invalid_configuration(kwargs):
    args = {"scale": 1.0, "offset": 0.0, "level_set": 1.0}
    args.update(kwargs)
    with pytest.raises(ValueError):
        poissonSurface(**args)


@pytest.mark.parametrize("shrink", [0.0, 1e-4])
def test_classify_box_rejects_corner_clipping_plane(shrink):
    # Plane f(x) = n . (x - x0) with |n| = 1 placed so its distance to the box
    # center equals (or is slightly below) half_diag: it touches/clips a corner.
    lower = jnp.array([-1.0, -1.0, -1.0], jnp.float32)
    upper = jnp.array([1.0, 1.0, 1.0], jnp.float32)
    half_diag = float(jnp.sqrt(3.0))
    n = jnp.ones(3, jnp.float32) / jnp.sqrt(3.0)

    def plane(params, x):
        return jnp.dot(n, x) - params["d"]

    surface = lipschitzImplicit(lipschitz_bound=1.0, fn=plane, level_set=0.0, dim=3)
    params = {"d": jnp.float32(half_diag * (1.0 - shrink))}
    assert int(surface.classify_box(params, lower, upper)) == 0
    # Well clear of the box: certified.
    assert int(surface.classify_box({"d": jnp.float32(2.0 * half_diag)}, lower, upper)) == -1


def test_uniform_ray_bbox_samplers_finite_and_on_box():
    keys = jr.split(jr.PRNGKey(42), 256)
    lo2, hi2 = jnp.array([-1.0, -2.0], jnp.float32), jnp.array([1.5, 0.5], jnp.float32)
    hit2, d2, p2, tmax2 = jax.vmap(sample_uniform_ray_bbox_2d, in_axes=(0, None, None, None))(
        keys, lo2, hi2, jnp.float32
    )
    assert bool(jnp.any(hit2))
    assert bool(jnp.all(jnp.isfinite(d2[hit2])))
    assert bool(jnp.all(jnp.isfinite(p2[hit2])))
    assert bool(jnp.all(tmax2[hit2] >= 0.0))
    np.testing.assert_allclose(jnp.linalg.norm(d2, axis=-1), 1.0, atol=1e-5)

    lo3, hi3 = jnp.array([-1.0, -1.0, -1.0], jnp.float32), jnp.array([1.0, 2.0, 1.5], jnp.float32)
    hit3, d3, p3, tmax3 = jax.vmap(sample_uniform_ray_bbox_3d, in_axes=(0, None, None, None))(
        keys, lo3, hi3, jnp.float32
    )
    assert bool(jnp.any(hit3))
    assert bool(jnp.all(jnp.isfinite(d3[hit3])))
    assert bool(jnp.all(jnp.isfinite(p3[hit3])))
    assert bool(jnp.all(tmax3[hit3] >= 0.0))
    np.testing.assert_allclose(jnp.linalg.norm(d3, axis=-1), 1.0, atol=1e-5)


def test_camera_jit_and_material_zero_ref_idx():
    cam = Camera.setup_Camera(
        image_width=16,
        image_height=16,
        vfov=45.0,
        lookfrom=jnp.array([0.0, 0.0, 3.0], jnp.float32),
        lookat=jnp.zeros(3, jnp.float32),
        vup=jnp.array([0.0, 1.0, 0.0], jnp.float32),
        defocus_angle=0.0,
    )
    # Also callable on an instance because of @staticmethod.
    cam_defocus = cam.setup_Camera(
        image_width=16,
        image_height=16,
        vfov=45.0,
        lookfrom=jnp.array([0.0, 0.0, 3.0], jnp.float32),
        lookat=jnp.zeros(3, jnp.float32),
        vup=jnp.array([0.0, 1.0, 0.0], jnp.float32),
        defocus_angle=2.0,
    )
    gen = jax.jit(generate_antialiased_ray)
    o0, d0 = gen(cam, jnp.int32(4), jnp.int32(5), jr.PRNGKey(0))
    o1, d1 = gen(cam_defocus, jnp.int32(4), jnp.int32(5), jr.PRNGKey(0))
    assert bool(jnp.all(jnp.isfinite(o0))) and bool(jnp.all(jnp.isfinite(d0)))
    assert bool(jnp.all(jnp.isfinite(o1))) and bool(jnp.all(jnp.isfinite(d1)))

    materials = Material(
        type=jnp.array([0, 1], jnp.int32),
        albedo=jnp.ones((2, 3), jnp.float32),
        fuzz=jnp.array([0.0, 0.1], jnp.float32),
        ref_idx=jnp.array([0.0, 0.0], jnp.float32),
    )
    for idx in (0, 1):
        out = scatter(
            materials,
            idx,
            jnp.array([0.0, 0.0, -1.0], jnp.float32),
            jnp.array([0.0, 0.0, 1.0], jnp.float32),
            jr.PRNGKey(idx),
        )
        assert bool(jnp.all(jnp.isfinite(out[0])))
