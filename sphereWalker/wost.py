"""Walk on Stars with an implicit Dirichlet query and a Neumann enclosure.

Solves -Delta u = f, u = g on the Dirichlet surface, du/dn = h on
Neumann surfaces supporting star_radius and intersect_ray. Homogeneous h is
general; nonzero outward flux uses a specialized analytic-box estimator.
Geometric derivatives are deliberately unsupported.
"""

import math

import jax.numpy as jnp
import jax.random as jr

from sphereWalker.gradient import FirstBallGradientMixin
from sphereWalker.ops import average_draws, finish_step, walker_status
from sphereWalker.types import GradientStarWalkerState, StarWalkerState, WoStParameters
from sphereWalker.types import WalkerTermination as WT
from sphereWalker.walk import WalkIntegrator, check_broadcast
from surfaces.implicit.base import ImplicitSurface
from surfaces.implicit.lipschitz import sdfImplicit
from surfaces.implicit.primitives.sdfnd import bbox_sdf, box_contact_count
from utils.sampling_primitives import (
    sample_box_boundary_in_ball,
    sample_green_radius,
    weighted_greens_fn_box_boundary,
)
from utils.types import Termination as T


class WoStIntegrator(WalkIntegrator[StarWalkerState]):
    """Same batch driver and Dirichlet/source callbacks as WoS.

    neumann_condition_fn(problem_params, position, outward_normal) supplies
    outward derivative data, scalar or broadcastable to the boundary shape.
    Omit it for homogeneous data. Nonzero flux currently supports analytic boxes.

    Pass neumann_params explicitly to integrate_batch/pre_warm_compaction_buckets.
    Initial points may be interior or on supported Neumann contacts (including
    box edges/corners). Unresolved queries fail explicitly. eps retains the
    current WoS field-residual termination convention, not a distance certificate.
    """

    def __init__(self, empty_sphere_fn, boundary_condition_fn, source_term_fn, dim, *,
                 neumann_surface, max_radius=math.inf, neumann_condition_fn=None, **kwargs):
        if neumann_surface.dim != dim:
            raise ValueError("Neumann surface dimension must match the integrator")
        for name in ("star_radius", "intersect_ray"):
            if getattr(type(neumann_surface), name, None) is getattr(ImplicitSurface, name):
                raise NotImplementedError(
                    f"{type(neumann_surface).__name__} does not implement {name}")
        if (neumann_condition_fn is not None
                and not (type(neumann_surface) is sdfImplicit and neumann_surface.fn is bbox_sdf)):
            raise NotImplementedError(
                "nonzero Neumann data currently requires an analytic box sampling estimator")
        if math.isnan(max_radius) or max_radius <= 0:
            raise ValueError("max_radius must be positive")
        super().__init__(empty_sphere_fn, boundary_condition_fn, source_term_fn, dim, **kwargs)
        self.neumann_surface = neumann_surface
        self.max_radius = max_radius
        self.neumann_condition_fn = neumann_condition_fn

    def integrate_batch(self, params, initial_position_batch, rng_key, init_trust_region_size,
                        n_samples, *args, neumann_params, **kwargs):
        return super().integrate_batch(
            WoStParameters(params, neumann_params), initial_position_batch, rng_key,
            init_trust_region_size, n_samples, *args, **kwargs)

    def pre_warm_compaction_buckets(self, params, initial_position_batch, rng_key,
                                    init_trust_region_size, n_samples, *args, neumann_params, **kwargs):
        return super().pre_warm_compaction_buckets(
            WoStParameters(params, neumann_params), initial_position_batch, rng_key,
            init_trust_region_size, n_samples, *args, **kwargs)

    def _problem_params(self, params: WoStParameters):
        return params.problem

    def _get_initial_state(self, init_position_batch, rng_keys, init_tr, buffer_size):
        dtype = init_position_batch.dtype
        return StarWalkerState(
            position=init_position_batch,
            nSteps=jnp.zeros(buffer_size, jnp.int32),
            w=jnp.zeros((buffer_size,) + self._value_shape(dtype), dtype),
            trust_region_size=jnp.full((buffer_size,), init_tr, dtype),
            rng_key=rng_keys,
            done_flag=jnp.zeros(buffer_size, bool),
            status=jnp.zeros(buffer_size, jnp.int32),
            on_neumann=jnp.zeros(buffer_size, bool),
            normal=jnp.zeros_like(init_position_batch),
        )

    def _first_ball_clearance(self, params, state, query):
        # A first ball must clear both boundaries; starts on either are invalid.
        # Validate both clearances before clipping: min(+inf, nclearance) would
        # otherwise hide an invalid Dirichlet radius.
        nvalue, nclearance, n_next_tr = self.neumann_surface.empty_sphere(
            params.neumann, state.position, state.trust_region_size)
        valid = (jnp.isfinite(query.radius) & (query.radius >= 0)
                 & jnp.isfinite(nvalue) & (nvalue < 0)
                 & jnp.isfinite(nclearance) & (nclearance >= 0))
        radius = jnp.minimum(jnp.minimum(query.radius, nclearance), self.max_radius)
        # A limiting (or simultaneously stalled) Neumann enclosure proposes the retry region.
        next_tr = jnp.where(
            nclearance < query.radius,
            n_next_tr,
            jnp.where(nclearance == query.radius,
                      jnp.minimum(n_next_tr, query.next_trust_region),
                      query.next_trust_region),
        )
        return radius, valid, next_tr

    def _box_flux_contribution(self, params, state, radius):
        """Nonzero Neumann estimator for the supported analytic box."""
        # Independent boundary sampling (WoSt 2023, Eq. 18/Algorithm 1).
        # Positive Green function for -Lap u=f and OUTWARD derivative h
        # gives +G*h/(alpha*pdf); the paper uses the opposite h convention.
        # fold_in leaves homogeneous walk directions/source samples intact.
        sample = sample_box_boundary_in_ball(
            jr.fold_in(state.rng_key, 0x4E4555), state.position, radius,
            params.neumann['min_corner'], params.neumann['max_corner'])
        # alpha is the fraction of the sphere inside the reflecting box.
        contacts = box_contact_count(params.neumann, state.position)
        weight = weighted_greens_fn_box_boundary(sample, radius) * jnp.exp2(
            contacts.astype(state.position.dtype))
        raw_flux = self.neumann_condition_fn(params.problem, sample.position, sample.normal)
        flux = jnp.asarray(raw_flux, dtype=jnp.result_type(raw_flux))
        check_broadcast(flux.shape, state.w.shape, 'Neumann flux')
        # Weight in the promoted precision, then cast the contribution.
        return jnp.where(weight > 0, weight * flux, 0.).astype(state.w.dtype)

    def _continuation_step(self, params, state, query):
        residual, clearance, next_tr = query
        status = walker_status(state, residual, clearance, next_tr, self.eps, self.max_steps)
        active = status == WT.ACTIVE
        dtype = state.position.dtype
        # Recognize exact wall points every step, not only reported hits: starts,
        # tangent first-ball samples, and clear rays whose endpoint rounds onto
        # the wall (~1e-8 per step, enough to fail long solves).
        value = self.neumann_surface.signed_value(params.neumann, state.position)
        on_boundary = state.on_neumann | (value == 0)
        initial_normal = self.neumann_surface.normal_at(params.neumann, state.position)
        normal = jnp.where(state.on_neumann, state.normal, initial_normal)
        normal_valid = jnp.all(jnp.isfinite(normal)) & (jnp.abs(jnp.linalg.norm(normal) - 1) < 1e-4)
        cap = jnp.minimum(clearance, self.max_radius)
        # Zero clearance is an undecided Dirichlet query: like WoS, stay put
        # and retry with the next trust region.
        stalled = active & (cap == 0)
        query_cap = jnp.where(active & ~stalled, cap, 1.0)
        star = self.neumann_surface.star_radius(
            params.neumann, state.position, query_cap, on_boundary=on_boundary, normal=normal)
        geometry_valid = (
            (star.status == T.EXIT)
            & jnp.isfinite(star.radius)
            & (star.radius > 0)
            & (star.radius <= query_cap)
            & (~on_boundary | normal_valid)
        )
        # Containment applies even to terminal Dirichlet and stalled points.
        status = jnp.where(~geometry_valid, WT.INVALID, status)
        moving = (status == WT.ACTIVE) & ~stalled
        radius = jnp.where(moving, star.radius, 1.0)
        direction_key, source_key, next_key = jr.split(state.rng_key, 3)
        direction = self.sphere_sampling_fn(direction_key, dtype=dtype)
        inward = self.neumann_surface.inward_direction(params.neumann, state.position, direction, normal)
        direction = jnp.where(on_boundary, inward, direction)
        hit = self.neumann_surface.intersect_ray(
            params.neumann, state.position, direction, radius, on_boundary=on_boundary)
        hit_valid = (hit.status == T.HIT) | (hit.status == T.EXIT)
        hit_valid &= jnp.isfinite(hit.distance) & (hit.distance > 0) & (hit.distance <= radius)
        hit_valid &= jnp.all(jnp.isfinite(hit.position))
        hit_valid &= (hit.status != T.HIT) | (
            jnp.all(jnp.isfinite(hit.normal)) & (jnp.abs(jnp.linalg.norm(hit.normal) - 1) < 1e-4))
        status = jnp.where(moving & ~hit_valid, WT.INVALID, status)
        active = status == WT.ACTIVE
        advance = active & ~stalled
        problem = params.problem
        green = radius ** 2 / (2 * self.dim)
        if self.source_samples == 1:
            source_distance = radius * sample_green_radius(source_key, self.dim, dtype)
            # Don't evaluate f outside its domain, even when its weight would be zero.
            inside_star = source_distance < hit.distance
            source_position = jnp.where(inside_star & advance,
                                        state.position + source_distance * direction, state.position)
            value = self._source_components(problem, source_position, state.w.shape)
            contribution = jnp.where(inside_star, value * green, 0.0).astype(state.w.dtype)
        else:
            def sample_source(key):
                kr, kd = jr.split(key)
                d = self.sphere_sampling_fn(kd, dtype=dtype)
                d = jnp.where(on_boundary, self.neumann_surface.inward_direction(
                    params.neumann, state.position, d, normal), d)
                source_hit = self.neumann_surface.intersect_ray(
                    params.neumann, state.position, d, radius, on_boundary=on_boundary)
                distance = radius * sample_green_radius(kr, self.dim, dtype)
                inside = distance < source_hit.distance
                x = jnp.where(inside & advance, state.position + distance * d, state.position)
                value = self._source_components(problem, x, state.w.shape)
                # Same ray checks as the continuation ray.
                valid = (((source_hit.status == T.HIT) | (source_hit.status == T.EXIT))
                         & jnp.isfinite(source_hit.distance)
                         & (source_hit.distance > 0) & (source_hit.distance <= radius))
                return jnp.where(valid, jnp.where(inside, value * green, 0.), jnp.nan)
            contribution = average_draws(
                source_key, sample_source, count=self.source_samples).astype(state.w.dtype)
        if self.neumann_condition_fn is not None:
            contribution += self._box_flux_contribution(params, state, radius)
        updated = state._replace(
            position=jnp.where(advance, hit.position, state.position),
            # Stalls count toward max_steps, as in WoS.
            nSteps=state.nSteps + active.astype(jnp.int32),
            w=jnp.where(advance, state.w + contribution, state.w),
            trust_region_size=jnp.where(active, next_tr, state.trust_region_size),
            rng_key=next_key,
            on_neumann=jnp.where(advance, hit.status == T.HIT, state.on_neumann),
            normal=jnp.where(advance, hit.normal, state.normal),
            status=jnp.asarray(status, jnp.int32),
            done_flag=status != WT.ACTIVE,
        )
        return finish_step(state, updated)


class WoStGradientIntegrator(FirstBallGradientMixin, WoStIntegrator):
    """WoSt estimate of grad u; see FirstBallGradientMixin.

    The first radius is bounded by both Dirichlet and Neumann clearance (and
    max_radius). Output shape follows WoSGradientIntegrator.
    """

    _gradient_state_cls = GradientStarWalkerState
