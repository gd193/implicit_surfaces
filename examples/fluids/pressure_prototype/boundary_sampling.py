"""Specialized sampling for the experimental pressure estimator.

Supported domains: closed analytic box/sphere and exterior analytic sphere/torus.
These adapters own sampling PDFs, normals, and signed transition weights. They
do not extend the generic Surface API. The private implicit-hit helper exists
only to implement the torus adapter using existing SurfaceSampler queries.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from surfaceSampler.sampler import SurfaceSampler
from utils.sampling_primitives import sample_uniform_ray_bbox_3d
from utils.types import RayBuffer, Termination


class BoundaryHits(NamedTuple):
    points: jax.Array
    normals: jax.Array
    valid: jax.Array
    patch: jax.Array       # 0: box face, 1: implicit wall
    complete: jax.Array    # per line; failed queries must never be treated as misses


class BoxBoundary:
    """Closed analytic box, outward normals pointing out of the fluid."""

    def __init__(self, lower, upper):
        lo, hi = np.asarray(lower, np.float32), np.asarray(upper, np.float32)
        if (lo.shape != (3,) or hi.shape != (3,) or
                not np.isfinite([lo, hi]).all() or not np.all(lo < hi)):
            raise ValueError("boundary requires finite 3D lower < upper")
        self.lower, self.upper = jnp.asarray(lo), jnp.asarray(hi)
        size = hi - lo
        self.box_area = float(2 * (size[0]*size[1] + size[0]*size[2] + size[1]*size[2]))
        self._box_hits = jax.jit(self._intersect_box)

    def contains(self, x):
        return jnp.all((x > self.lower) & (x < self.upper))

    def _intersect_box(self, origins, directions):
        inv = jnp.where(jnp.abs(directions) > 1e-20, 1 / directions, jnp.inf)
        a, b = (self.lower-origins)*inv, (self.upper-origins)*inv
        near, far = jnp.minimum(a, b), jnp.maximum(a, b)
        times = jnp.stack([jnp.max(near, -1), jnp.min(far, -1)], -1)
        points = origins[:, None, :] + times[..., None]*directions[:, None, :]
        axes = jnp.stack([jnp.argmax(near, -1), jnp.argmin(far, -1)], -1)
        signs = jnp.take_along_axis(jnp.sign(directions), axes, axis=1)
        normals = jax.nn.one_hot(axes, 3, dtype=origins.dtype) * (signs*jnp.array([-1., 1.], origins.dtype))[..., None]
        points = jnp.where(jax.nn.one_hot(axes, 3).astype(bool),
                           jnp.where(normals > 0, self.upper, self.lower),
                           jnp.clip(points, self.lower, self.upper))
        valid = jnp.broadcast_to((times[:, :1] < times[:, 1:]) &
                                jnp.all(jnp.isfinite(points), axis=(1, 2))[:, None], times.shape)
        return BoundaryHits(points, normals, valid, jnp.zeros(valid.shape, jnp.int32),
                            jnp.all(jnp.isfinite(origins) & jnp.isfinite(directions), -1))

    def intersections(self, origins, directions):
        """All intersections of unoriented lines, including behind the origin."""
        return self._box_hits(jnp.asarray(origins), jnp.asarray(directions))

    def walk_origins(self, points):
        # Float32 edge contacts may round onto two box faces. An unoriented
        # line through that rounded edge can have a zero-length box interval.
        points = jnp.asarray(points)
        inset = 4*jnp.finfo(points.dtype).eps*(self.upper-self.lower)
        return jnp.clip(points, self.lower+inset, self.upper-inset)

    def other_intersections(self, points, directions, normals):
        hits = self.intersections(self.walk_origins(points), directions)
        distance = jnp.linalg.norm(hits.points-points[:, None, :], axis=-1)
        own = jnp.argmin(jnp.where(hits.valid, distance, jnp.inf), -1)
        valid = hits.valid.at[jnp.arange(len(points)), own].set(False)
        return hits._replace(valid=valid)

    def prepare_contacts(self, points, normals, patch):
        return points, normals

    def volume_radius(self, x):
        return jnp.linalg.norm(jnp.maximum(jnp.abs(self.lower-x), jnp.abs(self.upper-x)))

    def sample_area_boundary(self, key, dtype):
        kf, kp = jax.random.split(key)
        size = self.upper-self.lower
        areas = jnp.array([size[1]*size[2], size[0]*size[2], size[0]*size[1]])
        cumulative = jnp.cumsum(jnp.repeat(areas, 2))
        draw = jax.random.uniform(kf, (), dtype=dtype)*cumulative[-1]
        face = jnp.sum(draw >= cumulative, dtype=jnp.int32)
        axis, sign = face//2, 2*(face % 2)-1
        x = jax.random.uniform(kp, (3,), dtype=dtype, minval=self.lower, maxval=self.upper)
        x = x.at[axis].set(jnp.where(sign > 0, self.upper[axis], self.lower[axis]))
        return x, jax.nn.one_hot(axis, 3, dtype=dtype)*sign, jnp.asarray(self.box_area, dtype)


    def _choose(self, hits, key):
        valid = hits.valid[0]
        count = jnp.sum(valid)
        rank = jax.random.randint(key, (), 0, jnp.maximum(count, 1))
        index = jnp.argmax(jnp.cumsum(valid) > rank)
        point, normal = hits.points[:, index], hits.normals[:, index]
        point, normal = self.prepare_contacts(point, normal, hits.patch[:, index])
        return point[0], normal[0], count, jnp.all(hits.complete)

    def sample_uniform_boundary(self, key, dtype):
        def draw(carry):
            k, _, _, _, i = carry
            k, subkey = jax.random.split(k)
            hit, d, o, _ = sample_uniform_ray_bbox_3d(subkey, self.lower, self.upper, dtype)
            return k, hit, d, o, i+1
        k, hit, d, o, _ = jax.lax.while_loop(
            lambda c: ~c[1] & (c[4] < 10000), draw,
            (key, jnp.array(False), jnp.zeros(3, dtype), jnp.zeros(3, dtype), jnp.int32(0)))
        point, normal, count, valid = self._choose(self.intersections(o[None], d[None]), k)
        return point, normal, count*self.box_area/2, valid & hit

    def next_boundary(self, point, normal, key):
        kd, ki = jax.random.split(key)
        direction = jax.random.normal(kd, (3,), dtype=point.dtype)
        direction /= jnp.linalg.norm(direction)
        hits = self.other_intersections(point[None], direction[None], normal[None])
        p, n, count, valid = self._choose(hits, ki)
        sign = jnp.where(jnp.dot(n, p-point) > 0, 1., -1.)
        return p, n, sign*count, valid & (count > 0)


class _ImplicitBoxBoundary(BoxBoundary):
    """Negative implicit fluid clipped by a box, using existing multi-hit queries.

    epsilon is a field-value tolerance; refinement_limit caps each Newton
    correction in world space. Directional contact offsets are a numerical
    geometry approximation proportional to epsilon. The continuation bound is
    used only inside hit bands; empty-space stepping uses surface.ray_step.
    """

    def __init__(self, surface, params, lower, upper, *, lipschitz_bound, update_tr_fn,
                 epsilon=1e-5, refinement_limit=1e-3, max_steps=2048,
                 max_hits=64, max_active_states=4096):
        if type(self) is _ImplicitBoxBoundary:
            raise NotImplementedError('generic implicit boundary sampling is not supported')
        super().__init__(lower, upper)
        if surface.dim != 3 or not np.isfinite(refinement_limit) or refinement_limit <= 0:
            raise ValueError("requires a 3D surface and positive refinement limit")
        self.surface, self.params = surface, params
        self.refinement_limit = refinement_limit
        self.sampler = SurfaceSampler.from_surface(
            surface, epsilon=epsilon, max_step=max_steps, max_hits=max_hits,
            lipschitz_bound=lipschitz_bound, max_active_states=max_active_states,
            n_compactions=4, crossings_only=True, update_tr_fn=update_tr_fn)
        self._values = jax.jit(jax.vmap(lambda x: surface.signed_value(params, x)))
        gradient = jax.grad(lambda x: surface.signed_value(params, x))

        @jax.jit
        def refine(points, directions):
            # Accepted band entries are not roots. Refine along the same line,
            # bounding each Newton correction to refinement_limit.
            def step(_, p):
                f = jax.vmap(lambda x: surface.signed_value(params, x))(p)
                grad = jax.vmap(gradient)(p)
                slope = jnp.sum(grad*directions, -1)
                correction = jnp.where(jnp.abs(slope) > 1e-10, f/slope, 0.)
                return p-jnp.clip(correction, -refinement_limit, refinement_limit)[:, None]*directions
            points = jax.lax.fori_loop(0, 4, step, points)
            grad = jax.vmap(gradient)(points)
            return points, grad/jnp.maximum(jnp.linalg.norm(grad, axis=-1, keepdims=True), 1e-20)
        self._refine = refine

        @jax.jit
        def prepare_contacts(points, normals, patch):
            # At a rounded box/implicit junction there is no unique normal.
            # Retract contacts along their own boundary patch by O(epsilon),
            # away from its edge. This is an explicit quadrature tolerance
            # bias; the implicit field and certified empty-space queries stay
            # unchanged. It prevents subsequent walks starting in a sub-ulp
            # fluid wedge with no representable interior.
            margin = jnp.asarray(2*epsilon/lipschitz_bound, points.dtype)
            def update(_, p):
                grad = jax.vmap(gradient)(p)
                length2 = jnp.maximum(jnp.sum(grad*grad, -1), 1e-20)
                unit = grad/jnp.sqrt(length2[:, None])
                distances = jnp.concatenate([p-self.lower, self.upper-p], -1)
                face = jnp.argmin(distances, -1)
                inward = jax.nn.one_hot(face % 3, 3)*jnp.where(face < 3, 1., -1.)[:, None]
                tangent = inward-jnp.sum(inward*unit, -1, keepdims=True)*unit
                clearance = jnp.min(distances, -1)
                shift = jnp.maximum(margin-clearance, 0.)/jnp.maximum(jnp.sum(tangent*tangent, -1), 1e-12)
                implicit_point = p+jnp.minimum(shift, refinement_limit)[:, None]*tangent
                value = jax.vmap(lambda x: surface.signed_value(params, x))(implicit_point)
                implicit_point -= (value/length2)[:, None]*grad
                # For a box contact, correct F tangentially to its face.
                tangent = grad-jnp.sum(grad*normals, -1, keepdims=True)*normals
                value = jax.vmap(lambda x: surface.signed_value(params, x))(p)
                shift = jnp.maximum(value+epsilon, 0.)/jnp.maximum(jnp.sum(tangent*tangent, -1), 1e-12)
                box_point = p-jnp.minimum(shift, refinement_limit)[:, None]*tangent
                box_point = jnp.where(jnp.abs(normals) > .5, p,
                                      jnp.clip(box_point, self.lower+margin, self.upper-margin))
                return jnp.where((patch == 0)[:, None], box_point, implicit_point)
            points = jax.lax.fori_loop(0, 6, update, points)
            grad = jax.vmap(gradient)(points)
            normal = grad/jnp.maximum(jnp.linalg.norm(grad, axis=-1, keepdims=True), 1e-20)
            return points, jnp.where((patch == 0)[:, None], normals, normal)
        self._prepare_contacts = prepare_contacts


    def prepare_contacts(self, points, normals, patch):
        return self._prepare_contacts(points, normals, patch)

    def other_intersections(self, points, directions, normals):
        # Trace both half-lines from opposite sides of the known contact.
        # A single fluid-side offset can miss a concave wall at grazing angles.
        # The normal offset follows the outgoing direction, as in surface ray
        # tracing, so neither half-ray starts by re-hitting its own contact.
        points, directions, normals = map(jnp.asarray, (points, directions, normals))
        side = jnp.where(jnp.sum(normals*directions, -1) >= 0, 1., -1.)
        lipschitz_bound = jnp.asarray(self.sampler.lipschitz_bound, points.dtype)
        offset = .25*self.sampler.epsilon/lipschitz_bound
        box_clearance = jnp.min(jnp.minimum(points-self.lower, self.upper-points), -1)
        field_clearance = jnp.abs(self._values(points))/lipschitz_bound
        on_box = box_clearance <= 0
        # Do not bias a ray across the *other* wall at a box/implicit seam.
        # A sub-ulp offset rounds to zero; endpoint signs still determine the
        # short crossing, rather than deleting the narrow fluid interval.
        other_clearance = jnp.where(on_box, field_clearance, box_clearance)
        offset = jnp.minimum(offset, .25*jnp.maximum(other_clearance, 0.))
        delta = offset[:, None]*side[:, None]*normals
        origins = jnp.concatenate([points+delta, points-delta])
        dirs = jnp.concatenate([directions, -directions])
        hits = self._trace_intersections(origins, dirs, forward_only=True)
        def combine(a):
            return jnp.concatenate([a[:len(points)], a[len(points):]], axis=1)
        return BoundaryHits(*(combine(a) for a in hits[:4]),
                            hits.complete[:len(points)] & hits.complete[len(points):])

    def contains(self, x):
        return super().contains(x) & (self.surface.signed_value(self.params, x) < 0)

    def intersections(self, origins, directions):
        return self._trace_intersections(origins, directions, forward_only=False)

    def _trace_intersections(self, origins, directions, *, forward_only):
        origins, directions = jnp.asarray(origins), jnp.asarray(directions)
        box = self._box_hits(origins, directions)
        start, end = box.points[:, 0], box.points[:, 1]
        ray_valid = box.valid[:, 0]
        if forward_only:
            entry = jnp.sum((start-origins)*directions, -1)
            exit = jnp.sum((end-origins)*directions, -1)
            ray_valid = ray_valid & (exit > jnp.maximum(entry, 0.))
            start = origins+jnp.maximum(entry, 0.)[:, None]*directions
            box = box._replace(valid=box.valid & (jnp.stack([entry, exit], -1) > 0.))
        length = jnp.where(ray_valid, jnp.linalg.norm(end-start, axis=-1), 0.)
        rays = RayBuffer(start, directions, length)
        state = self.sampler._initialize(rays, jnp.arange(len(start)), jnp.ones(len(start), bool))
        # Reuse the sampler's actual query/continuation step inside the compiled
        # walker, rather than launching a separate Python traversal driver.
        result = jax.lax.while_loop(lambda s: jnp.any(~s.done_flag),
                                    lambda s: self.sampler.step_states_fn(self.params, s), state)
        complete = jnp.all(result.status == Termination.EXIT)
        ray_valid &= complete
        valid = (jnp.arange(self.sampler.max_hits)[None, :] < result.hit_counts[:, None]) & ray_valid[:, None]
        raw = start[:, None, :]+result.t_hits[..., None]*directions[:, None, :]
        dirs = jnp.broadcast_to(directions[:, None, :], raw.shape)
        points, normals = self._refine(raw.reshape(-1, 3), dirs.reshape(-1, 3))
        points, normals = points.reshape(raw.shape), normals.reshape(raw.shape)
        # Retain distinct crossings even arbitrarily close to a box face.
        internal = jnp.all((points >= self.lower) & (points <= self.upper), -1)
        box_valid = self._values(box.points.reshape(-1, 3)).reshape(-1, 2) < 0
        return BoundaryHits(jnp.concatenate([box.points, points], 1),
                            jnp.concatenate([box.normals, normals], 1),
                            jnp.concatenate([box.valid & box_valid & ray_valid[:, None], valid & internal], 1),
                            jnp.concatenate([box.patch, jnp.ones(valid.shape, jnp.int32)], 1),
                            box.complete & complete)


class SphereBoundary:
    """Thin sampling adapter for the existing analytic sphere surface queries."""
    def __init__(self, center=(0., 0., 0.), radius=1.):
        from surfaces.implicit.lipschitz import sdfImplicit
        from surfaces.implicit.primitives import sphere_sdf
        if radius <= 0 or not np.isfinite(radius):
            raise ValueError('positive finite sphere radius required')
        self.surface = sdfImplicit(fn=sphere_sdf, dim=3)
        self.params = {'center': jnp.asarray(center, jnp.float32), 'radius': jnp.float32(radius)}
        self.lower, self.upper = self.params['center']-radius, self.params['center']+radius

    def contains(self, x):
        return self.surface.signed_value(self.params, x) < 0

    def volume_radius(self, x):
        return jnp.linalg.norm(x-self.params['center'])+self.params['radius']

    def sample_uniform_boundary(self, key, dtype):
        from utils.sampling_primitives import sample_uniform_2sphere
        normal = sample_uniform_2sphere(key, dtype=dtype)
        r = self.params['radius']
        return self.params['center']+r*normal, normal, 4*jnp.pi*r*r, jnp.array(True)

    def next_boundary(self, point, normal, key):
        from utils.sampling_primitives import sample_uniform_2sphere
        direction = sample_uniform_2sphere(key, dtype=point.dtype)
        return self._next_boundary_in_direction(point, direction)

    def _next_boundary_in_direction(self, point, direction):
        delta = point-self.params['center']
        # Reprojection to world coordinates rounds the contact. Fold against
        # that actual contact, rather than its cached pre-rounding normal;
        # otherwise a grazing direction can become outward and have t=0.
        direction = self.surface.inward_direction(self.params, point, direction, delta)
        scale = jnp.maximum(jnp.max(jnp.abs(point)), self.params['radius'])
        slack = 8*jnp.finfo(point.dtype).eps*scale
        hit = self.surface.intersect_ray(self.params, point, direction,
                                         2*self.params['radius']+slack, on_boundary=True)
        tangent = ((jnp.dot(delta, direction) == 0)
                   & jnp.all(jnp.isfinite(direction))
                   & (jnp.abs(jnp.linalg.norm(delta)-self.params['radius']) <= slack))
        # An exactly tangent line has no second intersection. As in the
        # reference's origin-excluding query, it contributes zero and exits.
        return (jnp.where(tangent, point, hit.position),
                jnp.where(tangent, delta/jnp.linalg.norm(delta), hit.normal),
                jnp.where(tangent, 0., 1.).astype(point.dtype),
                tangent | (hit.status == Termination.HIT))


class ExteriorSphereBoundary(SphereBoundary):
    """Sphere obstacle in an unbounded fluid, truncated for volume integration.

    The box is a pseudo-boundary, never a reflecting wall. Sphere normals point
    out of the fluid (into the obstacle). Reuses analytic sphere ray queries.
    """
    def __init__(self, center, radius, lower, upper):
        super().__init__(center, radius)
        self.box = BoxBoundary(lower, upper)
        self.lower, self.upper = self.box.lower, self.box.upper
        c = np.asarray(self.params['center'])
        if np.any(c-radius <= np.asarray(lower)) or np.any(c+radius >= np.asarray(upper)):
            raise ValueError('sphere must lie strictly inside the integration box')

    def contains(self, x):
        return self.box.contains(x) & (self.surface.signed_value(self.params, x) > 0)

    def wall_clearance(self, x, trust_region_size=.5):
        return self.surface.signed_value(self.params, x)

    def volume_radius(self, x):
        return self.box.volume_radius(x)

    def sample_uniform_boundary(self, key, dtype):
        x, normal, area, valid = super().sample_uniform_boundary(key, dtype)
        return x, -normal, area, valid

    def next_boundary(self, point, normal, key):
        x, n, weight, valid = super().next_boundary(point, -normal, key)
        return x, -n, -weight, valid

    def sample_pseudo_boundary(self, key, dtype):
        return self.box.sample_area_boundary(key, dtype)


class ExteriorTorusBoundary(_ImplicitBoxBoundary):
    """Analytic torus obstacle; reuse the implicit sampler for all line hits.

    Requires JAX float64 support for the contact polynomial; fields stay float32.
    The symmetry axis is world y. The box truncates integration and does not
    participate in boundary walks. Root samples are exactly uniform in area.
    """
    def __init__(self, center, major, minor, lower, upper):
        if not jax.config.x64_enabled:
            raise ValueError("ExteriorTorusBoundary requires jax_enable_x64 for its geometry roots")
        from surfaces.implicit.lipschitz import sdfImplicit
        from surfaces.implicit.primitives import torus_sdf
        c = np.asarray(center, np.float32)
        if c.shape != (3,) or not np.isfinite(c).all() or not 0 < minor < major:
            raise ValueError('finite center and 0 < minor < major required')
        extent = np.array([major+minor, minor, major+minor])
        if np.any(c-extent <= lower) or np.any(c+extent >= upper):
            raise ValueError('torus must lie strictly inside the integration box')
        self.center = jnp.asarray(c)
        self.major, self.minor = float(major), float(minor)
        order = jnp.array([0, 2, 1])
        surface = sdfImplicit(fn=lambda p, x: torus_sdf(p, x[order]), dim=3)
        params = {'center': self.center[order], 'major': major, 'minor': minor}
        super().__init__(surface, params, lower, upper, lipschitz_bound=1.,
                         update_tr_fn=surface.update_ray_trust_region,
                         epsilon=1e-5, refinement_limit=.001, max_steps=4096, max_hits=8)
        self.box = BoxBoundary(lower, upper)

    def contains(self, x):
        return self.box.contains(x) & (self.surface.signed_value(self.params, x) > 0)

    def wall_clearance(self, x, trust_region_size=.5):
        return self.surface.signed_value(self.params, x)

    def sample_uniform_boundary(self, key, dtype):
        kt, kp = jax.random.split(key)
        def draw(carry):
            k, _, _ = carry
            k, ka, ku = jax.random.split(k, 3)
            theta = jax.random.uniform(ka, (), dtype=dtype, maxval=2*jnp.pi)
            accept = jax.random.uniform(ku, (), dtype=dtype) < (
                self.major+self.minor*jnp.cos(theta))/(self.major+self.minor)
            return k, theta, accept
        _, theta, _ = jax.lax.while_loop(lambda c: ~c[2], draw,
                                         (kt, jnp.array(0., dtype), jnp.array(False)))
        phi = jax.random.uniform(kp, (), dtype=dtype, maxval=2*jnp.pi)
        normal = jnp.array([jnp.cos(theta)*jnp.cos(phi), jnp.sin(theta),
                            jnp.cos(theta)*jnp.sin(phi)])
        ring = jnp.array([self.major*jnp.cos(phi), 0., self.major*jnp.sin(phi)])
        point = self.center+ring+self.minor*normal
        return point, -normal, jnp.asarray(4*jnp.pi**2*self.major*self.minor, dtype), jnp.array(True)

    def prepare_contacts(self, points, normals, patch):
        # The obstacle is strictly inside the box; no box/implicit seam exists.
        return points, normals

    def _trace_intersections(self, origins, directions, *, forward_only):
        hits = super()._trace_intersections(origins, directions, forward_only=forward_only)
        return hits._replace(valid=hits.valid & (hits.patch == 1), normals=-hits.normals)

    def _contact_line_hits(self, point, direction):
        from jnp_bounds.polynomial_roots import RootStatus, real_roots
        # Near-coincident roots can fail a float32 discriminant/residual check
        # after fusion. Only the four polynomial coefficients and three roots
        # use float64; contacts, fields and projection workspaces stay float32.
        with jax.enable_x64(True):
            p = point.astype(jnp.float64)-self.center.astype(jnp.float64)
            d = direction.astype(jnp.float64)
            major = jnp.asarray(self.major, point.dtype).astype(jnp.float64)
            minor = jnp.asarray(self.minor, point.dtype).astype(jnp.float64)
            a = jnp.dot(d, d)
            b = 2*jnp.dot(p, d)
            c = jnp.dot(p, p)+major**2-minor**2
            radial_a = d[0]**2+d[2]**2
            radial_b = p[0]*d[0]+p[2]*d[2]
            # Deflate the known contact Q(0)=0 from the torus quartic Q(t).
            coefficients = jnp.stack([2*b*c-8*major**2*radial_b,
                                      b*b+2*a*c-4*major**2*radial_a,
                                      2*a*b, a*a])
            result = real_roots(coefficients)
            t64 = jnp.where(result.valid, result.roots, 0.)
            derivative = coefficients[1]+t64*(2*coefficients[2]+3*coefficients[3]*t64)
            crossing = result.valid & (t64 != 0) & (derivative != 0)
            solved = result.status == RootStatus.OK
            t = t64.astype(point.dtype)
        points = point+t[:, None]*direction
        directions = jnp.broadcast_to(direction, points.shape)
        points, normals = self._refine(points, directions)
        # Repeated roots are tangencies, not crossings. Do not count the known
        # origin or add intersections with the integration cutoff box.
        valid = crossing
        valid &= jnp.all((points > self.lower) & (points < self.upper), -1)
        residual = jnp.abs(self._values(points))
        finite = jnp.all(jnp.isfinite(points) & jnp.isfinite(normals), -1)
        complete = solved & jnp.all(~valid | (finite & (residual < self.sampler.epsilon)))
        return BoundaryHits(points[None], -normals[None], valid[None],
                            jnp.ones((1, 3), jnp.int32), complete[None])

    def next_boundary(self, point, normal, key):
        kd, ki = jax.random.split(key)
        direction = jax.random.normal(kd, (3,), dtype=point.dtype)
        direction /= jnp.linalg.norm(direction)
        hits = self._contact_line_hits(point, direction)
        p, n, count, valid = self._choose(hits, ki)
        sign = jnp.where(jnp.dot(n, p-point) > 0, 1., -1.)
        return (jnp.where(count > 0, p, point), jnp.where(count > 0, n, normal),
                sign*count, valid)

    def sample_pseudo_boundary(self, key, dtype):
        return self.box.sample_area_boundary(key, dtype)
