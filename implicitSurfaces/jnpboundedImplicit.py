from typing import Optional, Tuple, cast

import jax
import jax.numpy as jnp
import jnp_bounds.autobounds as jb

from implicitSurfaces.implicit_surface import implicitSurface
from utils.jnpbounds_roots import roots_2nd_order_tensor
from utils.jnpbound_empty_sphere import empty_sphere_2nd_order_tensor
from utils.jnpbound_box_bounds import (
    box_grad_bounds as _poly_box_grad_bounds,
    box_hessian_bounds as _poly_box_hessian_bounds,
    box_value_bounds as _poly_box_value_bounds,
    polynomial_subbox_bounds as _poly_subbox_bounds,
)
from rayMarcher.utils import ray_aabb_intersection_inside


class jnpboundedImplicit(implicitSurface):
    """
    A class for implicit surfaces that can bounded by the jnp autobound library.
    Current limitations of autobound:
        - Only JAX-traceable functions can be automatically bounded.
        - Many JAX library functions are not yet supported. 
          What is supported is bounding MLPs or CNNs that uses the jax.nn.sigmoid, jax.nn.softplus, or jax.nn.swish activations.
        - To compute accurate bounds for deeper neural networks, you may need to use float64 rather than float32.
        - Tracing times can be very long because the library uses tons of explicit for loops --> jax.jit unrolls them,
          which can lead to long compilation times. (1-10minutes)
    For details see https://github.com/google/autobound/
    """
    def __init__(self,
                autobounds_max_degree: int,
                use_classify_subcubes: bool = True,
                use_warp: bool = True,
                **kwargs
                ):
        """
        Initialize the autobounded implicit surface.

        Args:
            fn (Callable): A jnp autobound-able JAX function that defines the implicit surface.
            params (Optional[dict]): Additional parameters for the jnp autobound library.
            level_set (float): The level set value for the implicit surface.
        """
        super().__init__(**kwargs)
        self.autobounds_max_degree = autobounds_max_degree
        self.use_classify_subcubes = use_classify_subcubes
        self.use_warp = use_warp
        # use_warp routes polynomial composition through fused NVIDIA-Warp kernels for the
        # supported activations (softplus / swish / sigmoid). GPU-only fast path; harmless
        # but unused on CPU. Default True since neural-surface workloads are the common case.
        self.taylor_bounds = jb.taylor_bounds(
            self.fn, max_degree=self.autobounds_max_degree, use_warp=self.use_warp,
        )


    def ray_step(self, params, ray_origin, ray_direction, trust_region_size, upper_lower_flag=None):

        # jb.taylor_bounds asserts trust_region_size is either 0-d or shape-matched to x0;
        # broadcast to ray_origin.shape so the per-axis form is always satisfied.
        trust_region_vec = jnp.broadcast_to(trust_region_size, ray_origin.shape)
        auto_bound = self.taylor_bounds(ray_origin, x_trust_region=None, trust_region_size=trust_region_vec)
        center_fn_val = auto_bound.polynomial_enclosure[0]
        taylor_enclosure = auto_bound.polynomial_enclosure

        if upper_lower_flag is None:
            upper_lower_flag = jnp.where(center_fn_val > self.level_set, jnp.array(-1), jnp.array(1)).squeeze() # -1 for lower, 1 for upper

        taylor_enclosure.real_coeffs = taylor_enclosure.real_coeffs.at[0].add(-self.level_set)
        hit, t_poly_intersection = roots_2nd_order_tensor(taylor_enclosure, ray_direction, upper_lower_flag, dim=ray_origin.shape[0])
        ### check if point is within bounds
        t_box_intersection = ray_aabb_intersection_inside(ray_origin, ray_direction, trust_region_size)
        in_box = jnp.logical_and(hit, t_poly_intersection < t_box_intersection)
        t = jnp.where(in_box, t_poly_intersection, t_box_intersection)

        # Networks that return shape (1,) (e.g., MLPs without an internal squeeze) propagate
        # a trailing axis through Taylor bounds. Squeeze to scalar so downstream state shapes
        # stay consistent with the marcher's per-ray scalar contract.
        center_fn_val = cast(jnp.ndarray, jnp.squeeze(center_fn_val))
        t = cast(jnp.ndarray, jnp.squeeze(t))
        in_box = jnp.squeeze(in_box)

        return center_fn_val, t, jnp.logical_not(in_box)
    

    def _box_polynomial(self, lower: jnp.ndarray, upper: jnp.ndarray):
        """One Taylor pass over the AABB; reused by classify_box / box_bounds / box_*_bounds."""
        center = 0.5 * (lower + upper)
        half_extent = 0.5 * (upper - lower)
        # taylor_bounds expects trust_region_size such that the box is [center - tr/2, center + tr/2],
        # so tr = 2 * half_extent (per-axis). half_extent returned for downstream use.
        auto_bound = self.taylor_bounds(center, x_trust_region=None,
                                        trust_region_size=2.0 * half_extent)
        return auto_bound.polynomial_enclosure, half_extent


    def classify_box(self, params, lower, upper):
        """Native AABB classification via the autobound polynomial enclosure.

        Tighter than the default empty-sphere fallback: the trust region is per-axis,
        matching the box exactly, instead of the circumscribed sphere.
        """
        poly, half_extent = self._box_polynomial(lower, upper)
        lo, hi = _poly_box_value_bounds(poly, half_extent)
        lo = lo - self.level_set
        hi = hi - self.level_set
        return jnp.where(hi < 0, jnp.int8(-1),
                jnp.where(lo > 0, jnp.int8(1), jnp.int8(0)))


    def box_bounds(self, params, lower, upper):
        poly, half_extent = self._box_polynomial(lower, upper)
        return _poly_box_value_bounds(poly, half_extent)


    def box_grad_bounds(self, params, lower, upper):
        """Per-axis (lo[d], hi[d]) bound on ∇f over the box. Requires autobounds_max_degree >= 2."""
        if self.autobounds_max_degree < 2:
            raise ValueError(
                f"box_grad_bounds requires autobounds_max_degree >= 2 "
                f"(got {self.autobounds_max_degree})"
            )
        poly, half_extent = self._box_polynomial(lower, upper)
        return _poly_box_grad_bounds(poly, dim=self.dim, half_extent=half_extent)


    def box_hessian_bounds(self, params, lower, upper):
        """(lo[d, d], hi[d, d]) bound on Hessian over the box. Requires autobounds_max_degree >= 3."""
        if self.autobounds_max_degree < 3:
            raise ValueError(
                f"box_hessian_bounds requires autobounds_max_degree >= 3 "
                f"(got {self.autobounds_max_degree})"
            )
        poly, half_extent = self._box_polynomial(lower, upper)
        return _poly_box_hessian_bounds(poly, dim=self.dim, half_extent=half_extent)


    def classify_subcubes(self, params, parent_lower, parent_upper, sub_lowers, sub_uppers):
        """One Taylor pass on the parent classifies all N sub-boxes via polynomial range.

        N MLP forward passes → 1. The parent polynomial is provably valid over the parent
        box and any subset thereof, so each sub-box's (lo, hi) is computed by polynomial
        interval evaluation, no extra autobound trace.
        """
        poly, _half_extent = self._box_polynomial(parent_lower, parent_upper)
        parent_center = 0.5 * (parent_lower + parent_upper)
        delta_lowers = sub_lowers - parent_center[None, :]
        delta_uppers = sub_uppers - parent_center[None, :]

        def classify_one(delta_lo, delta_hi):
            lo, hi = _poly_subbox_bounds(poly, delta_lo, delta_hi)
            lo = lo - self.level_set
            hi = hi - self.level_set
            return jnp.where(hi < 0, jnp.int8(-1),
                    jnp.where(lo > 0, jnp.int8(1), jnp.int8(0)))

        return jax.vmap(classify_one)(delta_lowers, delta_uppers)


    def empty_sphere(self, params: jnp.ndarray, x: jnp.ndarray, trust_region_size: jnp.ndarray, upper_lower_flag: Optional[str]=None) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Compute the largest empty sphere around the given position within the trust region.
        We assume global bounds from the autobound library, hence the trust_region_size is inf
        args:
            x (jnp.ndarray)(dim,): The current position of the walker
            trust_region_size (jnp.ndarray)(1,): The current trust region size
        Returns:
            fn_val (jnp.ndarray)(1,): The value of the function at the current position
            radius (jnp.ndarray)(1,): The radius of the largest empty sphere
            trust_region_empty_flag (jnp.ndarray)(1,): A boolean flag indicating whether the entire trust region is empty (True) or not (False)
        """
        auto_bound = self.taylor_bounds(x, x_trust_region=None, trust_region_size=trust_region_size)
        taylor_enclosure = auto_bound.polynomial_enclosure
        fn_val = taylor_enclosure[0].squeeze()

        if upper_lower_flag is None:
            upper_lower_flag = jnp.where(fn_val.squeeze() > self.level_set, jnp.array(-1), jnp.array(1)) # stay on same side of level set
        radius = empty_sphere_2nd_order_tensor(taylor_enclosure, upper_lower_flag, dim=x.shape[0], isolevel=self.level_set).squeeze()
        outside_trust_region = radius > trust_region_size * 0.5
        safe_radius = jnp.where(outside_trust_region, trust_region_size * 0.5, radius)
        return fn_val, safe_radius, outside_trust_region

