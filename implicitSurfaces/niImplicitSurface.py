"""niImplicitSurface — wraps neural-implicit-queries' classify_general_box for
adaptive trust-region empty-sphere queries used by sphereWalker.WoSIntegrator.

The third return of `empty_sphere` is interpreted by `WoSIntegrator.step` as the
*new* trust-region size for the next iteration (NOT a boolean flag — that is
the DifferentialWoS contract). When the σ-cube is classified definite-sign,
the inscribed σ-ball is empty: we return radius = σ and grow next σ. When
classification is UNKNOWN we return radius = 0 and shrink next σ — the walker
will not move that step but will retry at half the cube size.
"""
from typing import Optional, Tuple
import sys

import jax
import jax.numpy as jnp

from implicitSurfaces.implicit_surface import implicitSurface


_NIQ_SRC = "/system/user/volkmann/neural-implicit-queries/src"


def _import_niq():
    """Best-effort import of neural-implicit-queries' implicit_mlp_utils.

    Two collision points with implicit_surfaces_project:
      * top-level `utils` package (ISP) vs `utils.py` (niq).
      * niq's `implicit_function` is a top-level module in its src/.

    Strategy: if a `utils` module is already loaded that looks like ISP's
    `utils/`, save and shadow it temporarily; insert niq's src on sys.path,
    import implicit_mlp_utils + implicit_function, then restore the saved
    `utils` module. This must NOT permanently damage sys.modules.
    """
    saved_utils = sys.modules.pop("utils", None)
    saved_utils_subs = {k: sys.modules.pop(k) for k in list(sys.modules)
                        if k.startswith("utils.")}
    inserted = _NIQ_SRC not in sys.path
    if inserted:
        sys.path.insert(0, _NIQ_SRC)
    try:
        # import niq's `utils.py` first so its name binding is the niq one
        # while implicit_mlp_utils is being traced.
        import importlib
        if "utils" in sys.modules:
            del sys.modules["utils"]
        niq_utils = importlib.import_module("utils")
        import implicit_mlp_utils as _impl_mlp_utils
        import implicit_function as _impl_fn
        # Detach from sys.modules so they don't shadow ISP utils later.
        modules_to_pop = ["utils", "implicit_mlp_utils", "implicit_function",
                          "mlp", "sdf", "affine", "slope_interval",
                          "affine_layers", "slope_interval_layers",
                          "geometry"]
        for k in modules_to_pop:
            sys.modules.pop(k, None)
    finally:
        if inserted and _NIQ_SRC in sys.path:
            sys.path.remove(_NIQ_SRC)
        # restore ISP utils package so subsequent imports see ISP again
        if saved_utils is not None:
            sys.modules["utils"] = saved_utils
        for k, v in saved_utils_subs.items():
            sys.modules[k] = v
    return _impl_mlp_utils, _impl_fn


class niImplicitSurface(implicitSurface):
    """Empty-sphere via neural-implicit-queries' classify_general_box.

    Modes (mode kwarg): one of
        'interval' | 'affine_fixed' | 'affine_truncate' | 'affine_all'
        | 'slope_interval' | 'sdf'

    Adaptive trust-region pattern (compatible with WoSIntegrator.step,
    which feeds the third return back as the *next* trust-region size):
      * σ-cube classified -> radius = trust_region_size / 2; next σ = 1.5 * tr
      * σ-cube UNKNOWN     -> radius = 0; next σ = 0.5 * tr
    """

    def __init__(self,
                 mode: str,
                 npz_path: str,
                 level_set: float = 0.0,
                 dim: int = 3,
                 grow_factor: float = 1.5,
                 shrink_factor: float = 0.5,
                 max_trust_region_size: float = 2.0,
                 min_trust_region_size: float = 1e-6,
                 ray_safety_factor: float = 0.98,
                 ray_min_step: float = 1e-3,
                 **affine_kwargs):
        impl_mlp_utils, impl_fn = _import_niq()
        SIGN_UNKNOWN = impl_fn.SIGN_UNKNOWN

        impl, params = impl_mlp_utils.generate_implicit_from_file(
            npz_path, mode=mode, **affine_kwargs)
        self.mode = mode
        self.impl = impl
        self.params = params  # cache for callers
        self._SIGN_UNKNOWN = int(SIGN_UNKNOWN)
        self._grow = float(grow_factor)
        self._shrink = float(shrink_factor)
        self._max_tr = float(max_trust_region_size)
        self._min_tr = float(min_trust_region_size)
        self._ray_safety_factor = float(ray_safety_factor)
        self._ray_min_step = float(ray_min_step)
        self._classify_general_box = jax.jit(impl.classify_general_box)
        self._eval_fn = jax.jit(lambda p, x: impl(p, x))

        def fn(p, x):
            return self._eval_fn(p if p is not None else self.params, x)

        super().__init__(fn=fn, level_set=level_set, dim=dim)

    def classify_box(self, params, lower, upper):
        """Tight axis-aligned classification via NIQ's classify_general_box.

        NIQ's SIGN_* are 0 (unknown), 1 (positive), 2 (negative); we map to {-1, 0, +1}.
        """
        if params is None:
            params = self.params
        center = 0.5 * (lower + upper)
        half_extent = 0.5 * (upper - lower)
        box_vecs = jnp.diag(half_extent)
        niq_sign = self._classify_general_box(params, center, box_vecs)
        return jnp.where(niq_sign == 1, jnp.int8(1),
                jnp.where(niq_sign == 2, jnp.int8(-1), jnp.int8(0)))

    def empty_sphere(self,
                     params,
                     x: jnp.ndarray,
                     trust_region_size: jnp.ndarray,
                     upper_lower_flag: Optional[str] = None
                     ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        if params is None:
            params = self.params
        sigma = trust_region_size * 0.5  # half-width of the AABB cube
        eye = jnp.eye(self.dim, dtype=x.dtype)
        box_vecs = eye * sigma
        sign = self._classify_general_box(params, x, box_vecs)
        classified = sign != self._SIGN_UNKNOWN

        radius = jnp.where(classified, sigma, jnp.asarray(0.0, dtype=x.dtype))
        fn_val = self._eval_fn(params, x).squeeze() - self.level_set
        next_tr = jnp.where(classified,
                            trust_region_size * self._grow,
                            trust_region_size * self._shrink)
        next_tr = jnp.clip(next_tr, self._min_tr, self._max_tr)
        return fn_val, radius, next_tr

    def ray_step(self,
                 params,
                 ray_origin: jnp.ndarray,
                 ray_direction: jnp.ndarray,
                 trust_region_size: jnp.ndarray,
                 upper_lower_flag: Optional[str] = None
                 ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """One adaptive ray-march step using NIQ's affine/interval/slope_interval bound.

        Faithful port of `cast_rays_iter` from `~/neural-implicit-queries/src/queries.py`:
        classify the 1-D box of half-width `0.5 * trust_region_size` along the ray; if
        the box is decided (no level-set crossing), advance by `safety_factor * tr`;
        otherwise inch forward by `ray_min_step`. The integrator's tr-update handles
        grow/shrink based on the returned `(t_step, in_tr)` pair — exactly the semantics
        used by `jnpboundedImplicit.ray_step`, so `SimpleIntegrator` can drive either
        surface unchanged.

        Returns (fn_val, t_step, in_tr):
          fn_val: f(ray_origin) - level_set.
          t_step: distance to advance along ray_direction.
          in_tr:  True when the trust region was decided (= safe step taken); False when
                  UNKNOWN (= shrink tr next iteration). Convention matches
                  `update_trust_region_size` in `utils/helpers.py`.
        """
        if params is None:
            params = self.params

        # 1-D box along the ray: midpoint at half the trust-region distance, single
        # direction-vector spanning ±half_vec around that midpoint.
        half_vec = 0.5 * trust_region_size * ray_direction
        pos_mid = ray_origin + half_vec
        sign = self._classify_general_box(params, pos_mid, half_vec[None, :])
        in_tr = sign != self._SIGN_UNKNOWN

        fn_val = self._eval_fn(params, ray_origin).squeeze() - self.level_set

        safe_step = self._ray_safety_factor * trust_region_size
        unsafe_step = jnp.asarray(self._ray_min_step, dtype=trust_region_size.dtype)
        t_step = jnp.where(in_tr, safe_step, unsafe_step)

        return fn_val, t_step, in_tr
