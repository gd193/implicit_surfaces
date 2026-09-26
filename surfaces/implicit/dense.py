"""denseImplicit — fixed-step ground-truth marcher.

Step size is a constant `gt_step` regardless of f-value. Termination via
SimpleIntegrator's `level_set_termination_criterion` (`|f - level_set| < eps`)
catches the surface when a sample lands close to it. For smooth fields with
`|∇f| · gt_step ≤ eps` on the trace, two consecutive samples cannot sandwich
the surface without one of them satisfying `|f| < eps`, so this terminates
on every crossing — the same coverage NIQ's `gt_dense_march` provides.

Use this as the comparison oracle: it goes through the same
`SimpleIntegrator` pipeline as `jnpboundedImplicit` and `niImplicitSurface`,
so shading, AA jitter and post-process are bit-identical.
"""
from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp

from .base import implicitSurface


class denseImplicit(implicitSurface):
    def __init__(self, gt_step: float = 1e-3, **kwargs):
        super().__init__(**kwargs)
        self.gt_step = float(gt_step)

    def empty_sphere(self, params: Any, x: jnp.ndarray, trust_region_size: jnp.ndarray,
                     upper_lower_flag: Optional[jnp.ndarray] = None
                     ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Empty-sphere isn't meaningful for fixed-step dense — return
        (fn_val, gt_step as radius, gt_step as next_tr). Returning gt_step lets
        WoSIntegrator make at least one gt_step-sized hop per iteration; the
        method is not really suited to WoS but stays contract-compliant."""
        fn_val = self(params, x) - self.level_set
        step = jnp.asarray(self.gt_step, dtype=fn_val.dtype)
        return fn_val, step, step

    def ray_step(self, params: Any, ray_origin: jnp.ndarray, ray_direction: jnp.ndarray,
                 trust_region_size: jnp.ndarray, upper_lower_flag: Optional[jnp.ndarray] = None
                 ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Always step `gt_step` along the ray; report f at the current origin
        for the integrator's level-set termination test. trust_region_size is
        ignored — dense doesn't model a trust region.
        """
        fn_val = self(params, ray_origin) - self.level_set
        step = jnp.asarray(self.gt_step, dtype=fn_val.dtype)
        return fn_val, step, jnp.asarray(False)
