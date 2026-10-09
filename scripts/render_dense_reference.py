"""Dense ground-truth render via `denseImplicit` + `SimpleIntegrator`.

This is now a thin wrapper that uses the standard surface/integrator stack —
identical shading, AA jitter and post-process to the bound-based methods.
denseImplicit's ray_step always returns gt_step regardless of f, so the
integrator marches at fixed resolution; termination via the level-set
criterion (`|f - level_set| < epsilon`) catches every crossing as long as
`|∇f|·gt_step ≤ epsilon` on the trace — true for the bunny softplus MLP at
gt_step = epsilon = 1e-3.

Outputs to assets/reference_renders/dense_gt/image.{npz,png}.
"""
import os, sys, time
sys.path.insert(0, '/system/user/volkmann/implicit_surfaces_project')

import numpy as np
import jax, jax.numpy as jnp
# Match jnpboundedImplicit / niImplicitSurface: force fp32 matmuls.
jax.config.update("jax_default_matmul_precision", "highest")

from implicitSurfaces.denseImplicit import denseImplicit
from rayMarcher.camera import Camera
from rayMarcher.marcher import SimpleIntegrator

print(f"JAX: {[d.platform for d in jax.devices()]}", flush=True)

# ---- config (must match bench_raycast_clean.py) ----
WIDTH = HEIGHT = 1024
CAM = dict(
    image_width=WIDTH, image_height=HEIGHT, vfov=30.0,
    lookfrom=jnp.array([2.5, 2.0, 2.5]),
    lookat=jnp.array([0.0, 0.0, 0.0]),
    vup=jnp.array([0.0, 1.0, 0.0]),
)
LEVEL_SET = 0.0
MAX_DIST = 10.0
EPSILON = 1e-3
GT_STEP = 1e-3

# Dense needs N_MAX_STEP ≈ max_dist / gt_step ≈ 10000. Leave headroom to ensure
# every miss-ray is conclusively classified rather than capped.
N_MAX_STEP = int(np.ceil(MAX_DIST / GT_STEP)) + 1   # 10001

EQX = '/system/user/volkmann/implicit_surfaces_project/assets/equinox_models/bunny_softplus.eqx'
OUT_DIR = '/system/user/volkmann/implicit_surfaces_project/assets/reference_renders/dense_gt'
OUT_NPZ = OUT_DIR + '/image.npz'
OUT_PNG = OUT_DIR + '/image.png'


# ---- MLP load (same as bench) ----
def _load_eqx_layers(path):
    arrs = []
    with open(path, 'rb') as f:
        while True:
            try: arrs.append(np.lib.format.read_array(f))
            except (EOFError, ValueError): break
    return arrs

W1, b1, W2, b2, W3, b3, Wo, bo = [
    jnp.asarray(a, dtype=jnp.float32) for a in _load_eqx_layers(EQX)
]
def mlp(x):
    h = jax.nn.softplus(W1 @ x + b1)
    h = jax.nn.softplus(W2 @ h + b2)
    h = jax.nn.softplus(W3 @ h + b3)
    return Wo @ h + bo
def fn_2arg(_p, x): return mlp(x)


surface = denseImplicit(fn=fn_2arg, level_set=LEVEL_SET, dim=3, gt_step=GT_STEP)

# n_compactions=1: with N_MAX_STEP=10001 most rays do hundreds-to-thousands of
# steps; a single mid-march compaction is fine. Trust-region update is irrelevant
# for fixed-step dense; its query adapter preserves the trust-region size.
integrator = SimpleIntegrator.from_surface(
    camera=Camera.setup_Camera(**CAM),
    background_color=jnp.array([0.5, 0.7, 1.0]),
    surface=surface,
    max_ray_march_steps=N_MAX_STEP,
    max_ray_distance=MAX_DIST,
    epsilon=EPSILON,
    n_compactions=20,
)

print(f"Marching {WIDTH*HEIGHT:,} rays via SimpleIntegrator + denseImplicit "
      f"| gt_step={GT_STEP} | n_max_step={N_MAX_STEP}", flush=True)
t0 = time.time()
img, niter = integrator.render(params=None, key=jax.random.PRNGKey(0),
                                init_trust_region_size=GT_STEP)
img.block_until_ready()
print(f"Render time: {time.time()-t0:.1f}s", flush=True)

niter_np = np.asarray(niter)
img_np = np.asarray(img)
# `niter < N_MAX_STEP` covers BOTH outcomes the integrator considers "done"
# without iter-cap: hits AND rays that overshot max_dist (sky background).
# Surface-hit-rate is harder to read off without output_states.hit_flag, but
# the image colour already reflects the right thing (sky vs normal-as-color).
print(f"Rays terminated within iter cap: {(niter_np < N_MAX_STEP).mean()*100:.2f}%", flush=True)

# ---- save ----
os.makedirs(OUT_DIR, exist_ok=True)
np.savez(OUT_NPZ, image=img_np, niter=niter_np, hit=hit,
         gt_step=GT_STEP, max_dist=MAX_DIST, level_set=LEVEL_SET,
         width=WIDTH, height=HEIGHT)
print(f"Saved npz → {OUT_NPZ}", flush=True)
try:
    from PIL import Image
    Image.fromarray((np.nan_to_num(img_np) * 255).clip(0, 255).astype(np.uint8)).save(OUT_PNG)
    print(f"Saved png → {OUT_PNG}", flush=True)
except ImportError:
    print("PIL not available; npz saved without PNG.", flush=True)
