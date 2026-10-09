# Twisted-torus Softplus checkpoints

All checkpoints use three width-32 hidden layers (2,273 parameters), scaled
Softplus with beta 10, seed 7, and the twisted field from
`scripts/train_twisted_torus.py`. Adjacent JSON files duplicate the embedded
metadata: training configuration, device, held-out errors, and inference checks.
Weights are stored as `w0,b0,...,w3,b3` and load without pickle.

| Checkpoint | Twist | Steps | Validation MSE | Mean residual on target surface |
| --- | ---: | ---: | ---: | ---: |
| `twisted_torus_softplus.npz` | 5 | 16,000 | 0.0001380 | 0.00623 |
| `twisted_torus_twist10_softplus.npz` | 10 | 32,000 | 0.0003355 | 0.00790 |
| `twisted_torus_twist10_softplus_64k.npz` | 10 | 64,000 | 0.0001393 | 0.00532 |

Residuals are field values, not world-space distance errors. The 32,000-step
checkpoint records the initial stronger-twist fit; the 64,000-step model is used
for its rendered orbit. All runs explicitly enabled Triton GEMM.

Reproduce a training run from the repository root, varying twist, steps, and
output filename according to the table:

```sh
XLA_FLAGS=--xla_gpu_enable_triton_gemm=true XLA_PYTHON_CLIENT_PREALLOCATE=false \
python scripts/train_twisted_torus.py --device gpu --twist 10 --steps 64000 \
  --out assets/models/twisted_torus_twist10_softplus_64k.npz
```

The examples in `assets/renders/` contain 36-frame normal-shaded orbits with two
samples per pixel: the twist-5 model at 128 and 1080 square, and the final
twist-10 model at 1080 square. Requested timing is 12 fps; GIF rounds each frame
to 80 ms. Camera radius is 2.5, height offset 0.6, and vertical FOV 40 degrees.

```sh
XLA_FLAGS=--xla_gpu_enable_triton_gemm=true XLA_PYTHON_CLIENT_PREALLOCATE=false \
WARP_CACHE_PATH=/tmp/warp-cache python scripts/render_orbit.py --device gpu \
  --surface scripts/train_twisted_torus.py:load_surface \
  --surface-kwargs '{"checkpoint":"assets/models/twisted_torus_twist10_softplus_64k.npz"}' \
  --width 1080 --height 1080 --frames 36 --fps 12 --samples 2 \
  --max-active-states 4096 --radius 2.5 --height-offset 0.6 \
  --bounds -1 -1 -1 1 1 1 --epsilon 0.0003 --termination geometric \
  --max-steps 256 --trust-region 0.25 \
  --out assets/renders/twisted_torus_twist10_orbit_1080.gif
```

The older twist-5 1080 GIF predates the shared active-buffer implementation and
used image tiles. Regenerating it with the current renderer changes the sample
sequence while preserving the model and camera settings.
