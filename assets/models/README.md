# Neural Implicit Checkpoints

`assets/models/` holds the two tracked neural implicit models:

1. **Stanford Bunny** (`bunny_softplus.eqx` and `bunny_softplus.npz`) — softplus MLP stored in both Equinox stream (`.eqx`, for `jnpboundedImplicit`) and NIQ archive (`.npz`, for `niImplicitSurface`) formats.
2. **Twisted Torus (twist 10)** (`twisted_torus_twist10_softplus_64k.npz` + `.json`) — three width-32 hidden layers (2,273 parameters), scaled Softplus with beta 10, seed 7, trained for 64,000 steps on the twisted field from `scripts/train_twisted_torus.py`. Weights are stored as `w0,b0,...,w3,b3` and load without pickle.

| Checkpoint | Twist | Steps | Validation MSE | Mean residual on target surface |
| --- | ---: | ---: | ---: | ---: |
| `twisted_torus_twist10_softplus_64k.npz` | 10 | 64,000 | 0.0001393 | 0.00532 |

Residuals are field values, not world-space distance errors. Training explicitly enabled Triton GEMM.

Reproduce the twisted-torus training run from the repository root:

```sh
XLA_FLAGS=--xla_gpu_enable_triton_gemm=true XLA_PYTHON_CLIENT_PREALLOCATE=false \
python scripts/train_twisted_torus.py --device gpu --twist 10 --steps 64000 \
  --out assets/models/twisted_torus_twist10_softplus_64k.npz
```

Render a 36-frame normal-shaded orbit (generated outputs in `outputs/` stay gitignored):

```sh
XLA_FLAGS=--xla_gpu_enable_triton_gemm=true XLA_PYTHON_CLIENT_PREALLOCATE=false \
WARP_CACHE_PATH=/tmp/warp-cache python scripts/render_orbit.py --device gpu \
  --surface scripts/train_twisted_torus.py:load_surface \
  --surface-kwargs '{"checkpoint":"assets/models/twisted_torus_twist10_softplus_64k.npz"}' \
  --width 1080 --height 1080 --frames 36 --fps 12 --samples 2 \
  --max-active-states 4096 --radius 2.5 --height-offset 0.6 \
  --bounds -1 -1 -1 1 1 1 --epsilon 0.0003 --termination geometric \
  --max-steps 256 --trust-region 0.25 \
  --out outputs/renders/twisted_torus_twist10_orbit_1080.gif
```
