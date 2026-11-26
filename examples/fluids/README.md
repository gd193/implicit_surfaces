# Fluid examples

Run these modules from the repository root in the project environment.

## Passive smoke (Dirichlet vector potential)

```sh
python -m examples.fluids.smoke --grid 64 --steps 240 --walks 128 --source-samples 1024 --max-steps 8 --max-active-states 262144 --advection rk4 --continuous-smoke --flow-scale 2 --name torus_smoke
python -m examples.fluids.smoke --render --name torus_smoke
```

This is the working exterior-torus smoke scene, with zero potential on both
obstacle and outer box. `dirichlet.py` owns the velocity estimator and paired
continuations; it uses existing empty-sphere/source callbacks and the general
WoS driver. `vorticity.py` owns evolution/rendering, `transport.py` advection
and stretching, and `grid.py` field interpolation.

Other Dirichlet/streamfunction examples:

- `python -m examples.fluids.vorticity torus` or `twisted-torus`: interior-domain
  vorticity flow. Both accept `--solve` and `--render`.
- `python -m examples.fluids.channel`: 2D jet past a star.
- `python -m examples.fluids.curved_channel`: 2D driven circle-wave channel.

Existing output directory names and saved-volume formats are unchanged.
Temporary validation/profiling programs belong outside the repository.
