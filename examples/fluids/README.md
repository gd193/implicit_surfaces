# Fluid examples

Run these modules from the repository root in the project environment.
Scene variants share a runner; numerical solvers remain in `sphereWalker`.

## Passive smoke (Dirichlet vector potential)

```sh
python -m examples.fluids.smoke --grid 64 --steps 240 --walks 128 --source-samples 1024 --max-steps 8 --max-active-states 262144 --advection rk4 --continuous-smoke --flow-scale 2 --name torus_smoke
python -m examples.fluids.smoke --render --name torus_smoke
```

This is the working exterior-torus smoke scene, with zero potential on both
obstacle and outer box. `vorticity.py` provides evolution/rendering;
`transport.py` provides advection and stretching.

## Other flows

- `python -m examples.fluids.vorticity torus` or `twisted-torus`: interior-domain
  vorticity flow. Both accept `--solve` and `--render`.
- `python -m examples.fluids.buoyancy sphere` or `torus`: exterior pressure-based
  buoyant smoke, with a pseudo-boundary outer box. Simulation is the default;
  `--render` reads saved data and `--resume` continues a checkpoint.
  Torus also supports `--twist`. Scene-specific sampling defaults are retained;
  use `<scene> --help` to inspect budgets before running.
- `python -m examples.fluids.pressure sphere` or `gyroid`: closed-domain pressure
  flow with all-Neumann boundaries; `--render` reads the saved volume.
- `python -m examples.fluids.channel`: 2D jet past a star.
- `python -m examples.fluids.curved_channel`: 2D driven circle-wave channel.

Existing output directory names and saved-volume formats are unchanged.
Temporary validation/profiling programs belong outside the repository.
