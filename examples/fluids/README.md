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

## Isolated pressure prototype

`pressure_prototype/` owns the experimental pressure estimators, boundary-walk
state, specialized boundary samplers, and their Warp kernels. It depends on the
general integration and surface-query machinery; those libraries do not depend
on this prototype. The main Dirichlet simulation does not import it.

Supported sampling domains are explicit: closed analytic box/sphere and exterior
analytic sphere/torus. Unsupported surfaces and sampler subclasses are rejected
before tracing. A generic implicit surface's containment, ray, or empty-sphere
queries alone do not establish the sampling PDFs and transition weights required
by this estimator. The private implicit-hit helper only serves the torus adapter.
There is no generic implicit/gyroid or twisted-torus pressure entry point.

```sh
python -m examples.fluids.pressure_prototype.closed --help
python -m examples.fluids.pressure_prototype.buoyancy sphere --help
python -m examples.fluids.pressure_prototype.buoyancy torus --help
```

Buoyancy uses an exterior obstacle and a pseudo-boundary outer box. Simulation
is the default; `--render` reads saved data and `--resume` continues a checkpoint.
Scene sampling defaults are retained; inspect budgets before running.

Existing output directory names and saved-volume formats are unchanged.
Temporary validation/profiling programs belong outside the repository.
