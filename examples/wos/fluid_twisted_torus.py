"""3D Monte Carlo fluid in the neural twisted torus (twist 10).

  python fluid_twisted_torus.py [--solve | --render]

The tube is a thin twisted ribbon (interior depth 0.10-0.15), too thin for
resolved vortex rings at an affordable grid, so the flow starts from a smooth
random divergence-free vorticity field (random_vorticity) masked to the tube.
Every walk step uses a certified empty sphere from Taylor autobounds of the
network (about 11k walks/s), so u is solved on every 2nd grid point (36^3) and
interpolated to the 72^3 vorticity grid: about 8 s per step. Loop and outputs: fluid3d.py, in assets/renders/wos/fluid_twisted_torus/.
"""
from functools import partial

from fluid3d import random_vorticity, run3d
from shapes import twisted_torus

surface, params, _ = twisted_torus("assets/models/twisted_torus_twist10_softplus_64k.npz")
run3d("fluid_twisted_torus", surface, params, (-.72,) * 3, (.72,) * 3, 72,
      partial(random_vorticity, wavelength=.35, peak=20.),
      title="Neural twisted torus (twist 10)", n_walks=16, steps=200, velocity_stride=2,
      camera=(1.6, -1.9, 1.3))
