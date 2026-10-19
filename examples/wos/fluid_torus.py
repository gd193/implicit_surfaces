"""3D Monte Carlo fluid: a jet circulating around a torus breaks up.

  python fluid_torus.py [--solve | --render]

Torus R = 0.6, r = 0.3 (exact SDF). Wall psi = -c grad theta, theta =
atan2(z, rho - R) the angle around the tube: the wall stays impermeable and
the flux 2 pi c goes around the loop (fluid3d module doc). The flow starts as
a jet core along the tube centerline (torus_jet_vorticity); its shear layer
is unstable, rolls up and breaks apart while the flux carries it around.
Loop and outputs: fluid3d.py, in assets/renders/wos/fluid_torus/.
"""
from functools import partial

import jax.numpy as jnp
import numpy as np

from fluid3d import run3d, torus_jet_vorticity
from shapes import torus

MAJOR, MINOR, JET, WIDTH, BACKGROUND = .6, .3, 1.2, .08, .3
# Flux through the tube: background speed over the disk plus the jet core.
FLUX = BACKGROUND * np.pi * MINOR ** 2 + 2 * np.pi * JET * WIDTH ** 2 * np.log(2.)
C = FLUX / (2 * np.pi)


def wall_psi(x, floor=0.):
    """-c grad theta; on the tube e_theta = e_phi x e_d = -grad theta / |grad theta|.
    floor > 0 bounds it near the centerline (where grad theta ~ 1/d): the control variate."""
    rho = jnp.hypot(x[0], x[1])
    d2 = jnp.maximum((rho - MAJOR) ** 2 + x[2] ** 2, floor ** 2)
    grad_theta = jnp.stack([-x[2] * x[0] / rho, -x[2] * x[1] / rho, rho - MAJOR]) / d2
    return -C * grad_theta


surface, params, _ = torus()
run3d("fluid_torus", surface, params, (-.95, -.95, -.35), (.95, .95, .35), 96,
      partial(torus_jet_vorticity, major=MAJOR, jet=JET, width=WIDTH),
      title="Jet circulating in a torus", wall_psi=wall_psi,
      control_fn=partial(wall_psi, floor=.5 * MINOR), steps=400, n_walks=32,
      levels=(.35, .7), camera=(1.6, -1.9, 1.5))
