"""Monte Carlo fluid: flow snaking through a channel of staggered circle-wave walls.

  python fluid_circle_wave.py [--solve | --render]

The walls are the signed circle wave (shadertoy stGyzt) and its mirror image,
shifted by half a period: round bumps hanging into the channel alternate
between top and bottom, so the flow has to meander around every bump tip. The
no-slip source makes wall vorticity, so the flow separates behind each tip and
breaks up into vortices that fill the cavities. Inlet and outlet sit at a top
bump tip (y = 0.15) above a bottom cavity (y = -0.75).

psi: inflow.psi(y) on inlet and outlet, its end values on the top and bottom
wall (each wall is a streamline).
Loop and outputs: fluid.py, in assets/renders/wos/fluid_circle_wave/.
"""
import jax.numpy as jnp
import numpy as np

from fluid import BoundaryLayer, run
from surfaces.implicit.primitives import bbox_sdf, circle_wave2d_sdf

RADIUS, THICKNESS, TOP = .2, .8, .55
OPEN = np.pi * 5. / 6. * THICKNESS
PERIOD = 4. * RADIUS * np.sin(OPEN)
TIP = TOP + 2. * RADIUS * np.cos(OPEN) - RADIUS  # lowest point of a top bump
INLET = (-(TOP + RADIUS), TIP)                   # bottom cavity to top bump tip
LO, HI = (0., -.8), (6 * PERIOD, .8)
inflow = BoundaryLayer(U=1., delta=.08, halfwidth=(INLET[1] - INLET[0]) / 2,
                       center=(INLET[1] + INLET[0]) / 2)
geometry = {"box": {"min_corner": jnp.array([LO[0], -2.]), "max_corner": jnp.array([HI[0], 2.])},
            "wave": {"center": jnp.array([0., TOP]), "thickness": jnp.asarray(THICKNESS),
                     "radius": jnp.asarray(RADIUS)}}


def top_wall(params, x):
    return circle_wave2d_sdf(params["wave"], x)


def bottom_wall(params, x):
    """The top wave mirrored and shifted by half a period."""
    return circle_wave2d_sdf(params["wave"], jnp.stack([x[0] - PERIOD / 2, -x[1]]))


def walls(params, x):
    return jnp.maximum(top_wall(params, x), bottom_wall(params, x))


def domain(params, x):
    """The channel between the walls, cut to the x range (inlet and outlet)."""
    return jnp.maximum(bbox_sdf(params["box"], x), walls(params, x))


def psi_boundary(x):
    top, bottom = jnp.abs(top_wall(geometry, x)), jnp.abs(bottom_wall(geometry, x))
    ends = jnp.abs(bbox_sdf(geometry["box"], x))
    wall = jnp.where(top < bottom, inflow.psi(INLET[1]), inflow.psi(INLET[0]))
    return jnp.where(ends < jnp.minimum(top, bottom), inflow.psi(x[1]), wall)


run("fluid_circle_wave", domain, geometry, psi_boundary, inflow, LO, HI, nx=346, ny=134,
    no_slip=True, wall_fn=walls, title="flow through staggered circle-wave walls",
    info=["walls: circle wave (shadertoy stGyzt), bottom mirrored + half period",
          "Dirichlet psi = inflow psi(y) at the ends, constant on each wall",
          "no-slip: wall band omega -= alpha u_t / band each step"])
