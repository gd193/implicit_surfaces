"""Monte Carlo fluid: 2D inviscid jet hitting a star in a channel.

  python fluid_2d.py [--solve | --render]

psi = jet.psi(y) on the channel (walls are streamlines) and 0 on the star.
Loop and outputs: fluid.py, in assets/renders/wos/fluid_2d/.
"""
import jax.numpy as jnp

from fluid import Jet, run
from surfaces.implicit.primitives import bbox_sdf, star2d_sdf, subtraction

LO, HI = (-1.5, -1.), (2.5, 1.)
jet = Jet(U=.3, J=1.2, delta=.07, halfwidth=1.)
geometry = {"box": {"min_corner": jnp.array(LO), "max_corner": jnp.array(HI)},
            "star": {"center": jnp.array([-.4, 0.]), "radius": jnp.asarray(.35),
                     "m": jnp.asarray(3.)}}


def domain(params, x):
    return subtraction(bbox_sdf(params["box"], x), star2d_sdf(params["star"], x))


def psi_boundary(x):
    """0 on the star, the inflow stream function on the channel walls."""
    on_star = jnp.abs(star2d_sdf(geometry["star"], x)) < jnp.abs(bbox_sdf(geometry["box"], x))
    return jnp.where(on_star, 0., jet.psi(x[1]))


run("fluid_2d", domain, geometry, psi_boundary, jet, LO, HI, nx=320, ny=160, title="jet past a star",
    info=["Dirichlet psi = U y + J d tanh(y/d) on the channel, 0 on the star"])
