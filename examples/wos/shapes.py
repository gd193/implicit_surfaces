"""Shapes and boundary data shared by the 2D-slice and volume examples."""
import jax
import jax.numpy as jnp
import numpy as np

import common  # noqa: F401  (puts the repo root on sys.path)
from surfaces.implicit.lipschitz import lipschitzImplicit, sdfImplicit
from surfaces.implicit.primitives import smooth_union, sphere_sdf, torus_sdf


def torus():
    """Torus R = 0.6, r = 0.3 with g = cos(3 phi) around the ring."""
    params = {"center": jnp.zeros(3), "major": jnp.asarray(.6), "minor": jnp.asarray(.3)}
    g = lambda x: jnp.cos(3. * jnp.arctan2(x[1], x[0]))
    return sdfImplicit(fn=torus_sdf, dim=3), params, g


def saturn():
    """Sphere smoothly joined (blend 0.4) to a ring tilted 25 degrees about x.

    The smooth union is 1-Lipschitz, not an exact SDF. Data g = grad h with
    h = x^3 - 3xy^2 + z^3 - 1.5 z (x^2 + y^2) harmonic, so u = g exactly.
    """
    tilt = np.radians(25.)
    rot = jnp.array([[1., 0., 0.], [0., np.cos(tilt), -np.sin(tilt)], [0., np.sin(tilt), np.cos(tilt)]])
    params = {"planet": {"center": jnp.zeros(3), "radius": jnp.asarray(.42)},
              "ring": {"center": jnp.zeros(3), "major": jnp.asarray(.72), "minor": jnp.asarray(.14)}}
    fn = lambda p, x: smooth_union(sphere_sdf(p["planet"], x), torus_sdf(p["ring"], rot @ x), .4)
    h = lambda x: x[0] ** 3 - 3 * x[0] * x[1] ** 2 + x[2] ** 3 - 1.5 * x[2] * (x[0] ** 2 + x[1] ** 2)
    return lipschitzImplicit(lipschitz_bound=1., fn=fn, level_set=0., dim=3), params, jax.grad(h)


def twisted_torus(checkpoint="assets/models/twisted_torus_softplus.npz"):
    """Neural twisted torus: a Softplus MLP fit by scripts/train_twisted_torus.py.

    Empty spheres come from degree-2 Taylor autobounds of the network
    (jnpboundedImplicit), so walks are certified on the learned field itself.
    Data g = grad h as for Saturn: harmonic, so u = g inside any domain.
    """
    import sys
    sys.path.insert(0, str(common.ROOT / "scripts"))
    from train_twisted_torus import load_surface
    surface, params = load_surface(str(common.ROOT / checkpoint))
    h = lambda x: x[0] ** 3 - 3 * x[0] * x[1] ** 2 + x[2] ** 3 - 1.5 * x[2] * (x[0] ** 2 + x[1] ** 2)
    return surface, params, jax.grad(h)
