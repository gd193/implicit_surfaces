"""Neural twisted torus (twist 10) with vector g = grad h (u = g): one volume solve, all views.

Checkpoint: assets/models/twisted_torus_twist10_softplus_64k.npz (Softplus
MLP 3 x 32, 64k training steps, surface residual MAE 0.0053).

The domain is the zero level set of a Softplus MLP; every walk step uses a
certified empty sphere from Taylor autobounds of the network (about 70 s on
the GPU for 97^3 at 128 walks).

  python twisted_torus.py [--solve | --render]

Outputs in assets/renders/wos/twisted_torus/: volume.npz, slices.png (z = 0,
y = 0 with exact error), bc.png, and a still plus 20 s orbit for: bc, cut,
streamlines, arrows, cut_arrows,
all with a cut sweeping along the cut normal.
"""
from shapes import twisted_torus
from vector_volume import run

surface, params, g = twisted_torus("assets/models/twisted_torus_twist10_softplus_64k.npz")
run("twisted_torus", surface, params, g, (-1.,) * 3, (1.,) * 3, title="Neural twisted torus (twist 10)",
    slices=[(2, "z = 0  (x, y)"), (1, "y = 0  (x, z)")],
    cut_normal=(0., 0., 1.), cut_camera=(2.3, -2.7, 1.6),
    info=["Vector Laplace in a neural twisted torus, twist 10 (Softplus MLP)",
          "Dirichlet: u = g on the learned zero level set (arrows)",
          "g = grad h, h = x^3 - 3xy^2 + z^3 - 1.5 z (x^2 + y^2)"],
    n_walks=128, max_steps=2000)
