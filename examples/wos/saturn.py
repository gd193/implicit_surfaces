"""Saturn with vector g = grad h (harmonic, so u = g): one volume solve, all views.

  python saturn.py [--solve | --render]

Outputs in assets/renders/wos/saturn/: volume.npz, slices.png (x = 0, y = 0
with exact error), bc.png, and a still plus 20 s orbit for: bc, cut, streamlines, arrows,
cut_arrows,
all with a cut sweeping along the cut normal.
"""
from shapes import saturn
from vector_volume import run

surface, params, g = saturn()
run("saturn", surface, params, g, (-.95,) * 3, (.95,) * 3, title="Saturn",
    slices=[(0, "x = 0  (y, z): tilted ring"), (1, "y = 0  (x, z)")],
    cut_normal=(1., 0., 0.), cut_camera=(2.8, -1.8, 1.4),
    info=["Vector Laplace on Saturn (sphere + tilted ring, smooth union)",
          "Dirichlet: u = g on the surface (arrows)",
          "g = grad h, h = x^3 - 3xy^2 + z^3 - 1.5 z (x^2 + y^2)"],
    # In the blend web |f| underestimates distance, so walks take short
    # steps there; allow more steps than the default.
    n_walks=256, max_steps=5000)
