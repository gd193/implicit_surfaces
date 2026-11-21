# Triangle mesh surfaces with Sengi

`TriMeshSurface` uses a Sengi BVH for closest points, ray intersections, box
overlap and inside/outside classification. A surface takes the source
`TriMesh` and boundary data prepared from the same vertices and faces.
The prepared boundary carries its index data. Build the index once and prepare its boundary before creating the
surface. Queries return
face IDs in the original mesh order, so surface normals and per-vertex
velocity interpolation use the source mesh.

## Setup

Create the environment from the lab conda channel and conda-forge.

```sh
mamba env create -f environment-warp.yml
mamba activate implicit-surfaces-warp
```

Outside the lab, build Sengi 0.4.0 using its conda recipe and replace the lab
channel in `environment-warp.yml` with the resulting local channel URL.
[Sengi installation instructions](https://github.com/gd193/sengi#readme)

## Build a triangle mesh surface

```python
import jax
import jax.numpy as jnp

from sengi import bvh
from sengi.primitives.triangles import Triangles
from surfaces.base import TriMesh
from surfaces.explicit.trimesh import TriMeshSurface

mesh = TriMesh(
    vertices=jnp.asarray(vertices, dtype=jnp.float32),
    faces=jnp.asarray(faces, dtype=jnp.int32),
)
triangles = Triangles()
source = triangles.indexed(mesh.vertices, mesh.faces)
index = bvh.build(triangles, source)
boundary = bvh.prepare_boundary(index, source, silhouettes=False)
surface = TriMeshSurface(mesh, boundary)

point, distance, face = jax.jit(surface.closest_point)(None, points)
signed_distance = jax.jit(surface.signed_value)(None, points)
```

Signed distances use even-odd classification on closed meshes. Distances are
negative inside, positive outside and zero on the boundary. Use
`builder="lbvh"` for Morton-order construction. Point inputs can be shape
`(3,)` or `(..., 3)`.

Run the downstream integration tests on CPU with
`JAX_PLATFORMS=cpu python -m pytest tests/test_trimesh_surface.py`.
Sengi's API guide covers direct queries, boundary preparation and extension
points. [Sengi BVH guide](https://github.com/gd193/sengi/blob/main/docs/warp_bvh.md)
