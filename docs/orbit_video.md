# Orbit video from an implicit surface

`scripts/render_orbit.py` accepts any 3D surface implementing `ray_step` and
`normal_at`, together with its parameters. It uses the shared ray marcher and
reuses the renderer across frames. Geometric termination additionally requires
`ray_step_with_grad`. Analytical, Lipschitz, and bounded neural surfaces all use
the same entry point; loading a model belongs in the supplied factory.

## CLI

Run from the repository root. Without a factory, the script renders an analytical
torus:

```sh
python scripts/render_orbit.py --device cpu --frames 60 --fps 20 \
  --width 256 --height 256 --out orbit.gif
```

To supply a surface, write a Python factory returning `(surface, params)`:

```python
# my_scene.py
import jax.numpy as jnp
from surfaces.implicit.lipschitz import sdfImplicit

def make_surface(radius=0.7):
    surface = sdfImplicit(
        fn=lambda p, x: jnp.linalg.norm(x) - p['radius'], dim=3)
    return surface, {'radius': radius}
```

```sh
python scripts/render_orbit.py --surface my_scene.py:make_surface \
  --surface-kwargs '{"radius":0.9}' --axis 0,1,0.3 \
  --target 0 0 0 --radius 3 --height-offset 0.5 \
  --start-angle 20 --sweep 360 --frames 120 --fps 30 \
  --shading metal --albedo 0.9 0.7 0.3 --samples 8 --out orbit.mp4
```

`--surface package.module:factory` also works for importable modules. The factory
can accept checkpoint paths via `--surface-kwargs` and return the loaded surface
and its parameter tree. Device selection happens before importing the factory.
The factory is ordinary trusted Python code, executed locally.

## Python

```python
from scripts.render_orbit import OrbitConfig, render_orbit

render_orbit(surface, params, 'orbit.gif', OrbitConfig(
    axis=(0., 1., 0.), target=(0., 0., 0.),
    radius=3., height_offset=0.5, fov=40.,
    frames=90, fps=30., width=512, height=512,
    samples=4,
))
```

Select the JAX device before initializing it when calling from Python. Surface
parameters remain fixed during the orbit. The camera up vector follows the
orbit axis, and the final sweep endpoint is excluded so a 360-degree loop does
not repeat its first frame.

## Controls and output

- Camera: `--axis x|y|z|X,Y,Z`, `--target X Y Z`, `--radius`, `--height-offset`,
  `--start-angle`, `--sweep`, `--fov`. Angles are degrees; negative sweeps reverse
  direction. Radius is the distance from the orbit axis, not from the target.
- Image/video: `--width`, `--height`, `--frames`, `--fps`, `--samples`, `--seed`,
  `--background R G B`, `--out`.
- Shading: `--shading normals|diffuse|metal|glass`, `--albedo R G B`, `--fuzz`,
  `--ior`, `--max-depth`, `--ray-offset`. Normals are direct display RGB;
  material radiance is averaged in linear space then square-root encoded.
- Marching: `--max-steps`, `--max-distance`, `--epsilon`, `--trust-region`,
  `--n-compactions`, `--termination value|geometric`. Value epsilon is in field
  units; geometric epsilon uses the local gradient distance estimate.
- Scene bounds: default `[-2,2]^3`; override with
  `--bounds XMIN YMIN ZMIN XMAX YMAX ZMAX`, or disable with `--no-bounds`.
  Moving the orbit target does not move the scene bounds.
- Device: `--device cpu|gpu|gpu:N`. Backend-specific setup, such as selecting
  JAX versus Warp bounds for a neural surface, remains the factory's job.

GIF requires Pillow and buffers frames in memory. Its frame duration is rounded
to the nearest 10 ms, with a minimum of 10 ms. MP4 streams frames to FFmpeg and
requires even width/height. Supply `--ffmpeg /path/to/ffmpeg` if it is not on PATH.
Both formats replace the destination only after successful encoding.

As with the renderer, exhausted ray budgets are displayed as misses. Choose
limits and bounds appropriate for the field; the video interface does not
certify that every intersection was resolved.

Large images use the renderer's shared bounded active buffer
(`--max-active-states 4096` by default). Finished rays are replaced with pending
rays while other rays continue. Marching, path state, and normal queries obey
the limit; there is no orbit-specific tiling. Input rays and output images still
scale with resolution. Set `XLA_PYTHON_CLIENT_PREALLOCATE=false` to avoid JAX
reserving most GPU memory at startup. MP4 streams frames to FFmpeg; GIF holds
finished frames in CPU memory.
