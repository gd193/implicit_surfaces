"""Orbit any 3D implicit surface, supplied directly or by a Python factory.

python scripts/render_orbit.py --out orbit.gif
python scripts/render_orbit.py --surface my_scene.py:make_surface --out orbit.mp4

Factories take JSON keyword arguments and return (surface, params). Surfaces
provide ray_step and normal_at, plus ray_step_with_grad for geometric stopping.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from contextlib import contextmanager
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import shutil
import subprocess
import tempfile
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@dataclass(frozen=True)
class OrbitConfig:
    width: int = 256
    height: int = 256
    frames: int = 60
    fps: float = 20.
    radius: float = 2.5
    height_offset: float = 0.8
    target: tuple[float, float, float] = (0., 0., 0.)
    axis: tuple[float, float, float] = (0., 1., 0.)
    start_angle: float = 0.
    sweep: float = 360.
    fov: float = 40.
    samples: int = 1
    max_active_states: int = 4096
    seed: int = 0
    background: tuple[float, float, float] = (0.5, 0.7, 1.)
    max_steps: int = 256
    max_distance: float = 20.
    epsilon: float = 1e-4
    trust_region: float = 0.5
    n_compactions: int = 4
    bounds: Any = ((-2., -2., -2.), (2., 2., 2.))
    termination: str = 'value'
    shading: str = 'normals'
    albedo: tuple[float, float, float] = (0.8, 0.6, 0.3)
    fuzz: float = 0.05
    ior: float = 1.5
    max_depth: int = 8
    ray_offset: float = 1e-3
    ffmpeg: str = 'ffmpeg'

    def validate(self) -> None:
        for name in ('width', 'height', 'frames', 'samples', 'max_active_states', 'max_steps', 'max_depth'):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        for name in ('fps', 'radius', 'max_distance', 'epsilon', 'trust_region', 'ior', 'ray_offset'):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        for name in ('axis', 'target', 'background', 'albedo'):
            values = getattr(self, name)
            if len(values) != 3 or not all(math.isfinite(v) for v in values):
                raise ValueError(f'{name} must have three finite components')
        if not 0 < math.hypot(*self.axis) < math.inf:
            raise ValueError('axis must be nonzero and finite')
        if not all(math.isfinite(v) for v in (self.height_offset, self.start_angle, self.sweep)):
            raise ValueError('orbit offsets and angles must be finite')
        if not 0 < self.fov < 180 or not 0 <= self.fuzz <= 1 or self.n_compactions < 0:
            raise ValueError('require 0 < fov < 180, 0 <= fuzz <= 1, and n_compactions >= 0')
        if self.termination not in ('value', 'geometric') or self.shading not in ('normals', 'diffuse', 'metal', 'glass'):
            raise ValueError('unsupported termination or shading mode')


def render_orbit(surface: Any, params: Any, output: str | Path,
                 config: OrbitConfig | None = None, *, progress: bool = True) -> Path:
    """Render a static surface with explicit parameters; reuse kernels per frame.

    Angles are degrees. The sweep endpoint is excluded to avoid duplicate frames
    for a full revolution. Orbit-axis up keeps the camera roll stable. Normals
    use direct RGB; material radiance is square-root encoded for display.
    max_active_states bounds traversal and shading work buffers.
    MP4 requires FFmpeg and even image dimensions. GIF timing is rounded
    by the format to 10 ms. Configure JAX's device before calling this function.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np
    from rayMarcher.camera import Camera
    from rayMarcher.marcher import SimpleIntegrator
    from rayMarcher.integrator import MaterialIntegrator
    from rayMarcher.materials import Material
    from utils.helpers import level_set_termination_criterion, geometric_termination_criterion

    config = config or OrbitConfig()
    config.validate()
    output = Path(output).expanduser().resolve()
    if output.suffix.lower() not in ('.gif', '.mp4'):
        raise ValueError('output must end in .gif or .mp4')
    if output.suffix.lower() == '.mp4' and (config.width % 2 or config.height % 2):
        raise ValueError('MP4 width and height must be even')
    if getattr(surface, 'dim', 3) != 3:
        raise ValueError('orbit rendering requires a 3D surface')
    axis = np.asarray(config.axis, dtype=float) / math.hypot(*config.axis)
    reference = np.eye(3)[np.argmin(np.abs(axis))]
    e1 = reference - np.dot(reference, axis) * axis
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    target = np.asarray(config.target)

    def camera(frame: int) -> Camera:
        angle = math.radians(config.start_angle + config.sweep * frame / config.frames)
        eye = target + config.radius * (math.cos(angle)*e1 + math.sin(angle)*e2) + config.height_offset*axis
        return Camera.setup_Camera(config.width, config.height, config.fov,
                                   jnp.asarray(eye), jnp.asarray(target), jnp.asarray(axis))

    criterion = geometric_termination_criterion if config.termination == 'geometric' else level_set_termination_criterion
    options = dict(max_ray_march_steps=config.max_steps, max_ray_distance=config.max_distance,
                   epsilon=config.epsilon, n_compactions=config.n_compactions,
                   max_active_states=config.max_active_states,
                   scene_bounds=config.bounds, termination_crit_fn=criterion)
    normals = config.shading == 'normals'
    if normals:
        renderer = SimpleIntegrator.from_surface(camera(0), jnp.asarray(config.background), surface, **options)
        render_params = params
    else:
        material = Material(jnp.array([('diffuse', 'metal', 'glass').index(config.shading)]),
                            jnp.array([config.albedo]), jnp.array([config.fuzz]), jnp.array([config.ior]))
        renderer = MaterialIntegrator(camera(0), material, jnp.asarray(config.background), [surface],
                                      max_depth=config.max_depth, ray_offset=config.ray_offset, **options)
        render_params = [params]
    output.parent.mkdir(parents=True, exist_ok=True)
    # Publish only a completed video; preserve an existing output if rendering fails.
    fd, temporary = tempfile.mkstemp(prefix='.orbit-', suffix=output.suffix.lower(), dir=output.parent)
    os.close(fd)
    try:
        with video_writer(Path(temporary), config) as append_frame:
            for frame in range(config.frames):
                renderer.camera = camera(frame)
                accumulated = np.zeros((config.height, config.width, 3), dtype=np.float32)
                for sample in range(config.samples):
                    key = jax.random.fold_in(jax.random.PRNGKey(config.seed), frame*config.samples + sample)
                    result = renderer.render(render_params, key, config.trust_region)
                    accumulated += np.asarray(result[0] if normals else result)
                    del result
                image = accumulated / config.samples
                if not np.all(np.isfinite(image)):
                    raise RuntimeError(f'nonfinite pixels in frame {frame}')
                image = np.clip(image, 0., 1.)
                if not normals:
                    image = np.sqrt(image)
                append_frame(np.rint(image*255).astype(np.uint8))
                if progress:
                    print(f'Frame {frame+1}/{config.frames}', flush=True)
        os.replace(temporary, output)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return output


@contextmanager
def video_writer(path: Path, config: OrbitConfig):
    """Pillow buffers GIF frames; FFmpeg streams MP4 frames through stdin."""
    if path.suffix == '.gif':
        from PIL import Image
        frames = []
        try:
            yield lambda pixels: frames.append(Image.fromarray(pixels))
            frames[0].save(path, save_all=True, append_images=frames[1:],
                           duration=max(10, round(1000./config.fps/10)*10), loop=0, disposal=2)
        finally:
            for frame in frames:
                frame.close()
        return
    executable = shutil.which(config.ffmpeg)
    if executable is None:
        raise ValueError('FFmpeg is required for MP4; pass --ffmpeg /path/to/ffmpeg or write a GIF')
    command = [executable, '-hide_banner', '-loglevel', 'error', '-y',
               '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{config.width}x{config.height}',
               '-r', str(config.fps), '-i', 'pipe:0', '-an', '-c:v', 'libx264',
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path)]
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errors)
        try:
            yield lambda pixels: process.stdin.write(pixels.tobytes())
            process.stdin.close()
            if process.wait() != 0:
                errors.seek(0)
                raise RuntimeError('FFmpeg failed: '+errors.read().decode(errors='replace'))
        except BrokenPipeError as exc:
            process.wait()
            errors.seek(0)
            raise RuntimeError('FFmpeg failed: '+errors.read().decode(errors='replace')) from exc
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            if not process.stdin.closed:
                process.stdin.close()


def load_surface(spec: str, kwargs: dict) -> tuple[Any, Any]:
    """Load a trusted local Python factory, after device selection."""
    source, separator, name = spec.rpartition(':')
    if not separator or not source or not name:
        raise ValueError('--surface must be module:factory or path.py:factory')
    if source.endswith('.py'):
        path = Path(source).expanduser().resolve()
        module_spec = importlib.util.spec_from_file_location('_orbit_surface', path)
        if module_spec is None or module_spec.loader is None:
            raise ValueError(f'cannot load {path}')
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_spec.name] = module
        sys.path.insert(0, str(path.parent))
        try:
            module_spec.loader.exec_module(module)
            return getattr(module, name)(**kwargs)
        finally:
            sys.path.pop(0)
    return getattr(importlib.import_module(source), name)(**kwargs)


def demo_surface() -> tuple[Any, Any]:
    """An analytical upright torus for a quick standalone run."""
    import jax.numpy as jnp
    from surfaces.implicit.lipschitz import sdfImplicit
    def field(p, x):
        return jnp.linalg.norm(jnp.array([jnp.linalg.norm(x[:2])-p['major'], x[2]]))-p['minor']
    return sdfImplicit(fn=field, dim=3), {'major': 0.55, 'minor': 0.2}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--surface', help='module:factory or path.py:factory; default is an analytical torus')
    parser.add_argument('--surface-kwargs', default='{}', help='JSON keyword arguments, e.g. a checkpoint path')
    parser.add_argument('--device', default='cpu', help='cpu, gpu, or gpu:N; selected before loading the factory')
    parser.add_argument('--ffmpeg', default='ffmpeg', help='FFmpeg executable for MP4 encoding')
    parser.add_argument('--out', type=Path, default=Path('orbit.gif'))
    for name in ('width', 'height', 'frames', 'samples', 'max_active_states', 'seed', 'max_steps', 'n_compactions', 'max_depth'):
        parser.add_argument('--'+name.replace('_', '-'), type=int, default=getattr(OrbitConfig(), name))
    for name in ('fps', 'radius', 'height_offset', 'start_angle', 'sweep', 'fov', 'max_distance',
                 'epsilon', 'trust_region', 'fuzz', 'ior', 'ray_offset'):
        parser.add_argument('--'+name.replace('_', '-'), type=float, default=getattr(OrbitConfig(), name))
    for name in ('target', 'background', 'albedo'):
        parser.add_argument('--'+name, type=float, nargs=3, default=getattr(OrbitConfig(), name))
    parser.add_argument('--axis', default='y', help='x, y, z, or comma-separated components')
    bounds = parser.add_mutually_exclusive_group()
    bounds.add_argument('--bounds', nargs=6, type=float, metavar=('XMIN','YMIN','ZMIN','XMAX','YMAX','ZMAX'))
    bounds.add_argument('--no-bounds', action='store_true')
    parser.add_argument('--termination', choices=('value', 'geometric'), default='value')
    parser.add_argument('--shading', choices=('normals', 'diffuse', 'metal', 'glass'), default='normals')
    args = parser.parse_args()
    try:
        if args.device == 'cpu':
            os.environ['JAX_PLATFORMS'] = 'cpu'
        elif args.device == 'gpu' or (args.device.startswith('gpu:') and args.device[4:].isdigit()):
            os.environ['JAX_PLATFORMS'] = 'cuda'
            if ':' in args.device:
                os.environ['CUDA_VISIBLE_DEVICES'] = args.device.split(':')[1]
        else:
            raise ValueError('--device must be cpu, gpu, or gpu:N')
        axis = {'x': (1.,0.,0.), 'y': (0.,1.,0.), 'z': (0.,0.,1.)}.get(args.axis)
        if axis is None:
            axis = tuple(float(v) for v in args.axis.split(','))
        fields = {name: getattr(args, name) for name in OrbitConfig.__dataclass_fields__ if name not in ('axis', 'bounds')}
        config = OrbitConfig(**fields, axis=axis, bounds=None if args.no_bounds else (
            (args.bounds[:3], args.bounds[3:]) if args.bounds else OrbitConfig().bounds))
        config.validate()
        kwargs = json.loads(args.surface_kwargs)
        if not isinstance(kwargs, dict) or (kwargs and not args.surface):
            raise ValueError('--surface-kwargs must be a JSON object supplied with --surface')
        surface, params = load_surface(args.surface, kwargs) if args.surface else demo_surface()
        print(render_orbit(surface, params, args.out, config))
    except (ValueError, ImportError, AttributeError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
