"""Load pinned PhysTwin definitions without its renderer; keep upstream untouched."""
import ast
import hashlib
import importlib.util
import logging
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
import open3d as o3d
import torch
import warp as wp
import yaml

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'third_party/PhysTwin'
COMMIT = '81c718790a37e5e0102eb77af2c6edd34a9db25f'
_STREAM = None


def setup_cuda() -> None:
    global _STREAM
    wp.init()
    wp.set_device('cuda:0')
    if _STREAM is None:
        _STREAM = wp.Stream('cuda:0')
    torch.cuda.synchronize()
    wp.set_stream(_STREAM)
    torch.cuda.set_stream(wp.stream_to_torch(_STREAM))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def source_class(path: Path, name: str, method: str | None = None):
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.ClassDef) and n.name == name)
    if method:
        node = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == method)
    node.decorator_list = []
    return compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec')


def load_upstream():
    paths = ['qqtt/utils/config.py', 'qqtt/engine/trainer_warp.py',
             'qqtt/model/diff_simulator/spring_mass_warp.py']
    actual = subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != COMMIT:
        raise ValueError('Unsupported PhysTwin commit; review adapter before upgrading')
    for rel in paths:
        original = subprocess.check_output(['git', '-C', str(SOURCE), 'show', f'HEAD:{rel}'])
        if original != (SOURCE / rel).read_bytes():
            raise ValueError(f'Modified upstream source: {rel}')
    config_ns = {'yaml': yaml}
    exec(source_class(SOURCE / paths[0], 'Config'), config_ns)
    cfg = config_ns['Config']()
    cfg.use_graph = False  # Capture only after the replacement loss is initialized.
    cfg.collision_learn = False
    cfg.device = 'cuda:0'
    text = (SOURCE / paths[2]).read_text()
    lines = text.splitlines(keepends=True)
    definitions = []
    for node in ast.parse(text).body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
            definitions.append(''.join(lines[start:node.end_lineno]))
    # A real cached file lets Warp inspect original, unchanged kernel source.
    cache = Path.home() / '.cache/deformable-twins/phystwin-vi/source'
    cache.mkdir(parents=True, exist_ok=True)
    name = 'phystwin_vi_upstream_' + COMMIT[:12]
    path = cache / (name + '.py')
    path.write_text('import torch\nimport warp as wp\n\n' + '\n\n'.join(definitions))
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        mod.logger = logging.getLogger('phystwin')
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    mod = sys.modules[name]
    mod.cfg = cfg
    topology_ns = {'np': np, 'torch': torch, 'o3d': o3d, 'cfg': cfg}
    exec(source_class(SOURCE / paths[1], 'InvPhyTrainerWarp', '_init_start'), topology_ns)
    return cfg, mod, topology_ns['_init_start']


@wp.kernel
def tracking_nll(pred: wp.array(dtype=wp.vec3), gt: wp.array(dtype=wp.vec3),
                 visibility: wp.array(dtype=wp.int32), motion: wp.array(dtype=wp.int32),
                 inverse_variance: float, loss: wp.array(dtype=float)):
    i = wp.tid()
    if visibility[i] != 0 and motion[i] != 0:
        residual = pred[i] - gt[i]
        wp.atomic_add(loss, 0, 0.5 * inverse_variance * wp.dot(residual, residual))


def simulator_class(mod):
    class TrackingSimulator(mod.SpringMassSystemWarp):
        def calculate_loss(self):
            wp.launch(tracking_nll, self.num_original_points,
                      [self.wp_states[-1].wp_x, self.wp_current_object_points,
                       self.wp_current_object_visibilities, self.wp_current_object_motions_valid,
                       self.inverse_variance, self.loss])

        def clear_loss(self):
            self.loss.zero_()
    return TrackingSimulator


def validate_data(data: dict) -> None:
    points = np.asarray(data['object_points'])
    if points.ndim != 3 or points.shape[2] != 3 or points.shape[0] < 2:
        raise ValueError('Expected tracked points [frames, points, 3]')
    for name in ('object_visibilities', 'object_motions_valid'):
        a = np.asarray(data[name])
        if a.shape != points.shape[:2] or a.dtype != np.bool_:
            raise ValueError(f'Invalid boolean mask {name}')
    controls = np.asarray(data['controller_points'])
    if controls.ndim != 3 or controls.shape[0] != len(points) or controls.shape[2] != 3:
        raise ValueError('Controller trajectories must have matching frame count and XYZ coordinates')
    for name in ('object_points', 'controller_points', 'surface_points', 'interior_points'):
        a = np.asarray(data[name])
        if a.shape[-1] != 3 or not np.isfinite(a).all():
            raise ValueError(f'Invalid/non-finite coordinates: {name}')
    valid = data['object_visibilities'][1:] & data['object_motions_valid'][:-1]
    if not valid.any():
        raise ValueError('No valid observations')


def build_simulator(plan: dict):
    # Use a Warp-owned stream for CUDA capture; share it with Torch.
    setup_cuda()
    cfg, mod, topology = load_upstream()
    cfg.load_from_yaml(SOURCE / 'configs/real.yaml')
    with Path(plan['data']['optimal']).expanduser().open('rb') as f:
        cfg.set_optimal_params(pickle.load(f))  # Trusted local PhysTwin artifacts only.
    cfg.device, cfg.use_graph, cfg.collision_learn = 'cuda:0', False, False
    if cfg.self_collision:
        raise ValueError('Pilot supports only single-object scenes without self-collision')
    with Path(plan['data']['observations']).expanduser().open('rb') as f:
        data = pickle.load(f)
    validate_data(data)
    def tensor(a, dtype=torch.float32):
        return torch.as_tensor(np.asarray(a).copy(), dtype=dtype, device=cfg.device)
    points = tensor(data['object_points'])
    controls = tensor(data['controller_points'])
    structure = tensor(np.concatenate([data['object_points'][0], data['surface_points'], data['interior_points']]))
    vertices, edges, rest, mass, object_springs = topology(
        None, structure, controls[0], object_radius=cfg.object_radius,
        object_max_neighbours=cfg.object_max_neighbours, controller_radius=cfg.controller_radius,
        controller_max_neighbours=cfg.controller_max_neighbours)
    sim = simulator_class(mod)(
        vertices, edges, rest, mass, dt=cfg.dt, num_substeps=cfg.num_substeps,
        spring_Y=cfg.init_spring_Y, collide_elas=cfg.collide_elas, collide_fric=cfg.collide_fric,
        dashpot_damping=cfg.dashpot_damping, drag_damping=cfg.drag_damping,
        collide_object_elas=cfg.collide_object_elas, collide_object_fric=cfg.collide_object_fric,
        num_object_points=len(structure), num_surface_points=len(points[0]) + len(data['surface_points']),
        num_original_points=len(points[0]), controller_points=controls, reverse_z=cfg.reverse_z,
        spring_Y_min=cfg.spring_Y_min, spring_Y_max=cfg.spring_Y_max,
        gt_object_points=points, gt_object_visibilities=tensor(data['object_visibilities'], torch.bool),
        gt_object_motions_valid=tensor(data['object_motions_valid'], torch.bool), self_collision=False)
    sim.inverse_variance = 1.0 / plan['likelihood']['sigma_obs_m'] ** 2
    sim.object_springs = object_springs
    sim.valid_counts = (data['object_visibilities'][1:] & data['object_motions_valid'][:-1]).sum(axis=1)
    sim.resolved_config = cfg.to_dict()
    return Rollout(sim), data


class Rollout:
    def __init__(self, simulator):
        self.sim = simulator
        self.z_buffer = wp.to_torch(simulator.wp_spring_Y, requires_grad=False)
        self.training_graph = self.forward_graph = self.clear_graph = None

    def reset(self, z: torch.Tensor) -> None:
        if z.shape != self.z_buffer.shape or not torch.isfinite(z).all():
            raise ValueError('Invalid log-stiffness sample')
        with torch.no_grad():
            self.z_buffer.copy_(z.detach())
        self.sim.tape.zero()
        self.sim.clear_loss()
        self.sim.set_init_state(self.sim.wp_init_vertices, self.sim.wp_init_velocities)

    def capture(self) -> None:
        s = self.sim
        # Warm from the real initial state, not the zero-initialized work buffers.
        self.reset(self.z_buffer.clone())
        self.frame(1, backward=True, graph=False)
        s.tape.zero()
        s.tape = wp.Tape()
        self.reset(self.z_buffer.clone())
        with wp.ScopedCapture() as cap:
            with s.tape:
                s.step()
                s.calculate_loss()
            s.tape.backward(s.loss)
        self.training_graph = cap.graph
        with wp.ScopedCapture() as cap:
            s.step()
            s.calculate_loss()
        self.forward_graph = cap.graph
        with wp.ScopedCapture() as cap:
            s.tape.zero()
            s.clear_loss()
        self.clear_graph = cap.graph
        self.reset(self.z_buffer.clone())

    def frame(self, index: int, backward: bool, graph: bool = True):
        s = self.sim
        s.set_controller_target(index)
        if graph and self.training_graph is not None:
            wp.capture_launch(self.clear_graph)
            wp.capture_launch(self.training_graph if backward else self.forward_graph)
        else:
            s.tape.zero()
            s.clear_loss()
            s.tape = wp.Tape()
            if backward:
                with s.tape:
                    s.step()
                    s.calculate_loss()
                s.tape.backward(s.loss)
            else:
                s.step()
                s.calculate_loss()
        value = float(wp.to_torch(s.loss, requires_grad=False).item())
        if not np.isfinite(value):
            raise FloatingPointError(f'Non-finite loss at frame {index}')
        gradient = None
        if backward:
            gradient = wp.to_torch(s.wp_spring_Y.grad, requires_grad=False).detach().clone()
            if not torch.isfinite(gradient).all():
                raise FloatingPointError(f'Non-finite gradient at frame {index}')
        # shortcut: preserve upstream truncation; full trajectory ELBO requires a longer tape.
        s.set_init_state(s.wp_states[-1].wp_x, s.wp_states[-1].wp_v)
        return value, gradient

    def trajectory(self, z: torch.Tensor, end: int, backward: bool,
                   deadline: float = float('inf')):
        import time
        self.reset(z)
        losses = []
        gradient = torch.zeros_like(z)
        for frame in range(1, end):
            if time.monotonic() >= deadline:
                raise TimeoutError('Budget reached within rollout; incomplete iteration discarded')
            loss, g = self.frame(frame, backward)
            losses.append(loss)
            if g is not None:
                gradient.add_(g)
        return np.asarray(losses), gradient
