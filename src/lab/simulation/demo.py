"""Infrastructure scene: a free PhysTwin object, floor, and a force-driven sphere."""
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess

import newton
import numpy as np
import warp as wp

from .internal_forces import SolverInternalForces, SolverArticulatedInternalForces
from .phystwin import KERNEL_SHA256, PhysTwinInternalForces


@dataclass(frozen=True)
class DemoSettings:
    object_mass: float = .2
    particle_radius: float = .003
    sphere_mass: float = .1
    sphere_radius: float = .025
    contact_ke: float = 1000.
    contact_kd: float = .1
    contact_kf: float = .1
    contact_mu: float = .6
    pull_stiffness: float = 20.
    pull_force_max: float = 5.
    gravity: float = -9.8
    lift: float = .02


@wp.kernel
def pull_sphere(body_q: wp.array[wp.transform], body_qd: wp.array[wp.spatial_vector],
                target: wp.array[wp.vec3], enabled: wp.array[wp.int32],
                stiffness: float, damping: float, maximum: float,
                forces: wp.array[wp.spatial_vector]):
    if enabled[0] != 0:
        position = wp.transform_get_translation(body_q[0])
        velocity = wp.spatial_top(body_qd[0])
        force = stiffness * (target[0] - position) - damping * velocity
        force *= wp.min(1.0, maximum / wp.max(wp.length(force), 1.0e-12))
        wp.atomic_add(forces, 0, wp.spatial_vector(force, wp.vec3(0.0)))


class NewtonDemo:
    def __init__(self, graph, device='cuda:0', settings=DemoSettings(), capture=True, robot=False, fast=False):
        self.settings = settings
        self.fast = fast
        self.drag = float(graph['drag'])
        if not all(np.isfinite(v) for v in asdict(settings).values()):
            raise ValueError('Demo settings must be finite')
        if min(settings.object_mass, settings.particle_radius, settings.sphere_mass, settings.sphere_radius) <= 0:
            raise ValueError('Masses and radii must be positive')
        if min(settings.contact_ke, settings.contact_kd, settings.contact_kf, settings.contact_mu,
               settings.pull_stiffness, settings.pull_force_max) < 0:
            raise ValueError('Contact and pull coefficients must be non-negative')
        if not np.array_equal(graph['gravity'], [0, 0, 9.8]):
            raise ValueError('This demo expects the exported PhysTwin +Z gravity frame')
        self.n = int(graph['n_object'])
        self.dt, self.substeps = float(graph['dt']), int(graph['substeps'])
        self.rotation = np.diag([1., -1., -1.]).astype(np.float32)
        self.rest = graph['q'][:self.n] @ self.rotation.T
        self.translation = np.array([0., 0., settings.lift + settings.particle_radius - self.rest[:, 2].min()], np.float32)
        self.rest += self.translation
        self.mass_scale = settings.object_mass / float(graph['mass'][:self.n].sum())
        builder = newton.ModelBuilder(gravity=(0., 0., settings.gravity))
        builder.add_particles(pos=self.rest, vel=graph['v'][:self.n] @ self.rotation.T,
                              mass=graph['mass'][:self.n] * self.mass_scale,
                              radius=np.full(self.n, settings.particle_radius))
        cfg = newton.ModelBuilder.ShapeConfig(ke=settings.contact_ke, kd=settings.contact_kd,
                                              kf=settings.contact_kf, mu=settings.contact_mu,
                                              density=0., mu_rolling=0., mu_torsional=0.)
        builder.add_ground_plane(cfg=cfg)
        low, high = self.rest.min(0), self.rest.max(0)
        self.sphere_initial = np.array([low[0] - settings.sphere_radius - .015,
                                        (low[1] + high[1])/2, settings.sphere_radius + .002], np.float32)
        self.robot = None
        if robot:
            from .panda import Panda
            self.robot = Panda(builder, self.rest, settings)
        else:
            inertia = .4 * settings.sphere_mass * settings.sphere_radius**2
            body = builder.add_body(xform=wp.transform(self.sphere_initial, wp.quat_identity()),
                                    mass=settings.sphere_mass, inertia=wp.mat33(np.eye(3)*inertia),
                                    lock_inertia=True, label='pusher')
            builder.add_shape_sphere(body, radius=settings.sphere_radius, cfg=cfg)
        self.model = builder.finalize(device=device)
        if self.robot:
            self.robot.initialize(self.model)
        self.model.particle_grid = None
        self.model.particle_max_velocity = 1.e6
        for name in ('ke', 'kd', 'kf', 'mu'):
            setattr(self.model, 'soft_contact_' + name, getattr(settings, 'contact_' + name))
        self.law = PhysTwinInternalForces(graph, self.model, mass_scale=self.mass_scale)
        solver_cls = SolverArticulatedInternalForces if self.robot else SolverInternalForces
        options = {} if self.robot else dict(enable_tri_contact=False)
        self.solver = solver_cls(self.model, [self.law], drag=float(graph['drag']),
                                 angular_damping=0., **options)
        self.pipeline = newton.CollisionPipeline(self.model, soft_contact_max=(4 if fast else (32 if self.robot else 4))*self.n, rigid_contact_max=256)
        self.contacts = self.pipeline.contacts()
        self.state = self.model.state()
        self.scratch = self.model.state()
        self.control = self.model.control()
        self.target = wp.array([self.sphere_initial], dtype=wp.vec3, device=self.model.device)
        self.pull_enabled = wp.array([1], dtype=wp.int32, device=self.model.device)
        self.multirate = None
        if fast:
            if not self.robot:
                raise ValueError('Fast coupling requires a robot')
            from .multirate import MultiRate
            self.multirate = MultiRate(self)
        self.captured = None
        self.time = 0.
        # Warm all Newton collision/force kernels before CUDA capture.
        self._frame() if fast else self._substep(self.state, self.scratch)
        self.reset()
        if capture and self.model.device.is_cuda:
            with wp.ScopedCapture(device=self.model.device) as scope:
                self._frame()
            self.captured = scope.graph
            self.reset()

    def _substep(self, state_in, state_out):
        state_in.clear_forces()
        if self.robot:
            self.robot.apply(state_in, self.control)
        else:
            wp.launch(pull_sphere, 1,
                      [state_in.body_q, state_in.body_qd, self.target, self.pull_enabled,
                       self.settings.pull_stiffness,
                       2*np.sqrt(self.settings.pull_stiffness*self.settings.sphere_mass),
                       self.settings.pull_force_max, state_in.body_f], device=self.model.device)
        self.pipeline.collide(state_in, self.contacts)
        self.solver.step(state_in, state_out, self.control, self.contacts, self.dt)

    def _frame(self):
        if self.multirate is not None:
            self.multirate.frame()
            return
        a, b = self.state, self.scratch
        for _ in range(self.substeps):
            self._substep(a, b)
            a, b = b, a
        if a is not self.state:
            for field in ('particle_q', 'particle_qd', 'body_q', 'body_qd', 'joint_q', 'joint_qd'):
                wp.copy(getattr(self.state, field), getattr(a, field))

    def reset(self):
        for state in (self.state, self.scratch):
            wp.copy(state.particle_q, self.model.particle_q)
            wp.copy(state.particle_qd, self.model.particle_qd)
            wp.copy(state.body_q, self.model.body_q)
            wp.copy(state.body_qd, self.model.body_qd)
            wp.copy(state.joint_q, self.model.joint_q)
            wp.copy(state.joint_qd, self.model.joint_qd)
            state.clear_forces()
        self.target.assign(np.array([self.sphere_initial]))
        if self.robot:
            self.robot.set_target(self.robot.initial)
        self.solver.reset_internal_models()
        self.time = 0.

    def step(self, target, pull=True):
        if self.robot:
            self.robot.set_target(target, self.dt*self.substeps)
        else:
            target = np.asarray(target, dtype=np.float32)
            if target.shape != (3,) or not np.isfinite(target).all():
                raise ValueError('Expected finite 3D sphere target')
            self.target.assign(target[None])
            self.pull_enabled.assign(np.array([int(pull)], np.int32))
        if self.captured is None:
            self._frame()
        else:
            wp.capture_launch(self.captured)
        for field in ('particle_q', 'particle_qd', 'body_q', 'body_qd', 'joint_q', 'joint_qd'):
            if not np.isfinite(getattr(self.state, field).numpy()).all():
                raise FloatingPointError(f'Non-finite Newton state: {field}')
        if self.multirate and self.multirate.peak.numpy()[0] > self.contacts.soft_contact_max:
            raise FloatingPointError('Particle contact buffer overflow')
        self.time += self.dt * self.substeps
        return self.state.particle_q.numpy().copy()

    @property
    def sphere_position(self):
        return self.state.body_q.numpy()[0, :3].copy()

    def write_manifest(self, provenance, ply_path, seed=0):
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        directory = Path.home()/'.cache/deformable-twins/newton-phystwin'/stamp
        directory.mkdir(parents=True, exist_ok=False)
        root = Path(__file__).resolve().parents[3]
        def git(*args):
            return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()
        manifest = dict(kind='infrastructure_demo', status='running', created_utc=stamp,
                        experiment_id=None, baseline_id=None, benchmark_id=None, protocol_id=None,
                        seed=seed, repository_commit=git('rev-parse','HEAD'), repository_dirty=git('status','--porcelain'),
                        external_commit=provenance['source_commit'], source_provenance=provenance,
                        kernels_sha256=KERNEL_SHA256, settings=asdict(self.settings),
                        mass_scale=self.mass_scale, dt=self.dt, substeps=self.substeps,
                        rotation=self.rotation.tolist(), translation=self.translation.tolist(),
                        object_particles=self.n, internal_springs=self.law.count,
                        device=str(self.model.device), hardware=self.model.device.name,
                        versions={p:importlib.metadata.version(p) for p in ['newton','warp-lang','numpy','viser']},
                        ply_path=str(ply_path), ply_sha256=hashlib.sha256(Path(ply_path).read_bytes()).hexdigest(),
                        source_hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                                       for p in [*Path(__file__).parent.glob('*.py'), root/'tutorials/view_phystwin_gaussians.py', root/'tutorials/view_phystwin_gl.py']},
                        notes=['Demonstration mass and contacts; not physical calibration.',
                               'Particle spheres; self-collision disabled; forward-only.'])
        manifest['coupling'] = dict(mode='multirate' if self.fast else 'reference',
                                    particle_dt=self.dt, rigid_stride=self.multirate.stride if self.fast else 1,
                                    mass_matrix_interval=self.multirate.rigid.update_mass_matrix_interval if self.fast else 1,
                                    soft_contact_max=self.contacts.soft_contact_max)
        if self.robot:
            manifest['robot'] = self.robot.metadata()
            manifest['robot']['mass_matrix_interval'] = manifest['coupling']['mass_matrix_interval']
        manifest['config_sha256'] = hashlib.sha256(json.dumps(dict(settings=manifest['settings'], robot=manifest.get('robot'), coupling=manifest['coupling']),sort_keys=True).encode()).hexdigest()
        (directory/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        return directory
