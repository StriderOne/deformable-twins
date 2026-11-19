"""Reference PhysTwin stepping and visual binding for the tutorial."""
from pathlib import Path
import sys
import numpy as np
from scipy.spatial import cKDTree
import warp as wp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lab.simulation.phystwin import DEFAULT_EXPORT, DEFAULT_SOURCE, KERNEL_SHA256, load_graph, load_kernels


class PhysTwinSimulation:
    """Original force, velocity and ground kernels; double-buffered forward state."""
    def __init__(self, graph, source=DEFAULT_SOURCE, device='cuda:0'):
        self.g = graph
        self.device = wp.get_device(device)
        self.k = load_kernels(source)
        self.n = int(graph['n_object'])
        self.dt = float(graph['dt'])
        self.substeps = int(graph['substeps'])
        self.reverse = -float(graph['gravity'][2]) / 9.8
        self.time = 0.
        def array(value, dtype):
            return wp.array(value, dtype=dtype, device=self.device)
        self.q = array(graph['q'][:self.n], wp.vec3)
        self.v = array(graph['v'][:self.n], wp.vec3)
        self.q_next, self.v_next = wp.empty_like(self.q), wp.empty_like(self.v)
        self.force, self.v_force = wp.zeros_like(self.q), wp.empty_like(self.v)
        self.initial_controls = graph['q'][self.n:].copy()
        self.previous = array(self.initial_controls, wp.vec3)
        self.target = array(self.initial_controls, wp.vec3)
        self.control = wp.empty_like(self.target)
        self.control_v = wp.zeros_like(self.target)
        self.edges = array(graph['edges'], wp.vec2i)
        self.rest = array(graph['rest'], float)
        self.log_y = array(graph['log_y'], float)
        self.mass = array(graph['mass'][:self.n], float)
        self.restitution = array(graph['restitution'], float)
        self.friction = array(graph['friction'], float)
        self.captured = None
        if self.device.is_cuda:
            wp.load_module(module=self.k, device=self.device)
            with wp.ScopedCapture(device=self.device) as capture:
                self._frame()
            self.captured = capture.graph
        self.reset()

    def _frame(self):
        q, v, q_next, v_next = self.q, self.v, self.q_next, self.v_next
        for i in range(self.substeps):
            self.force.zero_()
            wp.launch(self.k.set_control_points, len(self.initial_controls),
                      [self.substeps, self.previous, self.target, i, self.control], device=self.device)
            wp.launch(self.k.eval_springs, len(self.g['edges']),
                      [q, v, self.control, self.control_v, self.n, self.edges, self.rest, self.log_y,
                       float(self.g['damping']), float(self.g['y_min']), float(self.g['y_max']), self.force], device=self.device)
            wp.launch(self.k.update_vel_from_force, self.n,
                      [v, self.force, self.mass, self.dt, float(self.g['drag']), self.reverse, self.v_force], device=self.device)
            wp.launch(self.k.integrate_ground_collision, self.n,
                      [q, self.v_force, self.restitution, self.friction, self.dt, self.reverse, q_next, v_next], device=self.device)
            q, q_next, v, v_next = q_next, q, v_next, v
        if q is not self.q:
            wp.copy(self.q, q)
            wp.copy(self.v, v)
        wp.copy(self.previous, self.target)

    def reset(self):
        self.q.assign(self.g['q'][:self.n])
        self.v.assign(self.g['v'][:self.n])
        self.previous.assign(self.initial_controls)
        self.target.assign(self.initial_controls)
        self.time = 0.

    def step(self, target):
        target = np.asarray(target, dtype=np.float32)
        if target.shape != self.initial_controls.shape or not np.isfinite(target).all():
            raise ValueError('Invalid controller targets')
        self.target.assign(target)
        if self.captured is not None:
            wp.capture_launch(self.captured)
        else:
            self._frame()
        q = self.q.numpy().copy()
        if not np.isfinite(q).all() or not np.isfinite(self.v.numpy()).all():
            raise FloatingPointError('Non-finite state; reset and reduce the controller displacement')
        self.time += self.dt * self.substeps
        return q


class GaussianBinding:
    """Fixed rest-pose inverse-distance binding to local rigid particle transforms.

    Uses the same local rigid-fit idea as PhysTwin, but recomputes from the rest
    pose rather than accumulating frame-to-frame transforms. Visualization only.
    """
    def __init__(self, particles, centers, neighbors=16):
        self.rest = np.asarray(particles, dtype=np.float32)
        self.centers = np.asarray(centers, dtype=np.float32)
        tree = cKDTree(self.rest)
        k = min(neighbors, len(self.rest))
        if k < 3:
            raise ValueError('At least three physical particles are required for skinning')
        _, self.adj = tree.query(self.rest, k=k)
        distance, self.ids = tree.query(self.centers, k=k)
        self.weights = 1. / (distance + 1e-6)
        self.weights /= self.weights.sum(axis=1, keepdims=True)
        self.offset = self.centers[:, None] - self.rest[self.ids]
        self.adj_rest = self.rest[self.adj] - self.rest[:, None]

    def deform(self, particles):
        particles = np.asarray(particles, dtype=np.float32)
        if particles.shape != self.rest.shape or not np.isfinite(particles).all():
            raise ValueError('Invalid particle state for skinning')
        adj_new = particles[self.adj] - particles[:, None]
        covariance = np.einsum('nki,nkj->nij', adj_new, self.adj_rest)
        u, _, vh = np.linalg.svd(covariance)
        sign = np.linalg.det(u @ vh)
        u[:, :, -1] *= np.where(sign < 0, -1, 1)[:, None]
        rotations = u @ vh
        moved = np.einsum('nkij,nkj->nki', rotations[self.ids], self.offset) + particles[self.ids]
        return np.einsum('nk,nki->ni', self.weights, moved).astype(np.float32)
