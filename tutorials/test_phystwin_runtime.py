"""Infrastructure checks: forces, controller interpolation, contact and visual binding."""
import unittest
import numpy as np
import warp as wp
from phystwin_runtime import PhysTwinSimulation, GaussianBinding


def fixture(substeps=1):
    return dict(n_object=2, q=np.array([[0, 0, -.2], [.12, 0, -.2], [.25, 0, -.2]], np.float32),
                v=np.array([[.01, 0, 0], [-.02, 0, 0], [0, 0, 0]], np.float32),
                mass=np.array([.1, .2, 0], np.float32), edges=np.array([[0, 1], [1, 2]], np.int32),
                rest=np.array([.1, .1], np.float32), log_y=np.log(np.array([3, 4], np.float32)),
                damping=.15, y_min=1., y_max=100., drag=3., dt=.0001, substeps=substeps,
                gravity=np.array([0, 0, 9.8]), restitution=np.array([.5], np.float32),
                friction=np.array([.3], np.float32))


class PhysicsTests(unittest.TestCase):
    def test_forces_drag_controller_interpolation_and_buffer_parity(self):
        for steps in (1, 2, 3):
            g = fixture(steps)
            sim = PhysTwinSimulation(g, device='cpu')
            target = sim.initial_controls + [0, 0, -.01]
            q, v = g['q'][:2].copy().astype(float), g['v'][:2].copy().astype(float)
            for substep in range(steps):
                controls = sim.initial_controls + (target - sim.initial_controls) * ((substep + 1) / steps)
                all_q = np.concatenate([q, controls])
                all_v = np.concatenate([v, np.zeros_like(controls)])
                force = np.zeros_like(q)
                for e, (a, b) in enumerate(g['edges']):
                    delta = all_q[b] - all_q[a]
                    length = np.linalg.norm(delta)
                    direction = delta / length
                    spring = np.exp(g['log_y'][e]) * (length / g['rest'][e] - 1)
                    damping = g['damping'] * np.dot(all_v[b] - all_v[a], direction)
                    f = (spring + damping) * direction
                    if a < 2: force[a] += f
                    if b < 2: force[b] -= f
                v = (v + g['dt'] * (force / g['mass'][:2, None] + g['gravity'])) * np.exp(-g['dt'] * g['drag'])
                q += v * g['dt']
            np.testing.assert_allclose(sim.step(target), q, atol=2e-8)
            np.testing.assert_allclose(sim.v.numpy(), v, atol=2e-7)
            sim.reset()
            np.testing.assert_array_equal(sim.q.numpy(), g['q'][:2])

    def test_cuda_capture_matches_cpu_across_frames(self):
        if not wp.is_cuda_available():
            self.skipTest('CUDA unavailable')
        for steps in (2, 3):
            g = fixture(steps)
            cpu = PhysTwinSimulation(g, device='cpu')
            gpu = PhysTwinSimulation(g, device='cuda:0')
            target = cpu.initial_controls + [0, 0, -.002]
            for _ in range(3):
                np.testing.assert_allclose(cpu.step(target), gpu.step(target), atol=1e-7)
            gpu.reset()
            np.testing.assert_array_equal(gpu.q.numpy(), g['q'][:2])

    def test_ground_bounce_and_inactive_springs(self):
        g = fixture()
        g.update(dt=.001, drag=0., y_min=10., friction=np.array([0.], np.float32))
        g['q'][:2, 2] = -.0001
        g['v'][:2] = [0., 0., 1.]
        sim = PhysTwinSimulation(g, device='cpu')
        q = sim.step(sim.initial_controls)
        speed = 1. + 9.8 * g['dt']
        toi = .0001 / speed
        expected_z = -.5 * speed * (g['dt'] - toi)
        np.testing.assert_allclose(q[:, 2], expected_z, atol=1e-8)
        np.testing.assert_allclose(sim.v.numpy()[:, 2], -.5 * speed, atol=1e-7)
        np.testing.assert_allclose(q[:, :2], g['q'][:2, :2], atol=1e-8)

    def test_visual_binding_preserves_rigid_motion(self):
        rng = np.random.default_rng(7)
        particles = rng.normal(size=(32, 3)).astype(np.float32)
        centers = rng.normal(size=(50, 3)).astype(np.float32)
        binding = GaussianBinding(particles, centers)
        np.testing.assert_allclose(binding.deform(particles), centers, atol=1e-6)
        rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        translation = [.1, .2, -.3]
        moved = particles @ rotation.T + translation
        np.testing.assert_allclose(binding.deform(moved), centers @ rotation.T + translation, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
