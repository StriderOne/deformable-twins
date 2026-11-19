import unittest
import numpy as np
from lab.simulation.grasp_cycle import surface_candidates, GraspCycle


class GraspTests(unittest.TestCase):
    def test_upper_surface_excludes_internal_points(self):
        points=np.array([[0,0,.1],[0,0,.2],[.02,0,.3],[.02,0,.05]])
        np.testing.assert_array_equal(surface_candidates(points),[1,2])

    def test_reset_preserves_random_sequence(self):
        class Robot:initial=np.zeros(9)
        class Sim:robot=Robot()
        a,b=GraspCycle(Sim(),seed=12),GraspCycle(Sim(),seed=12)
        self.assertEqual(a.rng.integers(100000),b.rng.integers(100000))
        a.number=3;a.reset()
        self.assertEqual(a.number,3)
        self.assertEqual(a.rng.integers(100000),b.rng.integers(100000))
        self.assertIsNone(a.target_point)

    def test_cycle_closes_lifts_and_resets_without_overshooting_targets(self):
        from types import SimpleNamespace
        class Robot:
            initial=np.array([0.]*7+[.04,.04],np.float32)
            def solve_pose(self, position, seed):
                q=seed.copy();q[:3]=position;return q
        class Sim:
            robot=Robot();dt=.1;substeps=1;time=0.;resets=0
            state=SimpleNamespace(particle_q=SimpleNamespace(numpy=lambda:np.array([[.1,.1,.05],[.2,.2,.08]])))
            def step(self,target):
                assert np.all(target[7:]<=float(np.float32(.04)))
                assert np.all(target[7:]>=0.)
                self.time+=self.dt
                return self.state.particle_q.numpy()
            def reset(self):self.time=0.;self.resets+=1
        sim=Sim();events=[];cycle=GraspCycle(sim,log=events.append)
        for _ in range(300):
            cycle.step()
            if sim.resets:break
        self.assertEqual(sim.resets,1)
        self.assertEqual(cycle.phase,'settle')
        self.assertEqual(sim.time,0.)
        phases=[e['phase'] for e in events if e['event']=='grasp_phase']
        self.assertEqual(list(dict.fromkeys(phases)),['approach','descend','close','lift','hold'])
        self.assertTrue(any(e['event']=='grasp_result' for e in events))
        targets={e['phase']:e['target'] for e in events if e['event']=='grasp_phase'}
        self.assertAlmostEqual(targets['hold'][2]-targets['close'][2],GraspCycle.LIFT_HEIGHT,places=6)


if __name__=='__main__':unittest.main()
