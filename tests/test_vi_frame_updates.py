import time
import unittest
import numpy as np
import torch
from lab.calibration.posterior import LowRankUniformPosterior
from lab.calibration.vi_frame_updates import train_pass

class ToyRollout:
    def __init__(self, center):
        self.z_buffer = center.detach().clone()
        self.frames = []
    def reset(self, z):
        self.z_buffer.copy_(z.detach())
    def frame(self, frame, backward):
        self.frames.append(frame)
        return float(self.z_buffer.square().sum()/2), self.z_buffer.clone()

class FrameUpdateTests(unittest.TestCase):
    def make(self):
        torch.manual_seed(42)
        q = LowRankUniformPosterior(torch.full((4,), 5.0), 1.5, 100, 100000, 16, 2)
        return q, torch.optim.Adam(q.parameters(), lr=.001)
    def test_intervals_and_whole_pass_equivalence(self):
        for interval in (42, 7, 1):
            q, opt = self.make()
            rollout = ToyRollout(q.mu)
            losses, penalty, _ = train_pass(rollout, q, opt, 43, interval, time.monotonic()+30)
            self.assertEqual(rollout.frames, list(range(1,43)))
            self.assertEqual(int(opt.state[q.mu]['step']), 42//interval)
            self.assertTrue(np.isfinite(losses).all())
            self.assertTrue(np.isfinite(penalty))
        q, opt = self.make()
        state = torch.get_rng_state()
        z = q.sample()
        kl = q.kl()
        torch.autograd.backward((z, kl), (42*z.detach(), torch.ones_like(kl)))
        opt.step()
        expected = {k:v.clone() for k,v in q.state_dict().items()}
        q, opt = self.make()
        torch.set_rng_state(state)
        train_pass(ToyRollout(q.mu), q, opt, 43, 42, time.monotonic()+30)
        for k,v in q.state_dict().items():
            torch.testing.assert_close(v, expected[k], rtol=1e-5, atol=1e-6)
    def test_deadline_fails(self):
        q, opt = self.make()
        with self.assertRaises(TimeoutError):
            train_pass(ToyRollout(q.mu), q, opt, 43, 7, 0)

if __name__ == '__main__':
    unittest.main()
