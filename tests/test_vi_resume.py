import copy
import random
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from lab.calibration.posterior import LogStiffnessPosterior
from lab.calibration.phystwin_vi import checkpoint, restore_checkpoint


class ResumeTests(unittest.TestCase):
    def test_exact_next_update_and_reject_changed_likelihood(self):
        q = LogStiffnessPosterior(torch.zeros(8, device='cuda'), .05, .5)
        opt = torch.optim.Adam(q.parameters(), lr=.001)
        def step():
            opt.zero_grad()
            sample = q.sample()
            (sample.square().sum() + q.kl()).backward()
            opt.step()
            return sample.detach().clone()
        step()
        config = dict(data={}, likelihood={'sigma': .01}, posterior={}, execution={},
                      optimizer={'max_iterations': 20, 'lr': .001})
        meta = dict(source_commit='test', topology_sha256='test', config=config)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'checkpoint.pt'
            checkpoint(path, q, opt, 1, 'vi', meta)
            expected_sample = step()
            expected = copy.deepcopy(q.state_dict())
            expected_random = (random.random(), np.random.rand(), torch.rand(2))
            self.assertEqual(restore_checkpoint(path, q, opt, 'vi', meta), 1)
            torch.testing.assert_close(step(), expected_sample, rtol=0, atol=0)
            for key, value in q.state_dict().items():
                torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
            self.assertEqual(random.random(), expected_random[0])
            self.assertEqual(np.random.rand(), expected_random[1])
            torch.testing.assert_close(torch.rand(2), expected_random[2], rtol=0, atol=0)
            bad = copy.deepcopy(meta)
            bad['config']['likelihood']['sigma'] = .02
            with self.assertRaises(ValueError):
                restore_checkpoint(path, q, opt, 'vi', bad)
