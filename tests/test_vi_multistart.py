import json
from pathlib import Path
import pickle
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import torch
from lab.calibration.posterior import LowRankUniformPosterior
from lab.calibration.vi_multistart_monitor import evaluate_current, MultistartMonitor

class MultistartTests(unittest.TestCase):
    def posterior(self):
        return LowRankUniformPosterior(torch.full((3,),np.log(2000),dtype=torch.float32),1.5,100,100000,16,2)

    def test_evaluation_preserves_rng_and_parameters(self):
        q=self.posterior()
        class Rollout:
            sim=types.SimpleNamespace(valid_counts=np.array([2,2,2]))
            def trajectory(self,z,end,backward,deadline):
                self.assert_fixed=z.detach().clone()
                return np.full(3,float(z.square().sum())),None
        plan={'periodic_evaluation':{'seed':43,'posterior_samples':2},'likelihood':{'sigma_obs_m':.01}}
        protocol={'splits':{'calibration_targets':[1,3],'continuation_targets':[3,4]}}
        rng=torch.get_rng_state().clone();state={k:v.clone() for k,v in q.state_dict().items()}
        with tempfile.TemporaryDirectory() as tmp:
            result=evaluate_current(Rollout(),q,plan,protocol,Path(tmp),100,float('inf'))
            self.assertIn('sample_mean',result)
            self.assertEqual(len(json.loads((Path(tmp)/'periodic-evaluation.jsonl').read_text())['samples']),3)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for k,v in q.state_dict().items():torch.testing.assert_close(v,state[k])

    def test_paired_images_and_physical_means(self):
        q=self.posterior()
        comet=MagicMock();comet.url='https://example.invalid/test';comet.get_key.return_value='test'
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);points=np.array([[0.,0,0],[1.,0,0],[0.,1,0]])
            data={'object_points':points[None,:,:],'surface_points':np.empty((0,3)),'interior_points':np.empty((0,3))}
            with (root/'observations.pkl').open('wb') as f:pickle.dump(data,f)
            np.savez(root/'topology.npz',edges=np.array([[0,1],[0,2],[1,2]]),rest=np.array([1.,1.,2**.5]))
            plan={'id':'E-0013','comet':{'project':'test'},'tensorboard':{'image_every':100},'data':{'observations':str(root/'observations.pkl')},'posterior':{'initial_median_Y':2000},'optimizer':{'seed':42,'max_iterations':1000,'update_every_frames':1}}
            with patch('comet_ml.start',return_value=comet):
                monitor=MultistartMonitor(plan,root,q,3)
                monitor.close()
            with np.load(root/'posterior-000000.npz') as actual:
                np.testing.assert_allclose(actual['mean_Y'],q.physical_moments()[0].detach().numpy())
                self.assertFalse(np.allclose(actual['mean_Y'],q.point_estimate().detach().exp().numpy()))
            self.assertTrue((root/'springs-000000.png').exists())
            comet.log_image.assert_called_once()

if __name__=='__main__':unittest.main()
