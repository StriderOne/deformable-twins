import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tutorials'))
import unittest
import numpy as np
import warp as wp
from phystwin_runtime import GaussianBinding
from lab.simulation.skinning import GpuGaussianBinding


class SkinTests(unittest.TestCase):
    def test_gpu_matches_reference(self):
        rng=np.random.default_rng(4)
        rest=rng.normal(size=(100,3)).astype(np.float32)*.1
        centers=rng.normal(size=(300,3)).astype(np.float32)*.1
        binding=GaussianBinding(rest,centers)
        for device in ['cpu']+(['cuda:0'] if wp.is_cuda_available() else []):
            gpu=GpuGaussianBinding(binding,device)
            for q in [rest,rest+[.1,.2,-.3],rest@np.array([[0,-1,0],[1,0,0],[0,0,1]],np.float32),rest+rng.normal(size=rest.shape)*.001]:
                q=q.astype(np.float32)
                actual=gpu.deform(wp.array(q,dtype=wp.vec3,device=device)).numpy()
                np.testing.assert_allclose(actual,binding.deform(q),atol=2e-6)


if __name__=='__main__':unittest.main()
