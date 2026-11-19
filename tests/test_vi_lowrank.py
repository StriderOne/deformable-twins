import unittest
import torch
from torch.func import functional_call
from lab.calibration.posterior import LowRankUniformPosterior

class LowRankTests(unittest.TestCase):
    def test_dense_kl_and_gradients(self):
        q=LowRankUniformPosterior(torch.full((4,),9.,dtype=torch.float64),1.5,100.,100000.,128,2)
        cov=torch.diag(q.diagonal_std.square())+q.factor@q.factor.T
        entropy=torch.distributions.MultivariateNormal(q.mu,covariance_matrix=cov).entropy()
        u=q.mu+q.std*q.nodes
        expected=((torch.nn.functional.softplus(u)+torch.nn.functional.softplus(-u))*q.weights).sum()-entropy
        torch.testing.assert_close(q.kl(),expected)
        q.forward=lambda: q.kl()+q.sample(torch.ones_like(q.mu),torch.ones(2,dtype=q.mu.dtype)).sum()
        params=dict(q.named_parameters())
        self.assertTrue(torch.autograd.gradcheck(lambda *v:functional_call(q,dict(zip(params,v)),()),tuple(params.values())))
        torch.testing.assert_close(q.std,torch.full_like(q.std,1.5))
