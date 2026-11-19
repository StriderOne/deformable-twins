import unittest
import math
import torch
from lab.calibration.posterior import UniformStiffnessPosterior


class UniformTests(unittest.TestCase):
    def test_support_and_median(self):
        center=torch.tensor([1000.,37511.,90000.],dtype=torch.float64).log()
        q=UniformStiffnessPosterior(center,1.5,100.,100000.)
        torch.testing.assert_close(q.point_estimate(),center)
        for eps in [-100.,0.,100.]:
            y=q.sample(torch.full_like(center,eps)).exp()
            self.assertTrue((y>=100.-1e-9).all() and (y<=100000.+1e-9).all())
        self.assertTrue(torch.isfinite(q.kl()))

    def test_change_of_variables_kl_and_quadrature(self):
        q=UniformStiffnessPosterior(torch.full((3,),math.log(37511.),dtype=torch.float64),1.5,100.,100000.,128)
        with torch.no_grad():q.mu.copy_(torch.tensor([-3.,0.,3.]))
        u=q.mu[None,:]+q.std[None,:]*q.nodes
        logq=torch.distributions.Normal(q.mu,q.std).log_prob(u)
        jac=(q.upper-q.lower).log()+torch.nn.functional.logsigmoid(u)+torch.nn.functional.logsigmoid(-u)
        expected=((logq-jac+(q.upper-q.lower).log())*q.weights).sum()
        torch.testing.assert_close(q.kl(),expected)
        mean,std=q.physical_moments()
        self.assertTrue((mean>q.lower).all() and (mean<q.upper).all() and (std>0).all())
        params=(q.mu.detach().requires_grad_(),q.rho.detach().requires_grad_())
        def objective(mu,rho):
            sd=torch.nn.functional.softplus(rho)
            u=mu[None,:]+sd[None,:]*q.nodes
            return (((torch.nn.functional.softplus(u)+torch.nn.functional.softplus(-u))*q.weights).sum(0)-sd.log()-.5*math.log(2*math.pi*math.e)).sum()
        self.assertTrue(torch.autograd.gradcheck(objective,params))

    def test_reparameterized_log_stiffness_gradient(self):
        q=UniformStiffnessPosterior(torch.tensor([math.log(37511.)],dtype=torch.float64),1.5,100.,100000.)
        eps=torch.tensor([.4],dtype=torch.float64)
        def f(mu,rho):return q.log_stiffness(mu+torch.nn.functional.softplus(rho)*eps)
        self.assertTrue(torch.autograd.gradcheck(f,(q.mu,q.rho)))
