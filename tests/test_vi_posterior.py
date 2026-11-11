import unittest
try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name == 'torch':
        raise unittest.SkipTest('VI tests require the separate phystwin-vi environment') from exc
    raise
from lab.calibration.posterior import LogStiffnessPosterior
from lab.calibration.phystwin_vi import metrics, ROOT


class PosteriorTests(unittest.TestCase):
    def test_kl_matches_distribution(self):
        q=LogStiffnessPosterior(torch.tensor([1.,2.],dtype=torch.float64),0.2,0.5)
        with torch.no_grad(): q.mu.add_(0.1)
        expected=torch.distributions.kl_divergence(
            torch.distributions.Normal(q.mu,q.std),
            torch.distributions.Normal(q.prior_mu,q.prior_std)).sum()
        torch.testing.assert_close(q.kl(),expected)

    def test_reparameterized_vjp(self):
        q=LogStiffnessPosterior(torch.zeros(2,dtype=torch.float64),0.2,0.5)
        eps=torch.tensor([0.3,-0.7],dtype=torch.float64)
        z=q.sample(eps);g=torch.tensor([2.,-3.],dtype=torch.float64)
        z.backward(g)
        torch.testing.assert_close(q.mu.grad,g)
        torch.testing.assert_close(q.rho.grad,g*eps*torch.sigmoid(q.rho))

    def test_zero_kl_at_prior(self):
        q=LogStiffnessPosterior(torch.zeros(3),0.5,0.5)
        self.assertAlmostEqual(q.kl().item(),0,places=6)

    def test_metrics_aggregation_and_split(self):
        import numpy as np
        m=metrics(np.array([1.,2.,3.]),np.array([1,2,3]),0.1,3)
        self.assertEqual(m['calibration']['valid_coordinates'],9)
        self.assertEqual(m['continuation']['nll_without_constant'],3.)
        self.assertAlmostEqual(m['calibration']['rmse_m'],(0.06/9)**0.5)

    def test_data_validation_rejects_bad_masks(self):
        import numpy as np
        from lab.calibration.phystwin_adapter import validate_data
        d=dict(object_points=np.zeros((2,2,3)),controller_points=np.zeros((2,1,3)),
               surface_points=np.zeros((0,3)),interior_points=np.zeros((0,3)),
               object_visibilities=np.ones((2,2),dtype=bool),object_motions_valid=np.ones((2,2),dtype=bool))
        validate_data(d)
        d['object_visibilities']=np.ones((2,2))
        with self.assertRaises(ValueError): validate_data(d)

    def test_prior_initialized_experiment(self):
        import yaml
        old=yaml.safe_load((ROOT/'experiments/E-0001.yaml').read_text())
        new=yaml.safe_load((ROOT/'experiments/E-0002.yaml').read_text())
        self.assertEqual(new['posterior']['initial_std'],new['posterior']['prior_std'])
        for key in ('data','likelihood','optimizer','budget','execution'):
            self.assertEqual(old[key],new[key])
        self.assertEqual(new['baseline_by_mode']['vi'],'BL-0004')

    def test_plan_splits(self):
        import json, yaml, jsonschema
        p=yaml.safe_load((ROOT/'experiments/E-0001.yaml').read_text())
        jsonschema.validate(p,json.loads((ROOT/'schemas/vi-experiment.schema.json').read_text()))
        proto=yaml.safe_load((ROOT/'registry/protocols/P-0001.yaml').read_text())
        self.assertEqual(proto['splits']['calibration_targets'][1],134)
        self.assertFalse(p['execution']['hardware_execution'])

if __name__=='__main__': unittest.main()
