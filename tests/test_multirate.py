import unittest
from types import SimpleNamespace
import newton
import numpy as np
import warp as wp
from lab.simulation.multirate import MultiRate


class CouplingTests(unittest.TestCase):
    def test_opposite_contact_impulse(self):
        class NoForce:
            def accumulate(self,*args):pass
        class NoMotor:
            def apply(self,*args):pass
        results=[]
        for device in ['cpu']+(['cuda:0'] if wp.is_cuda_available() else []):
            b=newton.ModelBuilder(gravity=(0,0,0))
            body=b.add_link(mass=.1,inertia=wp.mat33(np.eye(3)*2.5e-5),lock_inertia=True)
            j=b.add_joint_prismatic(-1,body,axis=newton.Axis.X,armature=0.,damping=0.)
            b.add_articulation([j])
            b.add_shape_sphere(body,radius=.025,cfg=newton.ModelBuilder.ShapeConfig(density=0.,ke=1000.,kd=0.,kf=0.,mu=0.))
            b.add_particle(pos=(.027,0,0),vel=(-.1,0,0),mass=.02,radius=.003)
            model=b.finalize(device=device);model.particle_grid=None
            model.soft_contact_ke=1000.;model.soft_contact_kd=0.;model.soft_contact_kf=0.;model.soft_contact_mu=0.
            pipeline=newton.CollisionPipeline(model)
            sim=SimpleNamespace(model=model,state=model.state(),scratch=model.state(),control=model.control(),
                contacts=pipeline.contacts(),pipeline=pipeline,law=NoForce(),robot=NoMotor(),dt=1e-5,substeps=47,drag=0.)
            for state in [sim.state,sim.scratch]:newton.eval_fk(model,model.joint_q,model.joint_qd,state)
            coupling=MultiRate(sim,stride=20)
            coupling.frame()
            pv=float(sim.state.particle_qd.numpy()[0,0]);rv=float(sim.state.joint_qd.numpy()[0])
            self.assertGreater(pv,-.1);self.assertLess(rv,0.)
            self.assertAlmostEqual(.02*pv+.1*rv,-.002,places=8)
            results.append([pv,rv])
        if len(results)==2:np.testing.assert_allclose(*results,atol=1e-7)


if __name__=='__main__':unittest.main()
