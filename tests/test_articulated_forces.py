"""Small independent check of bidirectional particle/articulation contacts."""
import unittest
import newton
import numpy as np
import warp as wp
from lab.simulation.internal_forces import SolverArticulatedInternalForces
from lab.simulation.panda import joint_pd


class ArticulatedTests(unittest.TestCase):
    def test_contact_reaction_cpu_cuda(self):
        results=[]
        for device in ['cpu'] + (['cuda:0'] if wp.is_cuda_available() else []):
            builder=newton.ModelBuilder(gravity=(0,0,0))
            body=builder.add_link(mass=.1,inertia=wp.mat33(np.eye(3)*2.5e-5),lock_inertia=True)
            joint=builder.add_joint_prismatic(-1,body,axis=newton.Axis.X,armature=0.,damping=0.)
            builder.add_articulation([joint])
            builder.add_shape_sphere(body,radius=.025,cfg=newton.ModelBuilder.ShapeConfig(density=0,ke=1000,kd=0,kf=0,mu=0))
            builder.add_particle(pos=(.027,0,0),vel=(-.1,0,0),mass=.02,radius=.003)
            model=builder.finalize(device=device);model.particle_grid=None
            model.soft_contact_ke=1000.;model.soft_contact_kd=0.;model.soft_contact_kf=0.;model.soft_contact_mu=0.
            a,b=model.state(),model.state();newton.eval_fk(model,model.joint_q,model.joint_qd,a)
            pipeline=newton.CollisionPipeline(model);contacts=pipeline.contacts()
            solver=SolverArticulatedInternalForces(model,[],angular_damping=0.)
            for _ in range(30):
                a.clear_forces();pipeline.collide(a,contacts)
                solver.step(a,b,None,contacts,1e-5);a,b=b,a
            pv=float(a.particle_qd.numpy()[0,0]);rv=float(a.joint_qd.numpy()[0])
            self.assertGreater(pv,-.1);self.assertLess(rv,0.)
            self.assertAlmostEqual(.02*pv+.1*rv,-.002,places=8)
            results.append([pv,rv])
        if len(results)==2:np.testing.assert_allclose(*results,atol=1e-7)

    def test_pd_effort_cap(self):
        arrays=[wp.array(x,dtype=float,device='cpu') for x in
                ([0.,0.],[0.,0.],[10.,-10.],[100.,100.],[2.,2.],[12.,10.],[0.,0.])]
        wp.launch(joint_pd,2,arrays,device='cpu')
        np.testing.assert_array_equal(arrays[-1].numpy(),[12.,-10.])


if __name__=='__main__':unittest.main()
