import unittest
import newton
import numpy as np
import warp as wp

from lab.simulation.internal_forces import SolverInternalForces
from lab.simulation.phystwin import PhysTwinInternalForces


def graph(y=3.):
    return dict(n_object=2, edges=np.array([[0,1],[1,2]],np.int32),
                rest=np.array([.1,.1],np.float32),log_y=np.log(np.array([y,4.],np.float32)),
                damping=.15,y_min=1.,y_max=10.,drag=3.,dt=.0001,substeps=3,
                q=np.array([[0,0,.2],[.12,0,.2],[.25,0,.2]],np.float32),
                v=np.array([[.01,0,0],[-.02,0,0],[0,0,0]],np.float32),
                mass=np.array([.1,.2,0],np.float32),gravity=np.array([0,0,9.8]))


def particles(g, scale=1., device='cpu', gravity=(0,0,0)):
    b=newton.ModelBuilder(gravity=gravity)
    b.add_particles(pos=g['q'][:2],vel=g['v'][:2],mass=g['mass'][:2]*scale,radius=[.003,.003])
    m=b.finalize(device=device);m.particle_grid=None;m.particle_max_velocity=1e6
    return m


class ConstantForce:
    def __init__(self, model):
        self.forces=wp.array([[.1,0,0],[-.1,0,0]],dtype=wp.vec3,device=model.device)
        self.was_reset=False
    def accumulate(self, model, state, dt):
        wp.launch(add_force,model.particle_count,[self.forces,state.particle_f],device=model.device)
    def reset(self): self.was_reset=True


@wp.kernel
def add_force(source:wp.array[wp.vec3], dest:wp.array[wp.vec3]):
    i=wp.tid()
    dest[i]=dest[i]+source[i]


class InternalForceTests(unittest.TestCase):
    def test_activation_clamp_damping_and_external_force(self):
        for raw in [.5,3.,1000.]:
            g=graph(raw);m=particles(g);s=m.state()
            s.particle_f.assign(np.array([[.7,0,0],[.7,0,0]],np.float32))
            law=PhysTwinInternalForces(g,m);law.accumulate(m,s,.001)
            self.assertEqual(law.count,1)
            amp=np.clip(raw,1,10)*(.12/.1-1)+.15*(-.02-.01) if raw>1 else 0.
            np.testing.assert_allclose(s.particle_f.numpy(),[[.7+amp,0,0],[.7-amp,0,0]],atol=1e-6)
            self.assertEqual(m.spring_count,0)

    def test_mass_scaling_preserves_trajectory_and_drag(self):
        g=graph();results=[]
        for scale in [1.,.002]:
            m=particles(g,scale,gravity=(0,0,-9.8));a,b=m.state(),m.state()
            solver=SolverInternalForces(m,[PhysTwinInternalForces(g,m,mass_scale=scale)],drag=3.)
            for _ in range(30):
                a.clear_forces();solver.step(a,b,None,None,g['dt']);a,b=b,a
            results.append(a.particle_q.numpy())
        np.testing.assert_allclose(*results,atol=1e-7)
        # Free fall with drag, independent of the internal law.
        m=particles(g,gravity=(0,0,-9.8));a,b=m.state(),m.state()
        solver=SolverInternalForces(m,[],drag=3.)
        a.clear_forces();solver.step(a,b,None,None,.001)
        expected_v=(g['v'][:2]+np.array([0,0,-9.8])*.001)*np.exp(-.003)
        np.testing.assert_allclose(b.particle_qd.numpy(),expected_v,atol=1e-8)
        np.testing.assert_allclose(b.particle_q.numpy(),g['q'][:2]+expected_v*.001,atol=1e-8)

    def test_replace_law_and_reset(self):
        g=graph();m=particles(g);law=ConstantForce(m)
        solver=SolverInternalForces(m,[law]);a,b=m.state(),m.state()
        a.clear_forces();solver.step(a,b,None,None,.001)
        np.testing.assert_allclose(b.particle_qd.numpy()[:,0],g['v'][:2,0]+np.array([.1,-.1])/g['mass'][:2]*.001,atol=1e-8)
        solver.reset_internal_models();self.assertTrue(law.was_reset)

    def test_cpu_cuda_trajectory(self):
        if not wp.is_cuda_available():self.skipTest('CUDA unavailable')
        g=graph();results=[]
        for device in ['cpu','cuda:0']:
            m=particles(g,device=device);a,b=m.state(),m.state()
            solver=SolverInternalForces(m,[PhysTwinInternalForces(g,m)],drag=3.)
            for _ in range(10):
                a.clear_forces();solver.step(a,b,None,None,.0001);a,b=b,a
            results.append(a.particle_q.numpy())
        np.testing.assert_allclose(*results,atol=1e-7)

    def test_contact_reaction_and_momentum_both_directions(self):
        for moving in ['particle','body']:
            b=newton.ModelBuilder(gravity=(0,0,0))
            b.add_particle(pos=(.027,0,0),vel=(-.1,0,0) if moving=='particle' else (0,0,0),mass=.02,radius=.003)
            body=b.add_body(mass=.1,inertia=wp.mat33(np.eye(3)*2.5e-5),lock_inertia=True)
            b.add_shape_sphere(body,radius=.025,cfg=newton.ModelBuilder.ShapeConfig(density=0,ke=1000,kd=0,kf=0,mu=0))
            m=b.finalize(device='cpu');m.particle_grid=None
            m.soft_contact_ke=1000.;m.soft_contact_kd=0.;m.soft_contact_kf=0.;m.soft_contact_mu=0.
            a,c=m.state(),m.state()
            if moving=='body': a.body_qd.assign(np.array([[.1,0,0,0,0,0]],np.float32))
            start_p=a.particle_qd.numpy().copy();start_b=a.body_qd.numpy().copy()
            momentum0=.02*start_p[0]+.1*start_b[0,:3]
            pipeline=newton.CollisionPipeline(m);contacts=pipeline.contacts()
            solver=SolverInternalForces(m,[],angular_damping=0.)
            for _ in range(30):
                a.clear_forces();pipeline.collide(a,contacts)
                solver.step(a,c,None,contacts,1e-5);a,c=c,a
            end_p=a.particle_qd.numpy();end_b=a.body_qd.numpy()
            self.assertGreater(end_p[0,0]-start_p[0,0],1e-4)
            self.assertLess(end_b[0,0]-start_b[0,0],-1e-4)
            np.testing.assert_allclose(.02*end_p[0]+.1*end_b[0,:3],momentum0,atol=1e-8)


if __name__=='__main__':unittest.main()
