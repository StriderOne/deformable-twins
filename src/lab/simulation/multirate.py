"""Contact-impulse coupling: 50 us particles, slower articulated dynamics."""
import copy
import warp as wp
from newton.solvers import SolverFeatherstone
from newton._src.solvers.semi_implicit.kernels_contact import eval_particle_body_contact_forces
from .internal_forces import SolverInternalForces


@wp.kernel
def average_reaction(total: wp.array[wp.spatial_vector], output: wp.array[wp.spatial_vector], n: float):
    i=wp.tid()
    output[i]=total[i]/n


@wp.kernel
def contact_peak(count: wp.array[int], peak: wp.array[int]):
    wp.atomic_max(peak,0,count[0])


class MultiRate:
    """Views alias Newton state buffers; only accumulated contact force is additional state.

    Contact geometry/body poses are held within each macro step. Forces on particles
    are recalculated every micro step; their opposite impulse drives the robot.
    """
    def __init__(self, sim, stride=20, mass_interval=8):
        self.sim=sim
        self.stride=stride
        model=sim.model
        rigid_model=copy.copy(model)
        rigid_model.particle_count=0
        rigid_model.particle_grid=None
        self.rigid=SolverFeatherstone(rigid_model,angular_damping=0.,update_mass_matrix_interval=mass_interval)
        self.particles=SolverInternalForces(model,[sim.law],drag=float(sim.drag))
        self.views={}
        for state in (sim.state,sim.scratch):
            view=copy.copy(state)
            view.particle_q=view.particle_qd=view.particle_f=None
            self.views[id(state)]=view
        self.rigid_contacts=copy.copy(sim.contacts)
        self.rigid_contacts.soft_contact_max=0
        self.reaction=wp.zeros(model.body_count,dtype=wp.spatial_vector,device=model.device)
        self.peak=wp.zeros(1,dtype=int,device=model.device)

    def frame(self):
        s=self.sim
        a,b=s.state,s.scratch
        self.rigid._step=0  # Every replay/reset starts with a fresh mass matrix.
        self.peak.zero_()
        for start in range(0,s.substeps,self.stride):
            count=min(self.stride,s.substeps-start)
            s.pipeline.collide(a,s.contacts)
            wp.launch(contact_peak,1,[s.contacts.soft_contact_count,self.peak],device=s.model.device)
            self.reaction.zero_()
            for _ in range(count):
                a.particle_f.zero_()
                s.law.accumulate(s.model,a,s.dt)
                eval_particle_body_contact_forces(s.model,a,s.contacts,a.particle_f,self.reaction)
                self.particles.integrate_particles(s.model,a,b,s.dt)
                a,b=b,a
            wp.launch(average_reaction,s.model.body_count,[self.reaction,a.body_f,float(count)],device=s.model.device)
            s.robot.apply(a,s.control)
            self.rigid.step(self.views[id(a)],self.views[id(b)],s.control,self.rigid_contacts,s.dt*count)
            for field in ('body_q','body_qd','joint_q','joint_qd'):
                wp.copy(getattr(a,field),getattr(b,field))
        if a is not s.state:
            for field in ('particle_q','particle_qd','body_q','body_qd','joint_q','joint_qd'):
                wp.copy(getattr(s.state,field),getattr(a,field))
