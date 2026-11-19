"""Pinned Panda asset and bounded joint-space drives for the Newton demo."""
from pathlib import Path
import hashlib
import numpy as np
import newton
import warp as wp
from newton._src.utils.download_assets import download_git_folder

ASSET_URL = 'https://github.com/google-deepmind/mujoco_menagerie.git'
ASSET_COMMIT = 'da76818e269b82289eba39808e2fb91d679d6994'


@wp.kernel
def joint_pd(q: wp.array[float], qd: wp.array[float], target: wp.array[float],
             kp: wp.array[float], kd: wp.array[float], limit: wp.array[float],
             force: wp.array[float]):
    i = wp.tid()
    force[i] = wp.clamp(kp[i] * (target[i] - q[i]) - kd[i] * qd[i], -limit[i], limit[i])


class Panda:
    def __init__(self, builder, rest, contact):
        self.asset = Path(download_git_folder(ASSET_URL, 'franka_emika_panda', ref=ASSET_COMMIT))
        center = (rest.min(0) + rest.max(0)) / 2
        self.approach_position = np.array([rest[:,0].min()-.07, center[1], .08], np.float32)
        self.push_position = self.approach_position + np.array([.16,0,0], np.float32)
        self.base = np.array([center[0]-.5, center[1], 0.], np.float32)
        first = builder.shape_count
        builder.add_mjcf(str(self.asset/'panda.xml'), xform=wp.transform(self.base, wp.quat_identity()),
                         floating=False, collapse_fixed_joints=True, enable_self_collisions=False,
                         parse_sites=False, skip_equality_constraints=True, parse_mujoco_options=False)
        if builder.joint_dof_count != 9 or builder.joint_coord_count != 9:
            raise ValueError('Expected seven Panda arm joints and two independent fingers')
        self.initial = np.array([0., -.5, 0., -2.2, 0., 1.7, np.pi/4, .04, .04], np.float32)
        builder.joint_q[:] = self.initial.tolist()
        # Explicit bounded drives below replace MJCF actuator semantics (including its tendon).
        builder.joint_target_ke[:] = [0.] * 9
        builder.joint_target_kd[:] = [0.] * 9
        self.kp = np.array([400.,400.,300.,300.,150.,100.,50.,100.,100.], np.float32)
        self.kd = np.array([40.,40.,30.,30.,15.,10.,5.,2.,2.], np.float32)
        self.effort = np.array([87.,87.,87.,87.,12.,12.,12.,10.,10.], np.float32)
        for i in range(first, builder.shape_count):
            for name in ('ke','kd','kf','mu'):
                getattr(builder, 'shape_material_'+name)[i] = getattr(contact, 'contact_'+name)
        self.shape_ids = list(range(first, builder.shape_count))

    def initialize(self, model):
        self.model = model
        self.lower, self.upper = model.joint_limit_lower.numpy(), model.joint_limit_upper.numpy()
        self.target = wp.array(self.initial, dtype=float, device=model.device)
        self.kp_gpu, self.kd_gpu, self.effort_gpu = [wp.array(a, dtype=float, device=model.device)
                                                   for a in (self.kp, self.kd, self.effort)]
        import newton.ik as ik
        from scipy.spatial.transform import Rotation
        self.ik_position = ik.IKObjectivePosition(link_index=6, link_offset=wp.vec3(0.,0.,.21),
            target_positions=wp.array([self.approach_position],dtype=wp.vec3,device=model.device))
        rotation=Rotation.from_euler('xyz',[np.pi,0.,np.pi/4]).as_quat()
        self.ik_rotation = ik.IKObjectiveRotation(link_index=6, link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array([rotation],dtype=wp.vec4,device=model.device),weight=.15)
        self.ik_solver=ik.IKSolver(model=model,n_problems=1,
            objectives=[self.ik_position,self.ik_rotation,ik.IKObjectiveJointLimit(
                joint_limit_lower=model.joint_limit_lower,joint_limit_upper=model.joint_limit_upper,weight=10.)],
            lambda_initial=.1,jacobian_mode=ik.IKJacobianType.ANALYTIC)
        self.ik_q=wp.array(self.initial[None],dtype=float,device=model.device)
        self.ik_solver.step(self.ik_q,self.ik_q,iterations=48)
        self.ik_graph=None
        if model.device.is_cuda:
            with wp.ScopedCapture(device=model.device) as capture:
                self.ik_solver.step(self.ik_q,self.ik_q,iterations=48)
            self.ik_graph=capture.graph
        self.initial = self.solve_pose(self.approach_position, self.initial)
        self.push_target = self.solve_pose(self.push_position, self.initial)
        model.joint_q.assign(self.initial)
        self.target.assign(self.initial)
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        wp.copy(model.body_q, state.body_q)
        wp.copy(model.body_qd, state.body_qd)

    def solve_pose(self, position, seed):
        """IK prepares joint targets only; it never writes the simulation state."""
        from scipy.spatial.transform import Rotation
        self.ik_q.assign(np.asarray(seed,np.float32)[None])
        self.ik_position.set_target_position(0,wp.vec3(*position))
        if self.ik_graph:
            wp.capture_launch(self.ik_graph)
        else:
            self.ik_solver.step(self.ik_q,self.ik_q,iterations=48)
        result=self.ik_q.numpy()[0]
        result[7:]=.04
        if np.any(result<self.lower-1e-5) or np.any(result>self.upper+1e-5):
            raise ValueError('IK solution exceeds joint limits')
        result=np.clip(result,self.lower,self.upper)
        state=self.model.state()
        q=wp.array(result,dtype=float,device=self.model.device)
        newton.eval_fk(self.model,q,self.model.joint_qd,state)
        pose=state.body_q.numpy()[6]
        rotation=Rotation.from_quat(pose[3:])
        error=np.linalg.norm(pose[:3]+rotation.apply([0.,0.,.21])-position)
        if error>.005:
            raise ValueError(f'Panda IK position residual {error:.6f} m exceeds 5 mm')
        return result

    def set_target(self, target, dt=None):
        target = np.asarray(target, dtype=np.float32)
        if target.shape != (9,) or not np.isfinite(target).all():
            raise ValueError('Expected nine finite Panda joint targets (rad, rad, rad, rad, rad, rad, rad, m, m)')
        if np.any(target < self.lower) or np.any(target > self.upper):
            raise ValueError('Panda target exceeds joint limits')
        if dt is not None:
            previous = self.target.numpy()
            rates = np.array([.5]*7+[.04]*2)
            target = previous + np.clip(target-previous, -rates*dt, rates*dt)
        self.target.assign(target)

    def apply(self, state, control):
        wp.launch(joint_pd, 9, [state.joint_q,state.joint_qd,self.target,self.kp_gpu,
                               self.kd_gpu,self.effort_gpu,control.joint_f], device=self.model.device)

    def metadata(self):
        files = sorted(p for p in self.asset.rglob('*') if p.is_file())
        return dict(name='Franka Emika Panda', asset_url=ASSET_URL, asset_commit=ASSET_COMMIT,
                    asset_path=str(self.asset), license='Apache-2.0', ik='Newton analytic LM, 48 iterations',
                    asset_sha256={str(p.relative_to(self.asset)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                    base=self.base.tolist(), initial_q=self.initial.tolist(), kp=self.kp.tolist(),
                    kd=self.kd.tolist(), effort_limits=self.effort.tolist(),
                    target_rate_limits=[.5]*7+[.04]*2,
                    control='bounded joint PD; independent finger drives with shared GUI target',
                    approach_position=self.approach_position.tolist(), push_target=self.push_target.tolist(),
                    self_collisions=False, solver='SolverFeatherstone', mass_matrix_interval=1)


class PandaVisual:
    def __init__(self, scene, simulation):
        model = simulation.model
        self.frames = [scene.add_frame(f'/panda/link_{i}', show_axes=False) for i in range(model.body_count)]
        bodies, flags = model.shape_body.numpy(), model.shape_flags.numpy()
        local, scales = model.shape_transform.numpy(), model.shape_scale.numpy()
        colors = model.shape_color.numpy()
        for i in simulation.robot.shape_ids:
            if not flags[i] & int(newton.ShapeFlags.VISIBLE):
                continue
            mesh = model.shape_source[i]
            if not isinstance(mesh, newton.Mesh):
                raise ValueError(f'Unexpected Panda visual geometry at shape {i}')
            parent = f'/panda/link_{bodies[i]}' if bodies[i] >= 0 else '/panda/static'
            scene.add_mesh_simple(f'{parent}/shape_{i}', vertices=np.asarray(mesh.vertices)*scales[i],
                                  faces=np.asarray(mesh.indices).reshape(-1,3),
                                  color=tuple((255*np.clip(colors[i],0,1)).astype(int)),
                                  position=local[i,:3], wxyz=np.roll(local[i,3:],1))
        self.update(simulation)

    def update(self, simulation):
        for frame, pose in zip(self.frames, simulation.state.body_q.numpy()):
            frame.position = pose[:3]
            frame.wxyz = np.roll(pose[3:],1)
