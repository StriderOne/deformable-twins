"""Seeded top-down contact grasp attempts; failures are recorded, never attached."""
import numpy as np


def surface_candidates(points, cell=.01):
    """Upper envelope of physical particles on a 1 cm XY grid."""
    bins = np.floor(points[:, :2] / cell).astype(np.int64)
    _, groups = np.unique(bins, axis=0, return_inverse=True)
    order = np.argsort(points[:, 2], kind='stable')[::-1]
    _, first = np.unique(groups[order], return_index=True)
    return order[first]


class GraspCycle:
    LIFT_HEIGHT = .30

    def __init__(self, simulation, seed=0, log=lambda event: None):
        self.sim = simulation
        self.rng = np.random.default_rng(seed)
        self.log = log
        self.number = 0
        self.reset()
        self.log(dict(event='grasp_demo', seed=seed, surface_grid_m=.01,
                      approach_clearance_m=.12, lift_m=self.LIFT_HEIGHT,
                      description='contact-only grasp attempt; no attachment'))

    def reset(self):
        self.phase = 'settle'
        self.elapsed = 0.
        self.waypoints = []
        self.target_point = None
        self.start_q = self.sim.robot.initial.copy()

    def plan(self):
        sim = self.sim
        points = sim.state.particle_q.numpy()
        candidates = surface_candidates(points)
        index = int(self.rng.choice(candidates))
        self.target_point = points[index].copy()
        self.number += 1
        self.log(dict(event='grasp_target', cycle=self.number, particle=index,
                      position=self.target_point.tolist(), candidates=len(candidates)))
        self.baseline = points.copy()
        # The tool reference is near the finger tips; lower it around the selected surface point.
        grasp = self.target_point.copy()
        grasp[2] = max(.012, grasp[2] - .018)
        above = grasp + [0., 0., .12]
        lifted = grasp + [0., 0., self.LIFT_HEIGHT]
        q = sim.robot.initial.copy()
        def add(phase, position, opening, minimum):
            nonlocal q
            goal = sim.robot.solve_pose(position, q)
            goal[7:] = opening
            duration = max(minimum, float(np.max(np.abs(goal[:7]-q[:7])))/.35)
            self.waypoints.append((phase, goal, duration))
            q = goal
        add('approach', above, .04, 1.5)
        for alpha in np.linspace(0., 1., 7)[1:]:
            add('descend', above*(1-alpha)+grasp*alpha, .04, .3)
        add('close', grasp, 0., 1.5)
        for alpha in np.linspace(0., 1., int(np.ceil(self.LIFT_HEIGHT/.02))+1)[1:]:
            add('lift', grasp*(1-alpha)+lifted*alpha, 0., .3)
        self.waypoints.append(('hold', q.copy(), .6))
        self.start_q = sim.robot.initial.copy()
        self.advance()

    def advance(self):
        if self.waypoints:
            self.phase, self.goal, self.duration = self.waypoints.pop(0)
            self.elapsed = 0.
            self.log(dict(event='grasp_phase', cycle=self.number, phase=self.phase,
                          target=self.goal.tolist(), duration=self.duration))
        else:
            dz = self.sim.state.particle_q.numpy()[:,2]-self.baseline[:,2]
            self.log(dict(event='grasp_result', cycle=self.number,
                          mean_height_change_m=float(dz.mean()), max_height_change_m=float(dz.max()),
                          fraction_raised_2cm=float(np.mean(dz>.02))))
            self.sim.reset()
            self.reset()
            self.log(dict(event='automatic_reset', cycle=self.number))

    def step(self):
        dt = self.sim.dt*self.sim.substeps
        if self.phase == 'settle':
            self.sim.step(self.sim.robot.initial)
            self.elapsed += dt
            if self.elapsed >= .5:
                try:
                    self.plan()
                except ValueError as error:
                    self.log(dict(event='grasp_unreachable', cycle=self.number, reason=str(error)))
                    self.sim.reset()
                    self.reset()
            return self.sim.state.particle_q.numpy().copy()
        alpha = min(1., (self.elapsed+dt)/self.duration)
        target = self.start_q.astype(float) + (self.goal.astype(float)-self.start_q)*alpha
        self.sim.step(target)
        self.elapsed += dt
        if self.elapsed >= self.duration:
            self.start_q = self.goal.copy()
            self.advance()
        return self.sim.state.particle_q.numpy().copy()
