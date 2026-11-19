"""Particle force laws composed with Newton contacts and integration."""
from typing import Protocol, Sequence

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverSemiImplicit, SolverFeatherstone


class InternalForceModel(Protocol):
    """Accumulate forces without clearing, advancing positions or resolving contacts."""
    def accumulate(self, model: newton.Model, state: newton.State, dt: float) -> None: ...
    def reset(self) -> None: ...


@wp.kernel
def apply_particle_drag(q0: wp.array[wp.vec3], q: wp.array[wp.vec3],
                        v: wp.array[wp.vec3], inv_mass: wp.array[float],
                        flags: wp.array[wp.int32], drag: wp.array[float], dt: float):
    i = wp.tid()
    if (flags[i] & newton.ParticleFlags.ACTIVE) != 0 and inv_mass[i] > 0.0:
        v1 = v[i] * wp.exp(-drag[i] * dt)
        v[i] = v1
        q[i] = q0[i] + dt * v1


class _InternalForces:
    """Newton's ordinary rigid/particle contacts plus replaceable particle forces.

    Caller clears state forces and adds external forces before each step.
    Input and output states must be distinct. This adapter is forward-only.
    """
    def __init__(self, model: newton.Model, laws: Sequence[InternalForceModel],
                 drag: float | np.ndarray = 0.0, **kwargs):
        super().__init__(model, **kwargs)
        self.laws = tuple(laws)
        rates = np.broadcast_to(np.asarray(drag, dtype=np.float32), (model.particle_count,)).copy()
        if not np.isfinite(rates).all() or np.any(rates < 0):
            raise ValueError('Drag must be finite and non-negative')
        self.drag = wp.array(rates, dtype=float, device=model.device)

    def step(self, state_in, state_out, control, contacts, dt):
        if state_in is state_out:
            raise ValueError('Use distinct Newton input/output states')
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('dt must be finite and positive')
        for law in self.laws:
            law.accumulate(self.model, state_in, dt)
        super().step(state_in, state_out, control, contacts, dt)

    def integrate_particles(self, model, state_in, state_out, dt):
        super().integrate_particles(model, state_in, state_out, dt)
        if model.particle_count:
            wp.launch(apply_particle_drag, model.particle_count,
                      [state_in.particle_q, state_out.particle_q, state_out.particle_qd,
                       model.particle_inv_mass, model.particle_flags, self.drag, dt], device=model.device)

    def reset_internal_models(self):
        for law in self.laws:
            law.reset()


class SolverInternalForces(_InternalForces, SolverSemiImplicit):
    """Particle force laws with maximal-coordinate rigid bodies."""


class SolverArticulatedInternalForces(_InternalForces, SolverFeatherstone):
    """Same particle laws and drag with dynamic reduced-coordinate robots."""
