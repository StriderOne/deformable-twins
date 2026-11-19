"""Manual infrastructure smoke test on the local full graph; not a scientific evaluation."""
from pathlib import Path
import json
import hashlib
import newton
import time

import numpy as np

from lab.simulation.demo import NewtonDemo
from lab.simulation.phystwin import load_graph, DEFAULT_EXPORT


def main():
    g, provenance = load_graph(DEFAULT_EXPORT)
    sim = NewtonDemo(g)
    directory = sim.write_manifest(provenance, Path.home() / 'PhysTwin/gaussian_output/double_stretch_sloth'
        / 'init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0/point_cloud/iteration_10000/point_cloud.ply')
    try:
        start = time.monotonic()
        steps = 0
        def step(target):
            nonlocal steps
            steps += 1
            return sim.step(target)
        first = step(sim.sphere_initial)
        for _ in range(2):
            sim.reset()
            np.testing.assert_array_equal(sim.state.particle_q.numpy(), sim.model.particle_q.numpy())
            np.testing.assert_allclose(step(sim.sphere_initial), first, atol=1e-6)
        for _ in range(29): step(sim.sphere_initial)
        settled = sim.state.particle_q.numpy().copy()
        target = sim.sphere_initial.copy()
        for i in range(100):
            target[0] = sim.sphere_initial[0] + min(i*.003, .25)
            moved = step(target)
        sphere_with_object = sim.sphere_position.copy()
        displacement = float(np.max(np.linalg.norm(moved-settled,axis=1)))
        assert displacement > .005, displacement
        assert moved[:,2].min() > -.01, 'Object penetrated ground by >1 cm'
        assert np.max(np.linalg.norm(moved-settled,axis=1)) < 1., 'Object escaped the scene'
        # Same commands with particle contact disabled: tests that the object pushes back on the sphere.
        sim.model.shape_flags.assign(sim.model.shape_flags.numpy() & ~int(newton.ShapeFlags.COLLIDE_PARTICLES))
        sim.reset()
        for _ in range(30):step(sim.sphere_initial)
        for i in range(100):
            target[0]=sim.sphere_initial[0]+min(i*.003,.25)
            step(target)
        sphere_without_object=sim.sphere_position
        reaction=float(np.linalg.norm(sphere_with_object-sphere_without_object))
        assert reaction > .0001, reaction
        result=dict(status='passed', kind='infrastructure_smoke_test', simulated_frames=steps, test_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    max_particle_displacement_m=displacement, sphere_contact_effect_m=reaction,
                    sphere_with_object=sphere_with_object.tolist(), sphere_without_object=sphere_without_object.tolist(),
                    elapsed_s=time.monotonic()-start, reset='passed')
        (directory/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        print(directory, result)
    except Exception as error:
        (directory/'result.json').write_text(json.dumps(dict(status='failed',failure_reason=str(error)),indent=2)+'\n')
        raise


if __name__=='__main__': main()
