"""Manual full-graph Panda contact check; no physical robot or calibration claims."""
from pathlib import Path
import hashlib
import json
import time
import newton
import numpy as np
from lab.simulation.demo import NewtonDemo
from lab.simulation.phystwin import DEFAULT_EXPORT, load_graph


def main():
    graph, provenance=load_graph(DEFAULT_EXPORT)
    sim=NewtonDemo(graph,robot=True)
    ply=Path.home()/'PhysTwin/gaussian_output/double_stretch_sloth/init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0/point_cloud/iteration_10000/point_cloud.ply'
    directory=sim.write_manifest(provenance,ply)
    print(directory,flush=True)
    start=time.monotonic()
    try:
        first=sim.step(sim.robot.initial)
        first_joints=sim.state.joint_q.numpy().copy()
        sim.reset()
        np.testing.assert_allclose(sim.step(sim.robot.initial),first,atol=1e-6)
        np.testing.assert_allclose(sim.state.joint_q.numpy(),first_joints,atol=1e-6)
        results=[]
        for contact in [True,False]:
            if not contact:
                flags=sim.model.shape_flags.numpy()
                flags[sim.robot.shape_ids] &= ~int(newton.ShapeFlags.COLLIDE_PARTICLES)
                sim.model.shape_flags.assign(flags)
            sim.reset()
            for _ in range(15):sim.step(sim.robot.initial)
            settled=sim.state.particle_q.numpy().copy()
            for _ in range(90):moved=sim.step(sim.robot.push_target)
            results.append((moved.copy(),sim.state.joint_q.numpy().copy()))
            assert moved[:,2].min()>-.01, 'Ground penetration >1 cm'
            assert np.max(np.linalg.norm(moved-settled,axis=1))<1., 'Object escaped'
            print('contact',contact,'max displacement',float(np.linalg.norm(moved-settled,axis=1).max()),flush=True)
        object_effect=float(np.linalg.norm(results[0][0]-results[1][0],axis=1).max())
        robot_effect=float(np.linalg.norm(results[0][1]-results[1][1]))
        assert object_effect>.005, object_effect
        assert robot_effect>1e-5, robot_effect
        result=dict(status='passed',kind='infrastructure_smoke_test',simulated_frames=212,
                    object_contact_effect_m=object_effect,robot_contact_effect_joint_norm=robot_effect,
                    robot_contact_effect_by_joint=(results[0][1]-results[1][1]).tolist(),
                    joint_units=['rad']*7+['m']*2,reset='passed',elapsed_s=time.monotonic()-start,
                    test_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    except Exception as error:
        result=dict(status='failed',failure_reason=str(error))
        (directory/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        raise
    (directory/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result,flush=True)


if __name__=='__main__':main()
