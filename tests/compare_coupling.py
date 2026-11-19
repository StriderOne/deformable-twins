"""Manual bounded comparison on identical joint commands; not calibration evidence."""
from pathlib import Path
import json,time,hashlib
import numpy as np
import warp as wp
from lab.simulation.demo import NewtonDemo
from lab.simulation.phystwin import DEFAULT_EXPORT,load_graph


def main():
    graph,provenance=load_graph(DEFAULT_EXPORT)
    results=[];trajectories=[];commands=None
    ply=Path.home()/'PhysTwin/gaussian_output/double_stretch_sloth/init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0/point_cloud/iteration_10000/point_cloud.ply'
    for fast in [False,True]:
        sim=NewtonDemo(graph,robot=True,fast=fast)
        if commands is None:
            initial=sim.robot.initial.copy()
            body=sim.model.body_q.numpy().copy()
            commands=[initial.copy() for _ in range(10)]+[initial+(sim.robot.push_target-initial)*min((i+1)/40,1) for i in range(50)]
        else:
            sim.robot.initial=initial.copy();sim.model.joint_q.assign(initial);sim.model.body_q.assign(body);sim.reset()
        directory=sim.write_manifest(provenance,ply)
        q,j,times=[],[],[]
        for command in commands:
            t=time.perf_counter();q.append(sim.step(command));j.append(sim.state.joint_q.numpy());times.append(time.perf_counter()-t)
        trajectories.append((np.array(q),np.array(j)))
        np.savez_compressed(directory/'comparison.npz',particle_q=q,joint_q=j,commands=commands)
        results.append(dict(mode='fast' if fast else 'reference',directory=str(directory),median_frame_ms=float(np.median(times)*1000)))
        print(results[-1],flush=True)
    dq=trajectories[1][0]-trajectories[0][0]
    dj=trajectories[1][1]-trajectories[0][1]
    metrics=dict(particle_rmse_m=float(np.sqrt(np.mean(dq*dq))),max_particle_error_m=float(np.linalg.norm(dq,axis=2).max()),max_joint_error=float(abs(dj).max()))
    passed=metrics['particle_rmse_m']<.005 and metrics['max_particle_error_m']<.02 and metrics['max_joint_error']<.02
    result=dict(status='passed' if passed else 'failed',metrics=metrics,runs=results,
                criteria=dict(particle_rmse_m=.005,max_particle_error_m=.02,max_joint_error=.02),
                test_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    for row in results:(Path(row['directory'])/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result,flush=True)
    assert passed,metrics


if __name__=='__main__':main()
