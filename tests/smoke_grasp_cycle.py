"""Manual contact grasp cycle check on the local PhysTwin export."""
import json
import argparse
from pathlib import Path
import hashlib
from lab.simulation.demo import NewtonDemo
from lab.simulation.grasp_cycle import GraspCycle
from lab.simulation.phystwin import load_graph, DEFAULT_EXPORT


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--fast',action='store_true')
    args=parser.parse_args()
    graph, provenance=load_graph(DEFAULT_EXPORT)
    sim=NewtonDemo(graph,robot=True,fast=args.fast)
    ply=Path.home()/'PhysTwin/gaussian_output/double_stretch_sloth/init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0/point_cloud/iteration_10000/point_cloud.ply'
    directory=sim.write_manifest(provenance,ply)
    events=[]
    def log(event):
        events.append(event)
        with (directory/'events.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
        print(event,flush=True)
    cycle=GraspCycle(sim,seed=0,log=log)
    print(directory,flush=True)
    try:
        for frame in range(900):
            cycle.step()
            if any(e['event']=='automatic_reset' for e in events):break
        else:raise AssertionError('No completed cycle within 900 frames')
        phases={e.get('phase') for e in events}
        assert {'approach','descend','close','lift','hold'}<=phases
        assert sim.time == 0. and cycle.phase=='settle'
        for _ in range(16):cycle.step()
        assert cycle.number>=2, 'Next random target was not selected'
        result=dict(status='passed',kind='infrastructure_cycle_check',frames=frame+17,
                    test_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    grasp_results=[e for e in events if e['event']=='grasp_result'])
    except Exception as error:
        (directory/'result.json').write_text(json.dumps(dict(status='failed',reason=str(error))))
        raise
    (directory/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result,flush=True)


if __name__=='__main__':main()
