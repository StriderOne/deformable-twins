"""Check sensitivity aggregation against an analytic prior-following example."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import torch
from lab.calibration.phystwin_adapter import ROOT
from lab.calibration.posterior import LogStiffnessPosterior


class SuiteSummaryTests(unittest.TestCase):
    def test_unit_response_and_common_start(self):
        spec=importlib.util.spec_from_file_location('suite',ROOT/'tools/prior_sensitivity_suite.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        variants={'reference':(1.,.5),'center_low':(.5,.5),'center_high':(2.,.5),'width_narrow':(1.,.25),'width_wide':(1.,1.)}
        base={'variants':{v:dict(prior_center_factor=f,prior_std=s) for v,(f,s) in variants.items()},'suite_runs':{},'checkpoint_iterations':[0,1],'optimizer':{'max_iterations':1}}
        with tempfile.TemporaryDirectory() as folder:
            module.ROOT=Path(folder)
            for i,(v,(factor,scale)) in enumerate(variants.items()):
                train=f'T{i}';ev=f'E{i}';base['suite_runs'][v]={'train':train,'evaluate':ev}
                artifact=Path(folder)/v;artifact.mkdir()
                for step in [0,1]:
                    q=LogStiffnessPosterior(torch.zeros(3,dtype=torch.float64),.5,.5)
                    if step:
                        with torch.no_grad():
                            q.mu.add_(torch.tensor(factor,dtype=torch.float64).log())
                            q.rho.fill_(torch.tensor(scale,dtype=torch.float64).expm1().log())
                    state={'posterior':q.state_dict(),'metadata':{'object_springs':2,'topology_sha256':'same'}}
                    torch.save(state,artifact/f'checkpoint-{step:06}.pt')
                torch.save(state,artifact/'checkpoint.pt')
                for rid,data in [(train,{'artifact_directory':str(artifact),'repository_commit':'test'}),(ev,{'results':[]})]:
                    p=Path(folder)/'runs'/rid;p.mkdir(parents=True)
                    (p/'manifest.json').write_text(json.dumps(data))
            module.summarize(base,Path(folder))
            result=json.loads((Path(folder)/'comparison.json').read_text())
            for row in result['paired_checkpoints']:
                for metric in ['center_response','width_response']:
                    self.assertAlmostEqual(row[metric]['mean'],row['iteration'])
