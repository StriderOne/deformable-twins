"""Run the validated E-0006 pilot and its evaluation in a detached process."""
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

from lab.calibration.phystwin_adapter import ROOT, sha256
from lab.calibration.phystwin_vi import read_plan, write_json


def main():
    config=ROOT/'experiments/E-0006.yaml'
    plan,_=read_plan(config)
    root=Path(plan['execution']['artifact_root']).expanduser()
    validation=json.loads((root/'E-0006-validation-v2/validation.json').read_text())
    assert validation['status']=='passed' and validation['config_sha256']==sha256(config)
    for source in ['src/lab/calibration/posterior.py','src/lab/calibration/phystwin_adapter.py','src/lab/calibration/phystwin_vi.py']:
        assert sha256(ROOT/source)==validation['source_hashes'][source],source
    assert json.loads((root/'E-0006-validation-v2/evaluation-smoke/validation.json').read_text())['status']=='completed'
    output=root/'E-0006-launch'
    ids=plan['run_ids']
    for rid in ids.values():
        if (ROOT/'runs'/rid).exists() or (root/rid).exists():raise FileExistsError(rid)
    status={'status':'running','started_at_unix':time.time(),'completed':[]}
    try:
        for mode,key in [('vi','train'),('evaluate','evaluate')]:
            status['active']=ids[key];write_json(output/'status.json',status)
            cmd=[sys.executable,'-m','lab.calibration.phystwin_vi','--config',str(config),'--mode',mode,'--run-dir',str(ROOT/'runs'/ids[key])]
            if mode=='evaluate':cmd+=['--vi-checkpoint',str(root/ids['train']/'checkpoint.pt')]
            subprocess.run(cmd,cwd=ROOT,check=True)
            manifest=json.loads((ROOT/'runs'/ids[key]/'manifest.json').read_text())
            if manifest['status']!='completed':raise RuntimeError(manifest['status'])
            status['completed'].append(ids[key])
        status['status']='completed'
    except BaseException:
        status.update(status='failed',error=traceback.format_exc());raise
    finally:
        status['ended_at_unix']=time.time();write_json(output/'status.json',status)


if __name__=='__main__':main()
