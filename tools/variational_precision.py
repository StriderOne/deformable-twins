"""Export per-spring variational precision gain; this is not a likelihood FIM."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def gain(std, prior_std):
    return std ** -2 - prior_std ** -2


def main():
    np.testing.assert_allclose(gain(np.array([.05, .5, 1.]), .5), [396., 0., -3.])
    output = ROOT / 'runs/vi-precision-gain.json'
    if output.exists():
        raise FileExistsError(output)
    artifact = Path.home()/'.local/share/deformable-twins/artifacts/vi-precision-gain'
    artifact.mkdir(exist_ok=False)
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    result = {'definition': 'posterior_std**(-2) - prior_std**(-2)',
              'interpretation': 'variational_precision_gain_not_likelihood_FIM',
              'coordinates': 'log_stiffness_before_clamp',
              'analysis_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'analysis_script_sha256': digest(Path(__file__)), 'runs': {}}
    for run in ['R-0006', 'R-0007']:
        manifest = json.loads((ROOT/'runs'/run/'manifest.json').read_text())
        path = Path(manifest['artifact_directory'])/'checkpoint.pt'
        assert digest(path) == manifest['artifacts'][str(path)]
        state = torch.load(path, map_location='cpu', weights_only=False)
        p = state['posterior']
        std = torch.nn.functional.softplus(p['rho'].double()).numpy()
        prior = p['prior_std'].double().numpy()
        delta = gain(std, prior)
        assert np.isfinite(delta).all() and (std > 0).all() and (prior > 0).all()
        n = state['metadata']['object_springs']
        target = artifact/f'{run}.csv'
        with target.open('x') as f:
            writer = csv.writer(f)
            writer.writerow(['spring_id', 'group', 'posterior_std', 'prior_std', 'posterior_precision', 'prior_precision', 'precision_gain'])
            for i in range(len(std)):
                writer.writerow([i, 'object' if i < n else 'controller', std[i], prior[i], std[i]**-2, prior[i]**-2, delta[i]])
        stats = {}
        for name, x in [('object', delta[:n]), ('controller', delta[n:])]:
            stats[name] = dict(count=len(x), mean=float(x.mean()),
                               quantiles=dict(zip(['min','p05','median','p95','max'], map(float, np.quantile(x,[0,.05,.5,.95,1])))),
                               positive_fraction=float((x>0).mean()), negative_fraction=float((x<0).mean()))
        result['runs'][run] = dict(checkpoint_sha256=digest(path), execution_commit=manifest['repository_commit'],
                                  iteration=state['iteration'], csv=str(target), csv_sha256=digest(target), groups=stats)
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result['runs'], indent=2))


if __name__ == '__main__':
    main()
