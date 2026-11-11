"""Export a real PhysTwin checkpoint without importing its renderer or running physics.

Run in an environment with torch, numpy, open3d and PyYAML. Only load trusted
local pickle files. The original Config and graph-construction method are loaded
from source via AST to avoid importing Gaussian-splatting/CUDA rendering modules.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
from types import SimpleNamespace

import numpy as np
import open3d as o3d
import torch
import yaml


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def source_node(path, class_name, method=None):
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.ClassDef) and n.name == class_name)
    if method:
        node = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == method)
    node.decorator_list = []
    return compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True, help='Scene folder with final_data.pkl')
    parser.add_argument('--optimal', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}')
    root = args.source.resolve()
    commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != '2b6630528141b9cba5a7677c8b88b2129b4a8390':
        raise ValueError('Unsupported PhysTwin revision; review the adapter before upgrading')
    config_source = root / 'qqtt/utils/config.py'
    trainer_source = root / 'qqtt/engine/trainer_warp.py'
    physics_source = root / 'qqtt/model/diff_simulator/spring_mass_warp.py'
    data_source = root / 'qqtt/data/real_data.py'
    for path in (config_source, trainer_source, physics_source, data_source):
        original = subprocess.check_output(['git', '-C', str(root), 'show', f'HEAD:{path.relative_to(root)}'])
        if path.read_bytes() != original:
            raise ValueError(f'Core source modified: {path}')
    namespace = {'yaml': yaml}
    exec(source_node(config_source, 'Config'), namespace)
    cfg = namespace['Config']()
    cfg.load_from_yaml(args.config)
    with args.optimal.open('rb') as stream:
        cfg.set_optimal_params(pickle.load(stream))
    if cfg.data_type != 'real' or cfg.self_collision:
        raise ValueError('This adapter supports single-object real scenes without self-collision')
    cfg.device = 'cpu'
    with (args.data / 'final_data.pkl').open('rb') as stream:
        data = pickle.load(stream)
    points = np.concatenate([data['object_points'][0], data['surface_points'], data['interior_points']])
    controls = np.asarray(data['controller_points'], dtype=np.float32)
    namespace = {'np': np, 'o3d': o3d, 'torch': torch, 'cfg': cfg}
    exec(source_node(trainer_source, 'InvPhyTrainerWarp', '_init_start'), namespace)
    q, edges, rest, mass, num_object_springs = namespace['_init_start'](
        None, torch.tensor(points, dtype=torch.float32), torch.tensor(controls[0]),
        object_radius=cfg.object_radius, object_max_neighbours=cfg.object_max_neighbours,
        controller_radius=cfg.controller_radius, controller_max_neighbours=cfg.controller_max_neighbours)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    y = checkpoint['spring_Y'].detach().cpu()
    if len(y) != len(edges) or int(checkpoint['num_object_springs']) != num_object_springs:
        raise ValueError('Checkpoint topology does not match data/config/optimal parameters')
    if not torch.isfinite(y).all() or not (y > 0).all():
        raise ValueError('Expected finite positive trained spring_Y')
    graph = dict(q=q.numpy(), v=np.zeros_like(q.numpy()),
                 mass=np.r_[mass.numpy()[:len(points)], np.zeros(len(controls[0]), dtype=np.float32)],
                 n_object=len(points), edges=edges.numpy(), rest=rest.numpy(), log_y=torch.log(y).numpy(),
                 damping=cfg.dashpot_damping, y_min=cfg.spring_Y_min, y_max=cfg.spring_Y_max,
                 drag=cfg.drag_damping, gravity=[0., 0., 9.8 if cfg.reverse_z else -9.8],
                 dt=cfg.dt, controls=controls, substeps=cfg.num_substeps,
                 object_collision_enabled=0, collision_dist=0.)
    for name, key in [('restitution', 'collide_elas'), ('friction', 'collide_fric'),
                      ('object_restitution', 'collide_object_elas'), ('object_friction', 'collide_object_fric')]:
        graph[name] = checkpoint[key].detach().cpu().numpy()
    inputs = [config_source, trainer_source, physics_source, data_source, args.config,
              args.optimal, args.checkpoint, args.data / 'final_data.pkl']
    inputs += [p for p in (args.data / 'split.json', args.data / 'calibrate.pkl', args.data / 'metadata.json') if p.exists()]
    provenance = dict(format='phystwin-newton-lesson-v1', length_unit='m', mass_unit='kg', time_unit='s',
                      coordinate_frame='PhysTwin final_data coordinates, unchanged; ' + ('gravity along +Z' if cfg.reverse_z else 'gravity along -Z'),
                      unit_note='PhysTwin simulator conventions; masses are nominal model masses, not measured object mass',
                      source_commit=commit, source_dirty=subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain'], text=True),
                      inputs={str(p.resolve()): sha256(p) for p in inputs}, config=cfg.to_dict(),
                      exporter_sha256=sha256(Path(__file__)),
                      versions={k: v for k, v in [('numpy', np.__version__), ('torch', torch.__version__), ('open3d', o3d.__version__)]},
                      contact_mode='exported_only_not_implemented_in_lesson_rollout',
                      topology_note='Original _init_start; counts checked, checkpoint has no edge hash to prove original ordering')
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output / 'graph.npz', **graph)
    provenance['graph_sha256'] = sha256(args.output / 'graph.npz')
    (args.output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(dict(output=str(args.output), particles=len(q), objects=len(points), springs=len(edges), frames=len(controls))))


if __name__ == '__main__':
    main()
