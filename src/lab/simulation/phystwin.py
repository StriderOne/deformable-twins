"""Forward-only PhysTwin adapter for the interactive tutorial (no training)."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import warp as wp

DEFAULT_EXPORT = Path.home() / '.cache/deformable-twins/phystwin-lesson/exports/double_stretch_sloth'
DEFAULT_SOURCE = Path(__file__).resolve().parents[3] / 'third_party/PhysTwin/qqtt/model/diff_simulator/spring_mass_warp.py'
KERNEL_SHA256 = '7deab9a25f4b8b8772f7df45c35571caf3767d014dd353cad151fe8eddceca1c'


def load_graph(folder):
    folder = Path(folder)
    provenance = json.loads((folder / 'provenance.json').read_text())
    if provenance['format'] != 'phystwin-newton-lesson-v1':
        raise ValueError('Unsupported export format')
    if [provenance[k] for k in ('length_unit', 'mass_unit', 'time_unit')] != ['m', 'kg', 's']:
        raise ValueError('Expected metres, kilograms and seconds')
    if hashlib.sha256((folder / 'graph.npz').read_bytes()).hexdigest() != provenance['graph_sha256']:
        raise ValueError('Graph checksum mismatch')
    with np.load(folder / 'graph.npz', allow_pickle=False) as archive:
        g = {k: archive[k].copy() for k in archive.files}
    kernel_hashes = [value for path, value in provenance.get('inputs', {}).items()
                     if path.endswith('/spring_mass_warp.py')]
    if kernel_hashes != [KERNEL_SHA256]:
        raise ValueError('Export must identify the matching PhysTwin kernel source')
    n = int(g['n_object'])
    e = g['edges']
    if not 0 < n < len(g['q']) or g['q'].shape[1:] != (3,) or g['v'].shape != g['q'].shape:
        raise ValueError('Expected dynamic particles and controller particles')
    if e.ndim != 2 or e.shape[1] != 2 or e.dtype.kind not in 'iu' or np.any(e < 0) or np.any(e >= len(g['q'])):
        raise ValueError('Invalid spring indices')
    if g['rest'].shape != (len(e),) or g['log_y'].shape != (len(e),) or np.any(g['rest'] <= 0):
        raise ValueError('Invalid spring parameters')
    if any(not np.isfinite(v).all() for v in g.values()):
        raise ValueError('Non-finite graph values')
    if g['mass'].shape != (len(g['q']),) or np.any(g['mass'][:n] <= 0) or np.any(g['mass'][n:] != 0):
        raise ValueError('Invalid dynamic/controller masses')
    if float(g['dt']) <= 0 or int(g['substeps']) <= 0 or float(g['y_min']) > float(g['y_max']):
        raise ValueError('Invalid integration/stiffness settings')
    if int(g['object_collision_enabled']):
        raise ValueError('This example does not support object/self collisions')
    if not (np.array_equal(g['gravity'], [0, 0, 9.8]) or np.array_equal(g['gravity'], [0, 0, -9.8])):
        raise ValueError('Expected original PhysTwin Z-axis gravity')
    np.testing.assert_allclose(g['q'][n:], g['controls'][0])
    return g, provenance


def load_kernels(source=DEFAULT_SOURCE):
    source = Path(source)
    text = source.read_text()
    if hashlib.sha256(source.read_bytes()).hexdigest() != KERNEL_SHA256:
        raise ValueError('PhysTwin kernels changed; review adapter before using this revision')
    names = {'eval_springs', 'update_vel_from_force', 'integrate_ground_collision', 'set_control_points'}
    lines = text.splitlines(keepends=True)
    functions = []
    for node in ast.parse(text).body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
            functions.append(''.join(lines[start:node.end_lineno]))
    if len(functions) != len(names):
        raise ValueError('Missing upstream kernels')
    # Extract unchanged functions to a real file so Warp can inspect their source.
    cache = Path.home() / '.cache/deformable-twins/phystwin-interactive'
    cache.mkdir(parents=True, exist_ok=True)
    name = 'phystwin_kernels_' + KERNEL_SHA256[:16]
    if name in sys.modules:
        return sys.modules[name]
    path = cache / (name + '.py')
    path.write_text('import warp as wp\n\n' + '\n\n'.join(functions))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    wp.set_module_options({'enable_backward': False}, module=module)
    return module



class PhysTwinInternalForces:
    """Unchanged PhysTwin spring kernel, bound to Newton's particle arrays.

    Object particles occupy a contiguous range starting at particle_offset.
    Only internal springs are retained. State and model masses live in Newton.
    """
    def __init__(self, graph, model, mass_scale=1.0, particle_offset=0, source=DEFAULT_SOURCE):
        if not np.isfinite(mass_scale) or mass_scale <= 0:
            raise ValueError('mass_scale must be positive and finite')
        n = int(graph['n_object'])
        if particle_offset < 0 or particle_offset + n > model.particle_count:
            raise ValueError('Object particle range does not fit the Newton model')
        mask = (graph['edges'] < n).all(axis=1)
        self.count = int(mask.sum())
        self.kernels = load_kernels(source)
        self.model = model
        self.edges = wp.array(graph['edges'][mask] + particle_offset, dtype=wp.vec2i, device=model.device)
        self.rest = wp.array(graph['rest'][mask], dtype=float, device=model.device)
        self.log_y = wp.array(graph['log_y'][mask] + np.log(mass_scale), dtype=float, device=model.device)
        self.empty = wp.empty(0, dtype=wp.vec3, device=model.device)
        self.damping = float(graph['damping']) * mass_scale
        self.y_min = float(graph['y_min']) * mass_scale
        self.y_max = float(graph['y_max']) * mass_scale

    def accumulate(self, model, state, dt):
        if model is not self.model:
            raise ValueError('Force law is bound to a different Newton model')
        if self.count:
            wp.launch(self.kernels.eval_springs, self.count,
                      [state.particle_q, state.particle_qd, self.empty, self.empty,
                       model.particle_count, self.edges, self.rest, self.log_y,
                       self.damping, self.y_min, self.y_max, state.particle_f], device=model.device)

    def reset(self):
        pass  # Elastic springs have no private history.
