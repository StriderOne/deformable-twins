"""PhysTwin + Newton demo in Viser, Newton OpenGL, or Isaac Sim RTX."""
import argparse
from pathlib import Path
import time
import threading
import json
from dataclasses import replace

import newton
import numpy as np
import viser
import viser.transforms as transforms
from plyfile import PlyData


DEFAULT_PLY = (
    Path.home() / 'PhysTwin/gaussian_output/double_stretch_sloth'
    / 'init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0'
    / 'point_cloud/iteration_10000/point_cloud.ply'
)


def main(isaac_app=None, parsed_args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('ply', nargs='?', type=Path, default=DEFAULT_PLY)
    parser.add_argument('--accurate', action='store_true', help='Original single-rate solver for comparison')
    parser.add_argument('--viewer', choices=['viser', 'gl', 'isaac'], default='viser')
    parser.add_argument('--headless', action='store_true', help='Offscreen window-viewer validation')
    parser.add_argument('--frames', type=int, help='Stop window viewer after this many rendered frames')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--seed', type=int, default=0, help='Random grasp target seed')
    parser.add_argument('--sphere', action='store_true', help='Use the previous sphere demo instead of Panda')
    parser.add_argument('--reference', action='store_true', help='Original PhysTwin controller-point demo')
    parser.add_argument('--object-mass', type=float, help='Demonstration total mass in kg (default: DemoSettings)')
    parser.add_argument('--static', action='store_true', help='Display the original static asset only')
    parser.add_argument('--graph', type=Path, help='Folder with graph.npz and provenance.json')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-gaussians', type=int, default=None, help='Rendering budget (0: all; default: all in Isaac, 40000 otherwise); physics unchanged')
    args = parsed_args or parser.parse_args()
    if args.viewer in ('gl', 'isaac') and (args.static or args.reference or args.sphere):
        parser.error('Window viewers currently support the automatic Panda demo')
    if args.frames is not None and args.frames < 1:
        parser.error('--frames must be positive')
    if (args.headless or args.frames is not None) and args.viewer == 'viser':
        parser.error('--headless and --frames require --viewer gl or isaac')
    if args.headless and args.frames is None:
        parser.error('--headless requires --frames')
    if args.max_gaussians is None:
        args.max_gaussians = 0 if args.viewer == 'isaac' else 40000
    if args.max_gaussians < 0:
        parser.error('--max-gaussians must be nonnegative')
    if not args.ply.is_file():
        parser.error(f'PLY not found: {args.ply}. Pass the path to a PhysTwin Gaussian point_cloud.ply.')

    if args.viewer == 'isaac' and isaac_app is None:
        from view_phystwin_isaac import launch
        launch(args)
        return

    vertex = PlyData.read(args.ply)['vertex'].data
    def columns(names):
        return np.column_stack([vertex[name] for name in names]).astype(np.float32)

    log_scales = columns(['scale_0'])
    if 'scale_1' in vertex.dtype.names or 'scale_2' in vertex.dtype.names:
        log_scales = columns(['scale_0', 'scale_1', 'scale_2'])
    else:
        log_scales = np.repeat(log_scales, 3, axis=1)
    rotations_xyzw = columns(['rot_1', 'rot_2', 'rot_3', 'rot_0'])
    norms = np.linalg.norm(rotations_xyzw, axis=1, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError('Zero-length Gaussian quaternion')
    gaussian = newton.Gaussian(
        positions=columns(['x', 'y', 'z']), rotations=rotations_xyzw / norms,
        scales=np.exp(log_scales),
        opacities=np.exp(-np.logaddexp(0., -vertex['opacity'])),
        sh_coeffs=columns(['f_dc_0', 'f_dc_1', 'f_dc_2']), sh_degree=0,
    )
    centers = np.asarray(gaussian.positions, dtype=np.float32)
    scales = np.asarray(gaussian.scales, dtype=np.float32)
    rotations = transforms.SO3(np.roll(gaussian.rotations, 1, axis=1)).as_matrix()
    covariances = np.einsum('nij,nj,nkj->nik', rotations, scales**2, rotations).astype(np.float32)
    # Viser accepts RGB, so use the degree-zero SH color, as Newton's ViewerViser does.
    rgbs = np.clip(0.28209479177387814 * gaussian.sh_coeffs[:, :3] + 0.5, 0, 1)
    opacities = np.asarray(gaussian.opacities, dtype=np.float32).reshape(-1, 1)
    if len(centers) == 0 or not all(np.isfinite(a).all() for a in (centers, covariances, rgbs, opacities)):
        raise ValueError('Empty or non-finite Gaussian asset')

    simulation = None
    newton_mode = not args.static and not args.reference
    robot_mode = newton_mode and not args.sphere
    run_directory = None
    if not args.static:
        from phystwin_runtime import DEFAULT_EXPORT, PhysTwinSimulation, GaussianBinding, load_graph
        from scipy.cluster.vq import kmeans2
        graph_path = args.graph or DEFAULT_EXPORT
        if not (graph_path / 'graph.npz').is_file():
            parser.error(f'Export missing: {graph_path}. See environments/README.md, or use --static.')
        graph, provenance = load_graph(graph_path)
        if not np.allclose(scales, scales[:, :1], rtol=1e-4, atol=1e-8):
            parser.error('Interactive example currently supports isotropic PhysTwin Gaussians only')
        selected = (np.arange(len(centers)) if args.max_gaussians == 0
                    else np.flatnonzero(opacities[:, 0] > 0.1))
        if len(selected) == 0:
            parser.error('No visible Gaussians above opacity 0.1')
        if args.max_gaussians and len(selected) > args.max_gaussians:
            selected = selected[np.linspace(0, len(selected) - 1, args.max_gaussians, dtype=int)]
        centers, covariances, rgbs, opacities = [a[selected] for a in (centers, covariances, rgbs, opacities)]
        rest = graph['q'][:int(graph['n_object'])]
        binding = GaussianBinding(rest, centers)
        nearest_distance = np.linalg.norm(centers - rest[binding.ids[:, 0]], axis=1)
        if np.quantile(nearest_distance, 0.95) > 0.05:
            parser.error('Gaussian PLY and physics graph appear misaligned (>5 cm for 95th percentile)')
        if newton_mode:
            from lab.simulation.demo import NewtonDemo, DemoSettings
            settings = DemoSettings()
            if args.object_mass is not None:
                settings = replace(settings, object_mass=args.object_mass)
            simulation = NewtonDemo(graph, device=args.device, settings=settings, robot=robot_mode, fast=robot_mode and not args.accurate)
            centers = centers @ simulation.rotation.T + simulation.translation
            covariances = simulation.rotation @ covariances @ simulation.rotation.T
            rest = simulation.rest
            binding = GaussianBinding(rest, centers)
            group_centers = [] if robot_mode else np.array([simulation.sphere_initial])
            run_directory = simulation.write_manifest(provenance, args.ply, seed=args.seed)
            print(f'Run metadata: {run_directory}', flush=True)
            print(f'Newton: {simulation.n} particles, {simulation.law.count} internal springs, '
                  f'mass={settings.object_mass} kg (demonstration assumption)', flush=True)
        else:
            simulation = PhysTwinSimulation(graph, device=args.device)
            group_centers, groups = kmeans2(simulation.initial_controls, 2, minit='++', seed=0)
            if len(np.unique(groups)) != 2:
                parser.error('Expected two non-empty controller groups')
        print(f'dt={simulation.dt}, substeps={simulation.substeps}', flush=True)

    if args.viewer == 'gl':
        from view_phystwin_gl import run_gl
        run_gl(simulation, args, run_directory, centers, covariances, rgbs, binding)
        return

    if args.viewer == 'isaac':
        from view_phystwin_isaac import run_isaac
        run_isaac(simulation, args, run_directory, centers, covariances, rgbs, opacities, binding, isaac_app)
        return

    if newton_mode:
        from lab.simulation.skinning import GpuGaussianBinding
        gpu_binding = GpuGaussianBinding(binding, simulation.model.device)

    server = viser.ViserServer(host='127.0.0.1', port=args.port, label='PhysTwin: mass-spring interaction' if simulation else 'PhysTwin: static Gaussian splats')
    up = (0., 0., 1.) if newton_mode else (0., 0., -1.)
    server.scene.set_up_direction('+z' if newton_mode else '-z')
    if newton_mode:
        server.scene.add_grid('/ground', width=2., height=2., cell_size=.05, plane='xy')
        if robot_mode:
            from lab.simulation.panda import PandaVisual
            robot_visual = PandaVisual(server.scene, simulation)
        else:
            sphere = server.scene.add_icosphere('/sphere', radius=simulation.settings.sphere_radius,
                                               position=simulation.sphere_initial, color=(40, 150, 240))
    splats = server.scene.add_gaussian_splats('/object', centers=centers, covariances=covariances,
                                     rgbs=rgbs, opacities=opacities)
    # Robust framing avoids distant low-opacity splats dominating the camera bounds.
    visible = centers[opacities[:, 0] > 0.1]
    if len(visible) == 0:
        visible = centers
    low, high = np.quantile(visible, [0.02, 0.98], axis=0)
    center = (low + high) / 2
    radius = max(float(np.linalg.norm(high - low)), 0.1)

    if simulation:
        if robot_mode:
            from lab.simulation.grasp_cycle import GraspCycle
            def log_cycle(event):
                with (run_directory/'events.jsonl').open('a') as log:
                    log.write(json.dumps(dict(event, wall_time=time.time()))+'\n')
            cycle = GraspCycle(simulation, seed=args.seed, log=log_cycle)
            server.gui.add_markdown('**Автоматический захват.** Случайная точка на верхней поверхности → '
                                    'подход сверху → закрытие пальцев → подъём → сброс. '
                                    'Захват только контактный: игрушка может выскользнуть.')
            target_marker = server.scene.add_icosphere('/grasp_target', radius=.006,
                                                       color=(240,60,40), visible=False)
        elif newton_mode:
            server.gui.add_markdown('**Тяни стрелки цели шара.** Шар движется от ограниченной силы, '
                                    'а объект — от контактов с ним.\n\n'
                                    'Отключи тягу, чтобы шар свободно реагировал на объект.')
            pulling = server.gui.add_checkbox('Тянуть шар', initial_value=True)
        else:
            server.gui.add_markdown('**Эталон PhysTwin: тяни стрелки двух захватов.**')
        paused = server.gui.add_checkbox('Пауза', initial_value=False)
        show_particles = server.gui.add_checkbox('Показать физические частицы', initial_value=False)
        reset_button = server.gui.add_button('Сбросить')
        status = server.gui.add_markdown('Готово')
        reset_requested = threading.Event()
        @reset_button.on_click
        def request_reset(event):
            reset_requested.set()
        handles = []
        for i, position in enumerate(group_centers):
            handle = server.scene.add_transform_controls(
                f'/target_{i}', position=position, scale=0.10,
                disable_rotations=True, depth_test=False,
                translation_limits=tuple((float(v - .5), float(v + .5)) for v in position))
            handles.append(handle)
        particles = server.scene.add_point_cloud('/particles', points=rest,
                                                 colors=(255, 80, 30), point_size=.003, visible=False)
        q = rest.copy()
        if not newton_mode:
            applied_target = simulation.initial_controls.copy()

    @server.on_client_connect
    def set_camera(client):
        client.camera.up_direction = up
        if robot_mode:
            client.camera.position = center + np.array([.9, -1.3, .9])
            client.camera.look_at = center + np.array([-.2,0,.2])
            return
        client.camera.position = center + radius * np.array([0., -1.3, .8 if newton_mode else -.3])
        client.camera.look_at = center

    print(f'Loaded {gaussian.count:,} Gaussian splats; displaying {len(centers):,} from {args.ply}', flush=True)
    print(f'Open http://127.0.0.1:{server.get_port()} — Ctrl+C to stop.', flush=True)
    try:
        while True:
            start = time.monotonic()
            if simulation:
                if reset_requested.is_set():
                    reset_requested.clear()
                    simulation.reset()
                    q = rest.copy()
                    if not newton_mode:
                        applied_target = simulation.initial_controls.copy()
                    else:
                        if robot_mode:
                            robot_visual.update(simulation)
                            cycle.reset()
                            target_marker.visible = False
                        else:
                            sphere.position = simulation.sphere_position
                        with (run_directory/'events.jsonl').open('a') as log:
                            log.write(json.dumps(dict(event='reset', wall_time=time.time()))+'\n')
                    for handle, position in zip(handles, group_centers):
                        handle.position = position.copy()
                    splats.centers = centers
                    particles.points = q
                    paused.value = False
                    status.content = 'Сброшено'
                particles.visible = show_particles.value
                if not paused.value and server.get_clients():
                    try:
                        if newton_mode:
                            if robot_mode:
                                q = cycle.step()
                                target_marker.visible = cycle.target_point is not None
                                if cycle.target_point is not None:
                                    target_marker.position = cycle.target_point
                                robot_visual.update(simulation)
                            else:
                                q = simulation.step(np.asarray(handles[0].position), pull=pulling.value)
                                sphere.position = simulation.sphere_position
                        else:
                            desired = simulation.initial_controls.copy()
                            for i, handle in enumerate(handles):
                                desired[groups == i] += np.asarray(handle.position) - group_centers[i]
                            delta = desired - applied_target
                            delta *= np.minimum(1., .005 / np.maximum(np.linalg.norm(delta, axis=1, keepdims=True), 1e-12))
                            applied_target += delta
                            q = simulation.step(applied_target)
                        splats.centers = gpu_binding.deform(simulation.state.particle_q).numpy() if newton_mode else binding.deform(q)
                        if show_particles.value:
                            particles.points = q
                        phase = f'Цикл {cycle.number} · {cycle.phase} · ' if robot_mode else ''
                        status.content = phase + f'Время симуляции: {simulation.time:.2f} с · кадр: {time.monotonic() - start:.3f} с'
                    except (FloatingPointError, np.linalg.LinAlgError) as error:
                        paused.value = True
                        status.content = f'Ошибка: {error}. Нажми «Сбросить».'
                        print(f'Simulation paused: {error}', flush=True)
                        if run_directory:
                            stamp = time.time_ns()
                            np.savez_compressed(run_directory/f'failure-{stamp}.npz',
                                                **{k:getattr(simulation.state,k).numpy() for k in
                                                   ('particle_q','particle_qd','body_q','body_qd','joint_q','joint_qd')})
                            with (run_directory/'events.jsonl').open('a') as log:
                                log.write(json.dumps(dict(event='failure', reason=str(error), wall_time=time.time()))+'\n')
            time.sleep(max(0.001, .05 - (time.monotonic() - start)))
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
        if run_directory:
            (run_directory/'result.json').write_text(json.dumps(dict(status='stopped', simulated_time=simulation.time)))


if __name__ == '__main__':
    main()
