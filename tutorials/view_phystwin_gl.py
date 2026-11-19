"""Native Newton window for the shared contact-only Panda grasp demo."""
import json
import importlib.metadata
import time
import numpy as np
import warp as wp
from newton.viewer import ViewerGL
from lab.simulation.grasp_cycle import GraspCycle
from lab.simulation.skinning import GpuGaussianBinding


def run_gl(simulation, args, directory, centers, covariances, colors, binding):
    viewer = ViewerGL(width=1280, height=900, headless=args.headless, num_frames=args.frames)
    viewer.set_model(simulation.model)
    viewer.show_particles = False
    center = (simulation.rest.min(0)+simulation.rest.max(0))/2
    viewer.set_camera(wp.vec3(*(center+np.array([.9,-1.3,.9]))), pitch=-30., yaw=125.)
    skin = GpuGaussianBinding(binding, simulation.model.device)
    points = skin.output
    radii = wp.array(np.sqrt(np.trace(covariances,axis1=1,axis2=2)/3), dtype=float, device=simulation.model.device)
    rgb = wp.array(colors, dtype=wp.vec3, device=simulation.model.device)
    marker_color = wp.array([[1., .1, .1]], dtype=wp.vec3, device=simulation.model.device)
    marker = wp.zeros(1, dtype=wp.vec3, device=simulation.model.device)
    def log(event):
        with (directory/'events.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(event, wall_time=time.time()))+'\n')
    cycle = GraspCycle(simulation, seed=args.seed, log=log)
    error_message = None
    def reset():
        nonlocal error_message
        simulation.reset()
        cycle.reset()
        points.assign(centers)
        error_message = None
        log(dict(event='manual_reset'))
    viewer.set_reset_callback(reset)
    def panel(imgui):
        imgui.text(f'Grasp cycle {cycle.number}: {cycle.phase}')
        imgui.text_wrapped('Gaussian centers rendered as colored spheres')
        if error_message:
            imgui.text_wrapped(error_message)
    viewer.register_ui_callback(panel)
    log(dict(event='viewer', backend='gl', representation='colored Gaussian-center spheres',
             versions={p:importlib.metadata.version(p) for p in ['pyglet','imgui-bundle','PyOpenGL']}))
    frames = 0
    frame_times = []
    try:
        while viewer.is_running() and (args.frames is None or frames < args.frames):
            frame_start = time.perf_counter()
            if viewer.should_step() and error_message is None:
                try:
                    q = cycle.step()
                    skin.deform(simulation.state.particle_q)
                except (FloatingPointError, np.linalg.LinAlgError) as error:
                    error_message = f'{error}. Use Reset to restart.'
                    np.savez_compressed(directory/f'failure-{time.time_ns()}.npz',
                        **{k:getattr(simulation.state,k).numpy() for k in
                           ('particle_q','particle_qd','body_q','body_qd','joint_q','joint_qd')})
                    log(dict(event='failure', reason=str(error)))
                    print(error_message, flush=True)
            viewer.begin_frame(simulation.time)
            viewer.log_state(simulation.state)
            viewer.log_points('/toy_surface',points,radii=radii,colors=rgb)
            if cycle.target_point is not None:
                marker.assign(cycle.target_point[None])
            viewer.log_points('/grasp_target',marker,radii=.006,colors=marker_color,
                              hidden=cycle.target_point is None)
            viewer.end_frame()
            frame_times.append(time.perf_counter()-frame_start)
            frames += 1
        if args.frames:
            from PIL import Image
            Image.fromarray(viewer.get_frame(render_ui=True).numpy()).save(directory/'gl-frame.png')
    except KeyboardInterrupt:
        pass
    except Exception as error:
        error_message = str(error)
        log(dict(event='viewer_failure', reason=error_message))
        raise
    finally:
        viewer.close()
        (directory/'result.json').write_text(json.dumps(dict(status='failed' if error_message else 'stopped',
            failure_reason=error_message, viewer='gl', rendered_frames=frames, simulated_time=simulation.time,
            median_frame_ms=float(np.median(frame_times)*1000) if frame_times else None,
            p95_frame_ms=float(np.quantile(frame_times,.95)*1000) if frame_times else None,
            mean_fps=float(len(frame_times)/sum(frame_times)) if frame_times else None),indent=2)+'\n')
