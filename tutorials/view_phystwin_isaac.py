"""RTX presentation of the existing Newton simulation; USD carries no physics."""
import hashlib
import importlib.metadata
import json
import os
import shutil
import traceback
from pathlib import Path
import time

import numpy as np


def launch(args):
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': args.headless, 'width': 1280, 'height': 720,
                         'window_width': 1280, 'window_height': 720,
                         'renderer': 'RaytracedLighting', 'multi_gpu': False,
                         'fast_shutdown': True,
                         'extra_args': ['--/telemetry/enableAnonymousData=false',
                                        '--/UJITSO/datastore/localDataStore/largeChunkDiskBudgetMB=512',
                                        '--/UJITSO/datastore/localDataStore/largeChunkBucketSizeMB=64',
                                        '--/UJITSO/datastore/localDataStore/diskGarbageCollectSeconds=1',
                                        '--/privacy/usage=false', '--/privacy/performance=false',
                                        '--/privacy/personalization=false']})
    exit_code = 0
    try:
        import newton
        import warp as wp
        if (newton.__version__, wp.__version__) != ('1.6.1', '1.17.0'):
            raise RuntimeError('Isaac viewer requires the pinned Newton 1.6.1 / Warp 1.17.0 image')
        from view_phystwin_gaussians import main
        main(isaac_app=app, parsed_args=args)
    except Exception as error:
        exit_code = 1
        traceback.print_exc()
        raise
    finally:
        app.close(exit_code=exit_code)


def run_isaac(simulation, args, directory, centers, covariances, colors, opacities, binding, app):
    try:
        _run(app, simulation, args, directory, centers, covariances, colors, opacities, binding)
    except Exception as error:
        (directory/'isaac-error.txt').write_text(repr(error)+'\n')
        raise


def _run(app, simulation, args, directory, centers, covariances, colors, opacities, binding):
    import carb
    import newton
    import warp as wp
    import omni.usd
    import omni.ui as ui
    from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, Vt
    from lab.simulation.grasp_cycle import GraspCycle
    from lab.simulation.skinning import GpuGaussianBinding

    import omni.kit.app
    omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate(
        'omni.usd.schema.usd_particle_field', True)
    settings = carb.settings.get_settings()
    settings.set_bool('/rtx-transient/dlssg/enabled', False)
    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    world = UsdGeom.Xform.Define(stage, '/World')
    stage.SetDefaultPrim(world.GetPrim())

    def pose_ops(prim, pose):
        xf = UsdGeom.Xformable(prim)
        translate = xf.AddTranslateOp()
        orient = xf.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble)
        set_pose((translate, orient), pose)
        return translate, orient

    def set_pose(ops, pose):
        ops[0].Set(Gf.Vec3d(*map(float, pose[:3])))
        ops[1].Set(Gf.Quatd(float(pose[6]), Gf.Vec3d(*map(float, pose[3:6]))))

    model = simulation.model
    body_ops = [pose_ops(UsdGeom.Xform.Define(stage, f'/World/Panda/link_{i}'), pose)
                for i, pose in enumerate(simulation.state.body_q.numpy())]
    bodies, flags = model.shape_body.numpy(), model.shape_flags.numpy()
    local, scales = model.shape_transform.numpy(), model.shape_scale.numpy()
    mesh_colors = model.shape_color.numpy()
    for i in simulation.robot.shape_ids:
        if not flags[i] & int(newton.ShapeFlags.VISIBLE):
            continue
        source = model.shape_source[i]
        if not isinstance(source, newton.Mesh):
            raise ValueError(f'Unexpected Panda visual shape {i}')
        parent = f'/World/Panda/link_{bodies[i]}' if bodies[i] >= 0 else '/World/Panda/static'
        mesh = UsdGeom.Mesh.Define(stage, f'{parent}/shape_{i}')
        vertices = np.asarray(source.vertices, np.float32) * scales[i]
        indices = np.asarray(source.indices, np.int32).reshape(-1)
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(indices))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(indices)//3, 3, np.int32)))
        mesh.CreateSubdivisionSchemeAttr('none')
        mesh.CreateDisplayColorAttr([Gf.Vec3f(*map(float, mesh_colors[i]))])
        pose_ops(mesh, local[i])
    table = UsdGeom.Cube.Define(stage, '/World/Table')
    table.CreateSizeAttr(1.)
    table.AddTranslateOp().Set(Gf.Vec3d(0., 0., -.025))
    table.AddScaleOp().Set(Gf.Vec3f(3., 3., .05))
    table.CreateDisplayColorAttr([Gf.Vec3f(.35, .38, .42)])
    light = UsdLux.DomeLight.Define(stage, '/World/Light')
    light.CreateIntensityAttr(700.)
    marker = UsdGeom.Sphere.Define(stage, '/World/GraspTarget')
    marker.CreateRadiusAttr(.006)
    marker.CreateDisplayColorAttr([Gf.Vec3f(1., .1, .1)])
    marker_position = marker.AddTranslateOp()

    # The existing binding supports isotropic splats, whose rotation is immaterial.
    sigma = np.sqrt(np.trace(covariances, axis1=1, axis2=2)/3).astype(np.float32)
    # Isaac 6.0 ships this schema as a Kit extension, before its UsdVol Python wrapper.
    splat = stage.DefinePrim('/World/Toy', 'ParticleField3DGaussianSplat')
    if not splat.IsA(UsdGeom.Boundable):
        raise RuntimeError('Native Gaussian schema failed to register')
    def attribute(name, kind, value):
        attr = splat.CreateAttribute(name, kind, custom=False)
        attr.Set(value)
        return attr
    positions = attribute('positions', Sdf.ValueTypeNames.Point3fArray,
                          Vt.Vec3fArray.FromNumpy(centers.astype(np.float32)))
    attribute('scales', Sdf.ValueTypeNames.Float3Array,
              Vt.Vec3fArray.FromNumpy(np.repeat(sigma[:, None], 3, axis=1)))
    attribute('orientations', Sdf.ValueTypeNames.QuatfArray, Vt.QuatfArray([Gf.Quatf(1.)] * len(centers)))
    attribute('opacities', Sdf.ValueTypeNames.FloatArray,
              Vt.FloatArray.FromNumpy(opacities.reshape(-1).astype(np.float32)))
    attribute('radiance:sphericalHarmonicsDegree', Sdf.ValueTypeNames.Int, 0)
    sh = attribute('radiance:sphericalHarmonicsCoefficients', Sdf.ValueTypeNames.Float3Array,
        Vt.Vec3fArray.FromNumpy(((colors-.5)/.28209479177387814).astype(np.float32)))
    UsdGeom.Primvar(sh).SetElementSize(1)
    UsdGeom.Primvar(sh).SetInterpolation(UsdGeom.Tokens.vertex)
    extent = UsdGeom.Boundable(splat).CreateExtentAttr()
    assert len(positions.Get()) == len(centers)
    assert len(splat.GetAttribute('opacities').Get()) == len(centers)
    assert stage.GetPrimAtPath('/World/Panda/link_0').IsValid()

    center = (simulation.rest.min(0)+simulation.rest.max(0))/2
    camera = UsdGeom.Camera.Define(stage, '/World/Camera')
    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*map(float, center+[.9,-1.3,.9])),
                                 Gf.Vec3d(*map(float, center+[-.15,0.,.2])), Gf.Vec3d(0.,0.,1.))
    camera.AddTransformOp().Set(view.GetInverse())
    camera.CreateFocalLengthAttr(24.)
    camera.CreateClippingRangeAttr(Gf.Vec2f(.01, 100.))
    viewport = get_active_viewport()
    viewport.camera_path = '/World/Camera'
    viewport.set_texture_resolution((1280,720))
    skin = GpuGaussianBinding(binding, model.device)

    def log(event):
        with (directory/'events.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(event, wall_time=time.time()))+'\n')
    cycle = GraspCycle(simulation, seed=args.seed, log=log)
    paused, failure = False, None
    def pause():
        nonlocal paused
        paused = not paused
    def reset():
        nonlocal failure
        simulation.reset()
        cycle.reset()
        failure = None
        log(dict(event='manual_reset'))
    window = ui.Window('PhysTwin — Newton physics', width=340, height=130)
    with window.frame:
        with ui.VStack():
            status = ui.Label('Starting')
            ui.Button('Pause / Resume', clicked_fn=pause)
            ui.Button('Reset', clicked_fn=reset)
    settings.set_int('/rtx/post/aa/op', 4)  # Set after stage initialization, which restores defaults.
    manifest = json.loads((directory/'manifest.json').read_text())
    manifest['renderer'] = dict(backend='Isaac Sim RTX', gaussians=len(centers), sh_degree=0,
        antialiasing='DLAA', antialiasing_op=settings.get('/rtx/post/aa/op'),
        max_gaussians=args.max_gaussians, opacity_cutoff=0.0 if args.max_gaussians == 0 else 0.1,
        resolution=[1280,720], physics='external Newton; USD visuals only',
        derived_geometry_cache_budget_mb=512, derived_geometry_cache_bucket_mb=64,
        derived_geometry_cache_gc_seconds=1,
        image_id=os.environ.get('PHYSTWIN_CONTAINER_IMAGE_ID'),
        isaac_build=Path('/isaac-sim/VERSION').read_text().strip(), usd_version=Usd.GetVersion(),
        scipy_version=importlib.metadata.version('scipy'),
        image='nvcr.io/nvidia/isaac-sim@sha256:a9d06fb72cc2b39b2bf1cacc002e9f7cd1a5bbe3c3cfbbcf2e6de3a1d726d101')
    manifest['source_hashes']['tutorials/view_phystwin_isaac.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (directory/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    times = []
    try:
        while app.is_running() and (args.frames is None or len(times)<args.frames):
            start = time.perf_counter()
            if len(times) % 60 == 0 and shutil.disk_usage(directory).free < 10*2**30:
                raise RuntimeError('Stopped: less than 10 GiB free disk space remains')
            if not paused and failure is None:
                try:
                    with wp.ScopedDevice(model.device):
                        cycle.step()
                except (FloatingPointError, np.linalg.LinAlgError) as error:
                    failure = str(error)
                    log(dict(event='failure', reason=failure))
                    np.savez_compressed(directory/f'failure-{time.time_ns()}.npz',
                        particle_q=simulation.state.particle_q.numpy(), body_q=simulation.state.body_q.numpy())
            with wp.ScopedDevice(model.device):
                skin.deform(simulation.state.particle_q)
                points = skin.output.numpy()
            positions.Set(Vt.Vec3fArray.FromNumpy(points))
            extent.Set(Vt.Vec3fArray.FromNumpy(np.array([(points-3*sigma[:,None]).min(0),
                                                       (points+3*sigma[:,None]).max(0)],np.float32)))
            for ops, pose in zip(body_ops, simulation.state.body_q.numpy()):
                set_pose(ops, pose)
            if cycle.target_point is not None:
                marker_position.Set(Gf.Vec3d(*map(float, cycle.target_point)))
                marker.MakeVisible()
            else:
                marker.MakeInvisible()
            status.text = failure or f'{cycle.phase} | t={simulation.time:.2f}s'
            app.update()
            times.append(time.perf_counter()-start)
        if args.frames:
            capture_viewport_to_file(viewport, str(directory/'isaac-frame.png'))
            for _ in range(20):
                app.update()
    except KeyboardInterrupt:
        pass
    except Exception as error:
        failure = str(error)
        raise
    finally:
        (directory/'result.json').write_text(json.dumps(dict(viewer='isaac',
            status='failed' if failure else 'stopped', failure_reason=failure,
            frames=len(times), simulated_time=simulation.time,
            mean_fps=len(times)/sum(times) if times else None,
            median_frame_ms=float(np.median(times)*1000) if times else None,
            p95_frame_ms=float(np.quantile(times,.95)*1000) if times else None),indent=2)+'\n')
