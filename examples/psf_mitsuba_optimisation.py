# #############################################################################
# caustics.py
# =================
# Author :
# Julien SAHLI [julien.sahli@epfl.ch]
# =================
# Based on the caustics optimisation script from mitsuba 3 :
# https://mitsuba.readthedocs.io/en/latest/src/inverse_rendering/caustics_optimization.html
# renders a scene illuminated by a spot light of arbitrary depth rather than directional light, to generate PSFs
#
# pip install mitsuba to install
# create a folder named meshes in the same directory as this script, and put the scene meshes in it
# download : https://rgl.s3.eu-central-1.amazonaws.com/scenes/tutorials/scenes.zip
# copy the meshes rectangle.obj and slab.obj from scenes/meshes
# #############################################################################

from os.path import realpath, join
import time
import os
import drjit as dr
import mitsuba as mi
import math

# main function that runs the caustics optimisation
# all dimensions provided are unitless.
# e.g if we have a scene with a 1x1 mm lens and a light source at 4 cm we need to input lens_size=1 and lens_to_scene=40
def run(ref_path, sensor_size=2, lens_size=1, sensor_to_lens=0.5, lens_to_scene=5, refractive_index=1.5, iterations=20,
        optimise=True, heightmap_path=None):

    mi_set_variants = ['llvm_ad_rgb', 'cuda_ad_rgb']
    mi.set_variant(mi_set_variants[0])

    # Set the target image as well as the optimization parameters
    config = make_config(
        image_path=ref_path,
        scene_path='meshes',
        output_path='outputs/' + ref_path.split('/')[-1].split('.')[0],
        max_iterations=iterations
    )

    # create the scene
    scene = make_scene(
        config=config,
        sensor_size=sensor_size,
        lens_size=lens_size,
        sensor_to_lens=sensor_to_lens,
        lens_to_scene=lens_to_scene,
        refractive_index=refractive_index,
        flat_shading=False  # does not works yet, because of the normals
    )

    if optimise:
        render(scene, config)
    else:
        assert heightmap_path is not None, "must specify a heightmap path when not optimizing"
        render(scene, config, heightmap_path)

    # possible improvements :
    #  - use flat shading, to represent actual glass slabs that would not be smooth but have facets
    #  - apply a grayscale texture to the slab, and give a refractive index identical to air makes an occluding mask


def make_config(image_path, scene_path, output_path, max_iterations,
                heightmap_resolution=512, samples_per_pixel=32, n_upsampling_steps=4, learning_rate=3e-5):

    scene_dir = realpath(scene_path)
    output_dir = realpath(join('.', output_path))

    config = {
        'n_upsampling_steps': n_upsampling_steps,
        'learning_rate': learning_rate,
        'spp': samples_per_pixel,
        'max_iterations': max_iterations,
        'scene_dir': scene_dir,
        'output_dir': output_dir,
        'reference':  image_path,
        'lens_fname': join(output_dir, 'lens_flat.ply'),
        'heightmap_res': (heightmap_resolution, heightmap_resolution), # we need to change this to use different width/height
    }

    return config


def make_scene(config, refractive_index, sensor_size, lens_size, sensor_to_lens, lens_to_scene,
               flat_shading=True, samples_per_pass=256, render_size=128):

    # If we want to specify different width/height for sensor / lens, we shoud take the bigger for calculating the angle
    spot_angle = (2.0 * math.atan2(lens_size * math.sqrt(0.5), lens_to_scene)) * 180.0 / math.pi
    cam_angle = (2.0 * math.atan2(sensor_size, sensor_to_lens)) * 180.0 / math.pi

    integrator = {
        'type': 'ptracer',
        'samples_per_pass': samples_per_pass,
        'max_depth': 4,
        'hide_emitters': False,
    }

    # Looking at the receiving plane, not looking through the lens
    cam_to_world = mi.ScalarTransform4f.look_at(
        origin=[0, -0.5 * sensor_to_lens, 0],
        target=[0, -sensor_to_lens, 0],
        up=[0, 0, 1]
    )

    cam = {
        'type': 'perspective',
        'near_clip': 0.45 * sensor_to_lens,
        'far_clip': 0.55 * sensor_to_lens + sensor_size*math.sqrt(2),
        'fov': cam_angle,
        'to_world': cam_to_world,

        'sampler': {
            'type': 'independent',
            'sample_count': samples_per_pass  # Not really used
        },
        'film': {
            'type': 'hdrfilm',
            'width': render_size,
            'height': render_size,
            'pixel_format': 'rgb',
            'rfilter': {
                # Important: smooth reconstruction filter with a footprint larger than 1 pixel.
                'type': 'gaussian'
            }
        },
    }

    scene = {
        'type': 'scene',
        'sensor': cam,
        'integrator': integrator,

        'lens-bsdf': {
            'type': 'dielectric',
            'id': 'simple-glass-bsdf',
            'ext_ior': 'air',
            'int_ior': refractive_index,
            'specular_reflectance': {'type': 'spectrum', 'value': 0},
        },
        'white-bsdf': {
            'type': 'diffuse',
            'id': 'white-bsdf',
            'reflectance': {'type': 'rgb', 'value': (1, 1, 1)},
        },
        'black-bsdf': {
            'type': 'diffuse',
            'id': 'black-bsdf',
            'reflectance': {'type': 'rgb', 'value': (0, 0, 0)},
        },
        # Receiving plane
        'receiving-plane': {
            'type': 'obj',
            'id': 'receiving-plane',
            'filename': 'meshes/rectangle.obj',
            'to_world': \
                mi.ScalarTransform4f.look_at(
                    origin=[0, -sensor_to_lens, 0],
                    target=[0, 0, 0],
                    up=[0, 0, 1]
                ).scale((sensor_size, sensor_size, sensor_size)),
            'bsdf': {'type': 'ref', 'id': 'white-bsdf'},
        },

        # Glass slab, excluding the 'exit' face (added separately below)
        'slab': {
            'type': 'obj',
            'id': 'slab',
            'filename': 'meshes/slab.obj',
            'to_world': mi.ScalarTransform4f.rotate(axis=(1, 0, 0), angle=90).scale((lens_size, lens_size, lens_size)),
            'bsdf': {'type': 'ref', 'id': 'lens-bsdf'},
        },
        # Glass rectangle, to be optimized
        'lens': {
            'type': 'ply',
            'id': 'lens',
            'filename': config['lens_fname'],
            'face_normals': flat_shading,
            'to_world': mi.ScalarTransform4f.rotate(axis=(1, 0, 0), angle=90).scale((lens_size, lens_size, lens_size)),
            'bsdf': {'type': 'ref', 'id': 'lens-bsdf'},
        },

        # Border to occlude light
        # Receiving plane
        'border_top': {
            'type': 'obj',
            'id': 'border_top',
            'filename': 'meshes/rectangle.obj',
            'to_world': \
                mi.ScalarTransform4f.look_at(
                    origin=[2 * lens_size, 0, 0],
                    target=[0, 1, 0],
                    up=[0, 0, 1]
                ).scale((lens_size, lens_size, lens_size)),
            'bsdf': {'type': 'ref', 'id': 'black-bsdf'},
        },
        'border_bottom': {
            'type': 'obj',
            'id': 'border_bottom',
            'filename': 'meshes/rectangle.obj',
            'to_world': \
                mi.ScalarTransform4f.look_at(
                    origin=[-2 * lens_size, 0, 0],
                    target=[0, 1, 0],
                    up=[0, 0, 1]
                ).scale((lens_size, lens_size, lens_size)),
            'bsdf': {'type': 'ref', 'id': 'black-bsdf'},
        },
        'border_left': {
            'type': 'obj',
            'id': 'border_left',
            'filename': 'meshes/rectangle.obj',
            'to_world': \
                mi.ScalarTransform4f.look_at(
                    origin=[0, 0, 2 * lens_size],
                    target=[0, 1, 0],
                    up=[0, 0, 1]
                ).scale((lens_size, lens_size, lens_size)),
            'bsdf': {'type': 'ref', 'id': 'black-bsdf'},
        },
        'border_right': {
            'type': 'obj',
            'id': 'border_right',
            'filename': 'meshes/rectangle.obj',
            'to_world': \
                mi.ScalarTransform4f.look_at(
                    origin=[0, 0, 2 * lens_size],
                    target=[0, 1, 0],
                    up=[0, 0, 1]
                ).scale((lens_size, lens_size, lens_size)),
            'bsdf': {'type': 'ref', 'id': 'black-bsdf'},
        },

        'emitter': {
            'type': 'spot',
            'cutoff_angle': spot_angle,
            'beam_width': spot_angle,
            'to_world': mi.ScalarTransform4f.look_at(
                origin=[0, lens_to_scene, 0],
                target=[0, 0, 0],
                up=[0, 0, 1]
            ),
            'intensity': {
                'type': 'spectrum',
                'value': 0.8
            },
        }
    }

    return scene


def render(scene, config, heightmap_path=None):

    optimization = heightmap_path is None  # when we provide an existing heightmap, we do not optimize it, only render

    os.makedirs(config['output_dir'], exist_ok=True)
    mi.Thread.thread().file_resolver().append(config['scene_dir'])

    # Create lens mesh to optimize/render
    create_flat_lens_mesh(config['heightmap_res']).write_ply(config['lens_fname'])

    scene = mi.load_dict(scene)

    if optimization:
        # create zero heightmap whose size matches the ref
        image_ref = load_ref_image(config, scene.sensors()[0].film().crop_size(), output_dir=config['output_dir'])
        height_bitmap = mi.Bitmap(dr.zeros(mi.TensorXf, [r // (2 ** config['n_upsampling_steps']) for r in config['heightmap_res']]))

    else:
        height_bitmap = mi.Bitmap(heightmap_path)

    heightmap_texture = mi.load_dict({
        'type': 'bitmap',
        'id': 'heightmap_texture',
        'bitmap': height_bitmap,
        'raw': True,
    })

    params = mi.traverse(heightmap_texture)
    params.keep(['data'])
    params_scene = mi.traverse(scene)
    spp = config['spp']

    positions_initial = dr.unravel(mi.Vector3f, params_scene['lens.vertex_positions'])
    normals_initial = dr.unravel(mi.Vector3f, params_scene['lens.vertex_normals'])
    lens_si = dr.zeros(mi.SurfaceInteraction3f, dr.width(positions_initial))
    lens_si.uv = dr.unravel(type(lens_si.uv), params_scene['lens.vertex_texcoords'])
    # print("vertex :", dr.shape(params_scene['lens.vertex_normals']))
    # print("face :", dr.shape(params_scene['lens.face_normals']))

    def apply_displacement(amplitude=1.):
        params['data'] = dr.clamp(params['data'], -0.01, 0.01)
        dr.enable_grad(params['data'])
        height_values = heightmap_texture.eval_1(lens_si)
        new_positions = (height_values * normals_initial * amplitude + positions_initial)
        params_scene['lens.vertex_positions'] = dr.ravel(new_positions)
        params_scene.update()

    # begin optimization
    if optimization:

        opt = mi.ad.Adam(lr=config['learning_rate'], params=params)
        iterations = config['max_iterations']

        rendering_steps = (int(0.7 * iterations), int(0.9 * iterations))
        print('The rendering quality will be increased at iterations:', rendering_steps)

        upsampling_steps = dr.linspace(mi.Float, 0, 1, config['n_upsampling_steps'] + 1, endpoint=False).numpy()[1:]
        upsampling_steps = (iterations * dr.sqr(upsampling_steps)).astype(int)
        print('The resolution of the heightfield will be doubled at iterations:', upsampling_steps)

        for it in range(iterations):
            t0 = time.time()

            # Gradient descent
            apply_displacement()
            loss = scale_independent_loss(mi.render(scene, params, seed=it, spp=2 * spp, spp_grad=spp), image_ref)
            dr.backward(loss)
            opt.step()

            # Increase resolution of the heightmap at given iterations
            if it in upsampling_steps:
                opt['data'] = dr.upsample(opt['data'], scale_factor=(2, 2, 1))

            # Increase rendering quality toward the end of the optimization
            if it in rendering_steps:
                spp *= 2
                opt.set_learning_rate(0.5 * opt.lr['data'])

            # update heightmap values
            params.update(opt)

            print("[Iteration %03i] Loss %.4f (%.2f ms)" % (it, loss[0], 1000. * (time.time() - t0)))

    # finally render the scene, either with the optimized heightmap or the provided one
    apply_displacement()

    # save the resulting psf
    save_bitmap(config, "result", mi.render(scene, params, spp=2 * spp, spp_grad=spp))

    # save the final heightmap
    save_bitmap(config, "heightmap_final", params['data'])  # also save the final heightmap

    # save the final heightmap mesh
    [m for m in scene.shapes() if m.id() == 'lens'][0].write_ply(join(config['output_dir'], 'lens_displaced.ply'))


def save_bitmap(config, name, data):
    fname = join(config['output_dir'], name + '.exr')
    mi.util.write_bitmap(fname, data)
    print('[+] Saved', name, 'to:', fname)

#check that is is well-behaved if we specify different resolutions for width and height
def create_flat_lens_mesh(resolution):
    # Generate UV coordinates
    U, V = dr.meshgrid(
        dr.linspace(mi.Float, 0, 1, resolution[0]),
        dr.linspace(mi.Float, 0, 1, resolution[1]),
        indexing='ij'
    )
    texcoords = mi.Vector2f(U, V)

    # Generate vertex coordinates
    X = 2.0 * (U - 0.5)
    Y = 2.0 * (V - 0.5)
    vertices = mi.Vector3f(X, Y, 0.0)

    # Create two triangles per grid cell
    faces_x, faces_y, faces_z = [], [], []
    for i in range(resolution[0] - 1):
        for j in range(resolution[1] - 1):
            v00 = i * resolution[1] + j
            v01 = v00 + 1
            v10 = (i + 1) * resolution[1] + j
            v11 = v10 + 1
            faces_x.extend([v00, v01])
            faces_y.extend([v10, v10])
            faces_z.extend([v01, v11])

    # Assemble face buffer
    faces = mi.Vector3u(faces_x, faces_y, faces_z)

    # Instantiate the mesh object
    mesh = mi.Mesh("lens-mesh", resolution[0] * resolution[1], len(faces_x), has_vertex_texcoords=True)

    # Set its buffers
    mesh_params = mi.traverse(mesh)
    mesh_params['vertex_positions'] = dr.ravel(vertices)
    mesh_params['vertex_texcoords'] = dr.ravel(texcoords)
    mesh_params['faces'] = dr.ravel(faces)
    mesh_params.update()

    return mesh


def load_ref_image(config, resolution, output_dir):
    b = mi.Bitmap(config['reference'])
    b = b.convert(mi.Bitmap.PixelFormat.RGB, mi.Bitmap.Float32, False)
    if b.size() != resolution:
        b = b.resample(resolution)

    mi.util.write_bitmap(join(output_dir, 'reference.exr'), b)

    print('[i] Loaded reference image from:', config['reference'])
    return mi.TensorXf(b)


def scale_independent_loss(image, ref):
    """Brightness-independent L2 loss function."""
    scaled_image = image / dr.mean(dr.detach(image))
    scaled_ref = ref / dr.mean(ref)
    return dr.mean(dr.sqr(scaled_image - scaled_ref))


if __name__ == '__main__':
    #run("psf.png")
    run("psf.png", optimise=False, heightmap_path="outputs/psf/heightmap_final.exr")
