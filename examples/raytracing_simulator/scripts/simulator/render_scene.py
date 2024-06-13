import os
import click
import numpy as np
import cv2

from simulator.utils.data_structures import load_camera, load_scene
from simulator.scene_rendering.convolution_scene_renderer import ConvolutionSceneRenderer
from simulator.scene_rendering.raytracing_scene_renderer import RaytracingSceneRenderer
from lensless.io import load_image


@click.command()
@click.option(
    "--radiance_fp",
    type=str,
    help="File name for loading the scene radiance map.",
)
@click.option(
    "--depths_fp",
    type=str,
    help="File name for loading the scene depths map.",
)
@click.option(
    "--psf_fp",
    type=str,
    help="File name for loading the psf.",
)
@click.option(
    "--camera_fp",
    type=str,
    default="",
    help="File name for loading the camera settings",
)
@click.option(
    "--scene_fp",
    type=str,
    default="",
    help="File name for loading the scene settings",
)
@click.option(
    "--out_fp",
    type=str,
    help="File name for saving the output",
)
@click.option(
    "--dest_shape",
    default=None,
    nargs=2,
    type=int,
    help="The shape to output (width, height). If None, will use input's dimensions instead.",
)
@click.option(
    "--no_normalize",
    is_flag=True,
    help="If selected, the image will not be normalized before being saved.",
)
@click.option(
    "--mode",
    type=str,
    default="convolution",
    help="The type of algorithm we use to render the psf. Is either 'convolution' or 'raytracing'",
)
@click.option(
    "--no_multiprocess",
    is_flag=True,
    help="If selected, the computations will not be distributed among several processes, which will likely make them"
         "much slower but spare the CPU"
)
def run(radiance_fp,
        depths_fp,
        psf_fp,
        camera_fp,
        scene_fp,
        out_fp,
        dest_shape,
        no_normalize,
        mode,
        no_multiprocess):

    if (radiance_fp is None) or (radiance_fp is None) or (radiance_fp == "") or (not os.path.isfile(radiance_fp)):
        print("Error : wrong file path provided for input scene radiance (--radiance_fp)")
        return

    if (depths_fp is None) or(depths_fp == "") or (not os.path.isfile(depths_fp)):
        print("Error : wrong file path provided for input scene depths (--depths_fp)")
        return

    if (psf_fp is None) or (psf_fp == "") or (not os.path.isfile(psf_fp)):
        print("Error : wrong file path provided for input psf (--psf_fp)")
        return

    if (out_fp is None) or (out_fp == ""):
        print("Error : no file path provided for output (--out_fp)")
        return

    radiance = load_image(radiance_fp).astype(np.float32)
    depths = load_image(depths_fp).astype(np.float32)
    cam = load_camera(camera_fp)  # providing no file path to the camera settings will make use default settings
    scene = load_scene(scene_fp)  # providing no file path to the scene settings will make use default settings

    if len(depths.shape) == 3:
        print("Averaging rgb image of depths into a grayscale image")
        depths = np.average(depths, axis=2)

    assert radiance.shape[:2] == depths.shape

    if dest_shape is None:
        dest_shape = radiance.shape

    if mode == "convolution":
        psf_3d = np.load(psf_fp)
        renderer = ConvolutionSceneRenderer(cam, scene, scene_radiance=radiance, scene_depths=depths, psf=psf_3d)

    elif mode == "raytracing":
        psf_2d = load_image(psf_fp)
        """
        image = scene_radiance.astype(np.float32) / 255
        psf = psf.astype(np.float32) / 255
        depths = scene_depths.astype(np.float32) / 255
        
        """
        renderer = RaytracingSceneRenderer(cam, scene, scene_radiance=radiance, scene_depths=depths, psf=psf_2d)

    else:
        print("Error : parameter --mode is wrong. Should be convolution or raytracing, but is", mode)
        return

    result = renderer.render(dest_shape, not no_normalize, not no_multiprocess)
    cv2.imwrite(out_fp, cv2.cvtColor(result.astype(np.float32), cv2.COLOR_RGB2BGR))

    print("Final result of color range from", np.min(result), "to", np.max(result), "saved at", out_fp)

    return


if __name__ == "__main__":
    run()
