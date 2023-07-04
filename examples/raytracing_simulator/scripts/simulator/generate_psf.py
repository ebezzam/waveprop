import os
import click
import cv2

from simulator.utils.data_structures import load_camera
from simulator.psf_generation.mask_psf_generator import MaskPSFGenerator
from simulator.psf_generation.diffuser_psf_generator import DiffuserPSFGenerator
from simulator.utils.normals import *
from lensless.io import load_image


@click.command()
@click.option(
    "--input_fp",
    type=str,
    help="File name for loading the mask or diffuser image.",
)
@click.option(
    "--camera_fp",
    type=str,
    default="",
    help="File name for loading the camera settings",
)
@click.option(
    "--output_fp",
    type=str,
    help="File name for saving the output",
)
@click.option(
    "--dest_shape",
    default=None,
    nargs=3,
    type=int,
    help="The shape to output (depth, width, height). If None, will use dest_depth and input's dimensions instead.",
)
@click.option(
    "--dest_depth",
    default=10,
    type=int,
    help="Used to define dest_shape when it is None.",
)
@click.option(
    "--no_normalize",
    is_flag=True,
    help="If selected, the psf will not be normalized before being saved",
)
@click.option(
    "--mode",
    type=str,
    default="mask",
    help="The type of data we use to build the psf. Is either 'mask', 'height', or 'normal'",
)
@click.option(
    "--opengl",
    is_flag=True,
    help="MODE : NORMAL - The default format of normal map is directx. This flag allows to use opengl instead",
)
@click.option(
    "--sobel_size",
    type=int,
    default=7,
    help="MODE : HEIGHT - The size of the sobel operator kernel, used to generate a normal map from the height map."
         "Min value to work is 3, but higher values can be used for smoother results"
)
@click.option(
    "--save_normals",
    is_flag=True,
    help="MODE : HEIGHT - If selected, the generated normal map generated from the height map will be saved as image."
)
@click.option(
    "--n_air",
    type=float,
    default=1,
    help="MODE : NORMAL / HEIGHT - Refractive index of the principal medium (default is 1 for air, hence the name)",
)
@click.option(
    "--n_dif",
    type=float,
    default=1.5,
    help="MODE : NORMAL / HEIGHT - Refractive index of the diffuser medium (default is 1.5 for optical glass)",
)
@click.option(
    "--oversample",
    type=float,
    default=1,
    help="MODE : NORMAL / HEIGHT - Scales up the normal map from this amount in both dimensions to trace more rays"
         "too big values may cause the program to crash by running out of memory if the normal map is already big",
)
@click.option(
    "--refract_once",
    is_flag=True,
    help="MODE : NORMAL / HEIGHT - If selected, the simulation will consider that the area between the diffuser and"
         "the sensor is not made of air, but of the same medium as the diffuser"
)
@click.option(
    "--no_multiprocess",
    is_flag=True,
    help="If selected, the computations will not be distributed among several processes, which will likely make them"
         "much slower but spare the CPU"
)
def run(input_fp,
        camera_fp,
        output_fp,
        dest_depth,
        dest_shape,
        no_normalize,
        mode,
        opengl,
        sobel_size,
        save_normals,
        n_air,
        n_dif,
        oversample,
        refract_once,
        no_multiprocess):

    if (input_fp is None) or (input_fp == "") or (not os.path.isfile(input_fp)):
        print("Error : wrong file path provided for input (--input_fp)")
        return

    if (output_fp is None) or (output_fp == ""):
        print("Error : no file path provided for output (--out_fp)")
        return

    image = load_image(input_fp)
    cam = load_camera(camera_fp)  # providing no file path to the camera settings will make use default settings

    if dest_shape is None:
        assert dest_depth > 0
        x, y = image.shape[:2] #TODO : check rgb psfs
        dest_shape = np.array([dest_depth, x, y]).astype(np.int32)

    if mode == "mask":
        generator = MaskPSFGenerator(cam, image)

    elif mode == "normals":
        assert len(image.shape) == 3
        if oversample > 1:
            dsize = (int(image.shape[0] * oversample), int(image.shape[1] * oversample))
            image = cv2.resize(image, dsize=dsize, interpolation=cv2.INTER_CUBIC)
        elif 0 < oversample < 1:
            dsize = (int(image.shape[0] * oversample), int(image.shape[1] * oversample))
            image = cv2.resize(image, dsize=dsize, interpolation=cv2.INTER_AREA)
        normal_map = image_to_normals(image, opengl)
        generator = DiffuserPSFGenerator(cam, normals=normal_map, n_air=n_air, n_dif=n_dif, refract_once=refract_once)

    elif mode == "height":
        if oversample > 1:
            dsize = (int(image.shape[0] * oversample), int(image.shape[1] * oversample))
            image = cv2.resize(image, dsize=dsize, interpolation=cv2.INTER_CUBIC)
        elif 0 < oversample < 1:
            dsize = (int(image.shape[0] * oversample), int(image.shape[1] * oversample))
            image = cv2.resize(image, dsize=dsize, interpolation=cv2.INTER_AREA)
        normal_map = heights_to_normals(
            image, width=cam.sensor_width,
            height=cam.sensor_height,
            thickness=cam.diffuser_thickness,
            kernel_size=sobel_size
        )
        generator = DiffuserPSFGenerator(cam, normals=normal_map, n_air=n_air, n_dif=n_dif, refract_once=refract_once)

        if save_normals:
            normals_img = normals_to_image(normal_map)
            norm_path = os.path.splitext(input_fp)[0] + "-normals.png"
            cv2.imwrite(norm_path, cv2.cvtColor(normals_img, cv2.COLOR_RGB2BGR))

    else:
        print("Error : parameter --mode is wrong. Should be mask, normals or height, but is", mode)
        return

    result = generator.generate(dest_shape, not no_normalize, not no_multiprocess)
    np.save(output_fp, result)

    print("Final result of color range from", np.min(result), "to", np.max(result), "saved at", output_fp)

    return


if __name__ == "__main__":
    run()
