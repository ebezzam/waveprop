import click
from simulator.utils.data_structures import *


@click.command()
@click.option(
    "--path",
    type=str,
    default="",
    help="Path to store the data.",
)
@click.option(
    "--sensor_width",
    type=float,
    default=default_sensor_width,
    help="Sensor width in metric units.",
)
@click.option(
    "--sensor_height",
    type=float,
    default=default_sensor_height,
    help="Sensor height in metric units. If 0, is equal to sensor_width",
)
@click.option(
    "--diffuser_width",
    type=float,
    default=default_diffuser_width,
    help="diffuser/mask width in metric units.",
)
@click.option(
    "--diffuser_height",
    type=float,
    default=default_diffuser_height,
    help="diffuser/mask width in metric units. If 0, is equal to diffuser_width",
)
@click.option(
    "--diffuser_thickness",
    type=float,
    default=default_diffuser_thickness,
    help="diffuser thickness in metric units. Used only to convert a height map to a normal map.",
)
@click.option(
    "--focal_distance",
    type=float,
    default=default_focal_distance,
    help="Distance between mask and sensor in metric units.",
)
@click.option(
    "--scene_min_depth",
    type=float,
    default=default_min_depth,
    help="Distance between mask and the front of the scene in metric units.",
)
@click.option(
    "--scene_max_depth",
    type=float,
    default=default_max_depth,
    help="Distance between mask and the back of the scene in metric units.",
)
@click.option(
    "--scene_width",
    type=float,
    default=default_scene_width,
    help="Width of the scene in metric units.",
)
@click.option(
    "--scene_height",
    type=float,
    default=default_scene_height,
    help="Width of the scene in metric units. If 0, is equal to scene_width",
)
@click.option(
    "--help",
    is_flag=True,
    help="If selected, will print the different fields available"
)
def create(path, sensor_width, sensor_height, diffuser_width, diffuser_height, diffuser_thickness,
           focal_distance, scene_min_depth, scene_max_depth, scene_width, scene_height, help):

    if path == "":
        print("Error : you must specify a --path argument. You may also try --help.")
        quit()
    if help:
        print("\nAvailable fields :\n"
              "path (mandatory)\n"
              "sensor_width,       ( default :", default_sensor_width,")\n"
              "sensor_height,      ( default :", default_sensor_height,")\n"
              "diffuser_width,     ( default :", default_diffuser_width,")\n"
              "diffuser_height,    ( default :", default_diffuser_height,")\n"
              "diffuser_thickness, ( default :", default_diffuser_thickness,")\n"
              "focal_distance,     ( default :", default_focal_distance,")\n"
              "scene_min_depth,    ( default :", default_min_depth,")\n"
              "scene_max_depth,    ( default :", default_max_depth,")\n"
              "scene_width,        ( default :", default_scene_width,")\n"
              "scene_height        ( default :", default_scene_height,")\n")
        return

    if diffuser_height == 0:
        diffuser_height = diffuser_width

    if sensor_height == 0:
        sensor_height = sensor_width

    if scene_height == 0:
        scene_height = scene_width

    assert path != ""
    assert sensor_width > 0
    assert sensor_height > 0
    assert diffuser_width > 0
    assert diffuser_height > 0
    assert diffuser_thickness > 0
    assert focal_distance > 0
    assert scene_max_depth > scene_min_depth > 0
    assert scene_width > 0
    assert scene_height > 0

    cam = CameraSettings(
        sensor_width=sensor_width,
        sensor_height=sensor_height,
        diffuser_width=diffuser_width,
        diffuser_height=diffuser_height,
        diffuser_thickness=diffuser_thickness,
        focal_distance=focal_distance,
        min_depth=scene_min_depth,
        max_depth=scene_max_depth
    )

    scene = SceneSettings(
        width=scene_width,
        height=scene_height,
        min_depth=scene_min_depth,
        max_depth=scene_max_depth
    )

    path = os.path.splitext(path)[0]  # removing the file extension from the path, if any

    cam.save(path + "-cam.npy")
    scene.save(path + "-scene.npy")


if __name__ == "__main__":
    create()
