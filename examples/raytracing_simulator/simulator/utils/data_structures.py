import numpy as np
import os

default_sensor_width = 2
default_sensor_height = 0
default_diffuser_width = 1
default_diffuser_height = 0
default_diffuser_thickness = 0.01
default_focal_distance = 0.1
default_min_depth = 1
default_max_depth = 2
default_scene_width = 5
default_scene_height = 0


"""
    The following classes are used to create and store settings for generating PSFs and rendering scenes.
    
    They are not sufficient themselves : additional data depending on the algorithms used to
    generate PSFs and render scenes will have to be provided.
    
    CameraSettings contains the physical dimensions of the simulated cameras, and as it serves
    to generate PSFs, two fields min_depth and max_depth are also added to store the depths range
    that will be took in account for the corresponding psf.
    
    SceneSettings contains the physical dimensions of the rendered scenes.

    Note 1 : The CameraSettings class has a field called diffuser_thickness. For now, this value is
    only used when generating a normal map from a height map. However, when generating PSFs from such
    normal maps, current algorithms do not take in account the small distance that the light rays travel
    inside the diffuser before exiting it on the other side, as typical thickness values are tiny.
    This allows to speed up the computations, but in the eventual case where one would simulate a thicker
    diffuser, the code would have to be updated accordingly.
    
    Note 2 : When rendering a scene with a certain PSF, the SceneSettings must have the same min_depth and
    max_depth values that the ones of the CameraSettings that were used to simulate the used PSF.
    
"""

class CameraSettings:
    def __init__(self, sensor_width, sensor_height, diffuser_width, diffuser_height, diffuser_thickness,
                 focal_distance, min_depth, max_depth):
        self.sensor_width = sensor_width
        self.sensor_height = sensor_height
        self.diffuser_width = diffuser_width
        self.diffuser_height = diffuser_height
        self.diffuser_thickness = diffuser_thickness
        self.focal_distance = focal_distance
        self.min_depth = min_depth
        self.max_depth = max_depth
        return

    def save(self, path):
        np.save(path, np.array([
            self.sensor_width,
            self.sensor_height,
            self.diffuser_width,
            self.diffuser_height,
            self.diffuser_thickness,
            self.focal_distance,
            self.min_depth,
            self.max_depth
        ]))


class SceneSettings:
    def __init__(self, width, height, min_depth, max_depth):
        self.width = width
        self.height = height
        self.min_depth = min_depth
        self.max_depth = max_depth
        return

    def save(self, path):
        np.save(path, np.array([
            self.width,
            self.height,
            self.min_depth,
            self.max_depth
        ]))


def default_camera():
    return CameraSettings(
        sensor_width=default_sensor_width,
        sensor_height=default_sensor_height if default_sensor_height != 0 else default_sensor_width,
        diffuser_width=default_diffuser_width,
        diffuser_height=default_diffuser_height if default_diffuser_height != 0 else default_diffuser_width,
        diffuser_thickness=default_diffuser_thickness,
        focal_distance=default_focal_distance,
        min_depth=default_min_depth,
        max_depth=default_max_depth
    )


def default_scene():
    return SceneSettings(
        width=default_scene_width,
        height=default_scene_height if default_scene_height != 0 else default_scene_width,
        min_depth=default_min_depth,
        max_depth=default_max_depth
    )


def load_camera(path):
    if not os.path.isfile(path):
        print("Notice : path for loading camera doesn't exists : ", path,"\n > Use default camera instead.")
        return default_camera()
    c = np.load(path)

    if not c.shape == (8,):
        print("Notice : data for loading camera as incorrect format :", path, " - shape : ", c.shape,
              "\n > Use default camera instead.")
        return default_camera()

    return CameraSettings(c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7])


def load_scene(path):
    if not os.path.isfile(path):
        print("Notice : path for loading scene doesn't exists : ", path, "\n > Use default scene instead.")
        return default_scene()
    s = np.load(path)

    if not s.shape == (4,):
        print("Notice : data for loading scene as incorrect format :", path, " - shape : ", s.shape,
              "\n > Use default camera instead")
        return default_scene()

    return SceneSettings(s[0], s[1], s[2], s[3])