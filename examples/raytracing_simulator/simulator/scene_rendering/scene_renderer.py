import abc
import time
import numpy as np
from multiprocessing import Pool

class SceneRenderer:
    def __init__(self, camera_settings, scene_settings):
        assert 0 < camera_settings.focal_distance and 0 < camera_settings.min_depth < camera_settings.max_depth
        self.cam = camera_settings
        self.scene = scene_settings
        self.normalisation_factor = 0
        self.processes = 1
        return

    def render(self, shape, normalize=True, multiprocess=True):
        """Produces a 2D scene by stacking the irradiance coming from different depth layers"""

        assert 2 <= len(shape) <= 3
        self.shape = shape
        self.multiprocess = multiprocess

        print("Beginning the rendering of a new Scene...")
        t = time.time()

        if multiprocess:
            print("Starting", self.processes, "processes...")
            with Pool() as pool:
                result = pool.map(self.render_layer, range(self.processes))
            result = np.array(result)
        else:
            result = np.array([self.render_layer(depth) for depth in range(self.processes)])

        result = np.sum(result, axis=0)

        if normalize and self.normalisation_factor > 0 and np.max(result) > 0:
            result *= (self.normalisation_factor / np.max(result))
        print(f"Total computation time : {time.time() - t} s")
        return result

    @abc.abstractmethod
    def render_layer(self, depth):
        """Produces a single 2D psf for a given depth, depending on the type of the generator"""
        return

