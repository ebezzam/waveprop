import abc
import numpy as np
import time
from multiprocessing import Pool

class PSFGenerator:
    def __init__(self, camera_settings):
        assert 0 < camera_settings.focal_distance and 0 < camera_settings.min_depth < camera_settings.max_depth
        self.cam = camera_settings
        self.normalisation_factor = 0
        return

    def init_output(self, shape):
        """Data output initialisation method. Can be overridden to compute additional data needed by the generator"""
        assert 3 <= len(shape) <= 4
        self.depths = np.linspace(self.cam.min_depth, self.cam.max_depth, shape[0])
        self.shape = shape[1:]

        return

    def generate(self, shape, normalize=True, multiprocess=True):
        """Produces a 3D psf by stacking several 2D psf together"""

        print("Beginning the generation of a new PSF...")
        t = time.time()
        self.init_output(shape)
        self.multiprocess = multiprocess

        if multiprocess:
            print("Starting", len(self.depths), "processes...")
            with Pool() as pool:
                result = pool.map(self.generate_layer, self.depths)
            result = np.array(result)
        else:
            result = np.array([self.generate_layer(depth) for depth in self.depths])

        if normalize and self.normalisation_factor > 0 and np.max(result) > 0:
            result *= (self.normalisation_factor / np.max(result))

        print(f"Total computation time : {time.time() - t} s")
        return result

    @abc.abstractmethod
    def generate_layer(self, depth):
        """Produces a single 2D psf for a given depth depending on the type of the generator"""
        return
