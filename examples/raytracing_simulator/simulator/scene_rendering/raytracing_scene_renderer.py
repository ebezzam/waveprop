import time
import numpy as np

from simulator.utils.lerp import lerp_read
from simulator.scene_rendering.scene_renderer import SceneRenderer

"""
At the contrary of the convolution scene renderer, the raytracing scene renderer, manually traces a ray between each
point of the scene and each point of the psf

It however only requires to provide a 2D psf and does not relies on the assumption of horizontal and vertical
shift-invarance of the psf, which can technically yield more realistic results, but at the cost of a much slower
computation, and should therefore not be used in practical cases.

Moreover, as its input is actually a mask rather than a psf (even though it's named psf for similarity with other
classes), the data that it outputs doesn't have to correspond to a psf and therefore can be scaled to an arbitrary size.
"""


class RaytracingSceneRenderer(SceneRenderer):

    def __init__(self, camera_settings, scene_settings, scene_radiance, scene_depths, psf):
        super(RaytracingSceneRenderer, self).__init__(camera_settings=camera_settings, scene_settings=scene_settings)
        assert len(scene_depths.shape) == 2
        self.radiance = scene_radiance
        self.depths = scene_depths
        if np.max(scene_depths) >0 :
            self.depths = scene_depths / np.max(scene_depths)
        self.psf = psf
        self.psf_w, self.psf_h = psf.shape[:2]
        self.processes = 1
        self.normalisation_factor = 255
        self.scene_min_width = scene_settings.width * scene_settings.min_depth / scene_settings.max_depth
        self.scene_min_height = scene_settings.height * scene_settings.min_depth / scene_settings.max_depth

        # if grayscale image/psf we add another axis to match rgb.
        if len(self.radiance.shape) == 2:
            self.radiance = self.radiance[:, :, np.newaxis]

        if len(self.psf.shape) == 2:
            self.psf = self.psf[:, :, np.newaxis]

        if self.radiance.shape[2] == 1 and self.psf.shape[2] == 3:
            self.radiance = np.repeat(self.radiance, 3, axis=2)

        if self.radiance.shape[2] == 3 and self.psf.shape[2] == 1:
            self.psf = np.repeat(self.radiance, 3, axis=2)

        return

    def render(self, shape, normalize=True, multiprocess=True):
        if len(shape) == 2:
            shape = (shape[0], shape[1], self.radiance.shape[2])
        assert len(shape) == 3
        self.shape = shape
        self.multiprocess = multiprocess

        print("Beginning the rendering of a new Scene...")
        t = time.time()

        result = self.simulate()

        if normalize and self.normalisation_factor > 0 and np.max(result) > 0:
            result *= (self.normalisation_factor / np.max(result))

        print(f"Total computation time : {time.time() - t} s")
        return result



    def simulate(self):

        result = np.zeros(self.shape * np.array([1, 1, 1])).astype(np.float32)
        for x in range(self.radiance.shape[0]):
            print(x, "/", self.radiance.shape[0])
            for y in range(self.radiance.shape[1]):
                if np.sum(self.radiance[x, y]) > 0:
                    result += self.evaluate_psf(
                        x=(x / (self.radiance.shape[0] - 1) - 0.5),
                        y=(y / (self.radiance.shape[1] - 1) - 0.5),
                        z=self.depths[x, y]
                    ) * self.radiance[x, y]

        return result

    def evaluate_psf(self, x, y, z):
        """
        x,y,z : relative input point in the scene coordinates, from [-0.5;-0.5, 0] to [0.5,0.5,1]
        """

        actual_x = x * (self.scene.width * z + self.scene_min_width * (1-z))
        actual_y = y * (self.scene.height * z + self.scene_min_height * (1-z))
        actual_z = z * (self.scene.max_depth - self.scene.min_depth) + self.scene.min_depth + self.cam.focal_distance
        depth_factor = self.cam.focal_distance / actual_z

        # coordinates of the sensor
        cx = self.cam.sensor_width * (np.arange(self.shape[0]) / (self.shape[0]-1) - 0.5)
        cy = self.cam.sensor_height * (np.arange(self.shape[1]) / (self.shape[1]-1) - 0.5)



        # light : squared distances between points and sensor
        dx = actual_x - cy
        dy = actual_y - cy
        d2 = np.array([[x*x+y*y+actual_z*actual_z for y in dy] for x in dx])[:,:,np.newaxis]

        # coordinates of intersection with the diffuser (absolute)
        cx = cx + depth_factor * (actual_x - cx)
        cy = cy + depth_factor * (actual_y - cy)

        #cooridantes of intersection with the diffuser (relative)
        cx = cx / self.cam.diffuser_width + 0.5
        cy = cy / self.cam.diffuser_height + 0.5

        result_in = cx[:, np.newaxis, np.newaxis] * [[[1, 0]]] + cy[np.newaxis, :, np.newaxis] * [[[0, 1]]]



        result = np.array([[lerp_read(self.psf, r[0], r[1], self.psf_w, self.psf_h) for r in res] for res in result_in])
        #account for geometric term
        return result / d2

    def get(self, arr):
        return lerp_read(self.psf, arr[:, :, 0], arr[:, :, 1], self.psf_w, self.psf_h)
