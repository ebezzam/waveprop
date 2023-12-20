import numpy as np
import cv2

from simulator.psf_generation.psf_generator import PSFGenerator


class MaskPSFGenerator(PSFGenerator):
    """
    The easiest kind of realistic PSF Generator. It projects a given mask that blocks part of the light on the sensor.
    Black pixels are totally opaque and white pixels are totally transparent, with each other color or gray shade
    inbetween. Of course, it also takes in account the reduction of the irradiance due to the distance between the
    sensor and the light source, as well as the one due to the angle between the sensor normal and the light rays
    """
    def __init__(self, camera_settings, mask):
        super(MaskPSFGenerator, self).__init__(camera_settings=camera_settings)
        self.mask = mask
        self.normalisation_factor = np.max(self.mask)
        return

    def init_output(self, shape):
        super(MaskPSFGenerator, self).init_output(shape)
        self.width_factor = self.shape[0] * self.cam.diffuser_width / self.cam.sensor_width
        self.height_factor = self.shape[1] * self.cam.diffuser_height / self.cam.sensor_height

        x = (((np.arange(self.shape[0]) + 0.5) / self.shape[0]) - 0.5) * self.cam.sensor_width
        y = (((np.arange(self.shape[1]) + 0.5) / self.shape[1]) - 0.5) * self.cam.sensor_height
        x2 = x*x
        y2 = y*y

        self.squared_distances = x2[:, np.newaxis] + y2
        return

    def generate_layer(self, depth):

        # Radiance of a point source decreases in an inverse-square law
        d = (depth + self.cam.focal_distance)

        distance_factor = 1 / (self.squared_distances + d*d)
        cos_factor = d * np.sqrt(distance_factor)
        light_factor = distance_factor * cos_factor

        if len(self.mask.shape) == 3: #rgb
            light_factor = light_factor[:, :, np.newaxis]

        # calculate how the mask gets projected on the diffuser
        fact = d / depth
        x = int(self.width_factor * fact)
        y = int(self.height_factor * fact)

        projection = cv2.resize(self.mask, (x, y), interpolation=cv2.INTER_AREA)
        left = (self.shape[0] - x) // 2
        top = (self.shape[1] - y) // 2
        right = self.shape[0] - x - left
        bottom = self.shape[1] - y - left
        padded = cv2.copyMakeBorder(projection, top=top, left=left, bottom=bottom, right=right,
                                    borderType=cv2.BORDER_CONSTANT, value=0)
        return padded * light_factor
