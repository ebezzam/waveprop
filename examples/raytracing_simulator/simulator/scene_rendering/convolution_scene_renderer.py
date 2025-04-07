import cv2
import numpy as np
from scipy import signal
from simulator.scene_rendering.scene_renderer import SceneRenderer


class ConvolutionSceneRenderer(SceneRenderer):

    def __init__(self, camera_settings, scene_settings, scene_radiance, scene_depths, psf, exclude_upper_depth=False):
        super(ConvolutionSceneRenderer, self).__init__(camera_settings=camera_settings, scene_settings=scene_settings)
        assert len(scene_depths.shape) == 2  # w, h
        assert 1 < len(scene_radiance.shape) < 4  # can either be (w, h), (w, h, 1) or (w, h, 3) for rgb
        assert 2 < len(psf.shape) < 5  # can either be (d, w, h), (d, w, h, 1) or (d, w, h, 3) for rgb
        assert scene_radiance.shape[:2] == scene_depths.shape  # radiance and depths must have same spatial dimensions

        # We are flipping the scene to represent the light rays going through the aperture
        self.scene_radiance = np.flipud(np.fliplr(scene_radiance))
        self.scene_depths = np.flipud(np.fliplr(scene_depths))
        self.scene_depths /= np.max(self.scene_depths)
        self.scene_depths = self.scene_depths * (scene_settings.max_depth - scene_settings.min_depth) + scene_settings.min_depth
        self.psf = psf
        self.processes = psf.shape[0]
        self.normalisation_factor = 255 #np.max(psf)


        # extending grayscale scene_radiance shape if needed
        if len(self.scene_radiance.shape) == 2:
            self.scene_radiance = self.scene_radiance[:, :, np.newaxis]

        # extending grayscale psf shape if needed
        if len(self.psf.shape) == 3:
            self.psf = self.psf[:, :, :, np.newaxis]

        # converting scene to rgb if grayscale with a rgb psf
        if self.scene_radiance.shape[2] == 1 and self.psf.shape[3] == 3:
            self.scene_radiance = np.repeat(self.scene_radiance, 3, axis=2)

        # converting psf to rgb if grayscale with a rgb scene
        if self.scene_radiance.shape[2] == 3 and self.psf.shape[3] == 1:
            self.psf = np.repeat(self.psf, 3, axis=3)


        # we need to scale the psf so that it has the same ratio of px by metric units as the ratio that the radiance
        # map would have if it was rescaled to fit at the same distance from the diffuser than the distance between
        # the diffuser and sensor
        self.scale_factor = (camera_settings.sensor_width * scene_settings.max_depth) /\
                            (scene_settings.width * camera_settings.focal_distance)

        dsize = (int(self.scale_factor * self.scene_radiance.shape[0]), int(self.scale_factor * self.scene_radiance.shape[1]))
        self.psf = np.array([cv2.resize(p, dsize=dsize, interpolation=cv2.INTER_CUBIC) for p in self.psf])

        # normalizing psf, as we also do it in the reconstruction algorithms
        self.psf = self.psf / np.max(self.psf)

        # calculate the depths corresponding to each psf layer
        self.psf_depths = np.linspace(self.scene.min_depth, self.scene.max_depth, self.psf.shape[0])

        # split the scene in different depths layers. Each layer will only contain radiance from elements that are
        # inside them. We will then use the corresponding layer of the psf on it.

        thresholds = [self.psf_depths[0]] + [(self.psf_depths[i] + self.psf_depths[i + 1]) / 2
                                             for i in range(self.psf_depths.size - 1)] + [self.psf_depths[-1]]
        if not exclude_upper_depth:
            epsilon = 0.01
            thresholds[-1] += epsilon

        #scale = np.linspace(1, self.scene.min_depth / self.scene.max_depth, self.psf.shape[0])

        self.slices = np.array(
            [self.split_layer(thresholds[i], thresholds[i + 1]) for i in range(self.psf_depths.size)])


        return

    def render_layer(self, d):

        #scene gets projected onto the sensor through the diffuser aperture
        scene = self.slices[d]
        psf = self.psf[d]

        conv = np.array([signal.fftconvolve(scene[:, :, c], psf[:,:,c]) for c in range(self.slices.shape[3])])
        layer = np.moveaxis(conv, 0, 2)
        dsize = (int(layer.shape[0] / self.scale_factor),int(layer.shape[1] / self.scale_factor))
        #scale back so that the output dimension is correct

        layer = cv2.resize(layer, dsize=dsize,interpolation=cv2.INTER_AREA)
        return layer


    def split_layer(self, min_depth, max_depth):
        nx, ny, c = self.scene_radiance.shape
        empty = [0 for _ in range(c)]

        layer = np.array([[self.scene_radiance[x, y] if min_depth <= self.scene_depths[x, y] < max_depth else empty
                          for y in range(ny)] for x in range(nx)])

        return layer

    def pad(self, image, new_shape, corner=False):
        img_shape = image.shape
        if corner:
            top = 0
            left = 0
        else:
            top = int((new_shape[1] - img_shape[0]) // 2)
            left = int((new_shape[0] - img_shape[1]) // 2)
        bottom = int(new_shape[1] - img_shape[0] - top)
        right = int(new_shape[0] - img_shape[1] - left)

        return cv2.copyMakeBorder(image, top=top, left=left, bottom=bottom, right=right,
                                  borderType=cv2.BORDER_CONSTANT, value=0)

    #Unused
    def scale_and_conv(self, d, c):
        image = self.slices[d, :, :, c]
        psf = self.psf[d, :, :, c]
        shape = (max(image.shape[0], psf.shape[0]), max(image.shape[1], psf.shape[1]))
        pad_image = self.pad(image, shape, corner=True)
        pad_psf = self.pad(psf, shape, corner=True)
        f_image = np.fft.rfftn(pad_image)
        f_psf = np.fft.rfftn(pad_psf)

        new_shape = (f_psf.shape[0], f_psf.shape[1])
        f_psf = self.pad(f_psf.astype(np.float32), new_shape)

        f_image = cv2.resize(f_image.astype(np.float32), dsize=new_shape)

        f_result = f_image * f_psf

        return np.fft.irfft2(f_result)