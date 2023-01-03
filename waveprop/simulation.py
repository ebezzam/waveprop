import numpy as np
from waveprop.util import prepare_object_plane
from waveprop.devices import sensor_dict, SensorParam
from waveprop.noise import add_shot_noise
import torch
from waveprop.pytorch_util import RealFFTConvolve2D


class ConvolutionWithPSF(object):

    """
    Images and PSFs should be one of following shape, if numpy arrays:
    - Grayscale: (H, W)
    - RGB: (H, W, 3)

    If PyTorch tensors: (..., H, W)

    """

    def __init__(
        self,
        psf,
        object_height,
        scene2mask,
        mask2sensor,
        sensor,
        snr_db=None,
        max_val=255,
        device_conv="cpu",
        random_shift=False,
        **kwargs
    ):
        """
        Parameters
        ----------
        psf : np.ndarray
            Point spread function.
        object_height : float or (float, float)
            Height of object in meters. Or range of values to randomly sample from.
        scene2mask : float
            Distance from scene to mask in meters.
        mask2sensor : float
            Distance from mask to sensor in meters.
        sensor : str
            Sensor name.
        snr_db : float, optional
            Signal-to-noise ratio in dB, by default None.
        max_val : int, optional
            Maximum value of image, by default 255.
        output_dtype : np.dtype, optional
            Data type of output image, by default uint8.

        """
        if torch.is_tensor(psf):
            self.axes = (-2, -1)
            output_dtype = torch.uint8
        else:
            self.axes = (0, 1)
            output_dtype = np.uint8

        # for resizing
        self.object_height = object_height
        self.scene2mask = scene2mask
        self.mask2sensor = mask2sensor
        self.sensor = sensor_dict[sensor]
        self.conv_dim = np.array([psf.shape[_ax] for _ax in self.axes])
        self.random_shift = random_shift

        # for convolution
        self.fft_shape = 2 * np.array(self.conv_dim) - 1
        if torch.is_tensor(psf):
            self.conv = RealFFTConvolve2D(psf, device=device_conv)
        else:
            self.H = np.fft.rfft2(psf, s=self.fft_shape, axes=self.axes)
            # -- for removing padding
            self.y_pad_edge = int((self.fft_shape[self.axes[0]] - self.conv_dim[self.axes[0]]) / 2)
            self.x_pad_edge = int((self.fft_shape[self.axes[1]] - self.conv_dim[self.axes[1]]) / 2)

        # at sensor
        self.snr_db = snr_db
        self.max_val = max_val
        self.output_dtype = output_dtype

    def propagate(self, obj, return_object_plane=False):
        """

        Parameters
        ----------
        obj : np.ndarray
            Object to propagate.
        return_object_plane : bool, optional
            Whether to return object plane, by default False.
        """

        if self.axes == (-2, -1):
            if obj.shape[-1] <= 3:
                raise ValueError("Channel dimension should not be last.")
        elif self.axes == (0, 1):
            if obj.shape[0] <= 3:
                raise ValueError("Channel dimension should not be first.")

        # 1) Resize image to PSF dimensions while keeping aspect ratio and
        # setting object height to desired value.
        if hasattr(self.object_height, "__len__"):
            object_height = np.random.uniform(low=self.object_height[0], high=self.object_height[1])
        else:
            object_height = self.object_height

        object_plane = prepare_object_plane(
            obj=obj,
            object_height=object_height,
            scene2mask=self.scene2mask,
            mask2sensor=self.mask2sensor,
            sensor_size=self.sensor[SensorParam.SIZE],
            sensor_dim=self.conv_dim,
            random_shift=self.random_shift,
        )

        # 2) Convolve with PSF
        if torch.is_tensor(object_plane):
            image_plane = self.conv(object_plane)
        else:
            I = np.fft.rfft2(object_plane, s=self.fft_shape, axes=self.axes)
            image_plane = np.fft.irfft2(self.H * I, s=self.fft_shape, axes=self.axes)
            image_plane = image_plane[
                self.y_pad_edge : self.y_pad_edge + self.conv_dim[0],
                self.x_pad_edge : self.x_pad_edge + self.conv_dim[1],
            ]

        # 3) Add shot noise
        if self.snr_db is not None:
            image_plane = add_shot_noise(image_plane, snr_db=self.snr_db)

        # 4) Quantize as on sensor
        image_plane /= image_plane.max()
        image_plane *= self.max_val
        if torch.is_tensor(image_plane):
            image_plane = image_plane.to(self.output_dtype)
        else:
            image_plane = image_plane.astype(self.output_dtype)

        if return_object_plane:
            return image_plane, object_plane
        else:
            return image_plane
