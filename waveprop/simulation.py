import numpy as np
from waveprop.util import prepare_object_plane, resize
from torchvision.transforms.functional import resize as resize_torch
from waveprop.devices import sensor_dict, SensorParam
from waveprop.noise import add_shot_noise
import torch
from waveprop.pytorch_util import RealFFTConvolve2D
import warnings


class FarFieldSimulator(object):

    """
    Simulate far-field propagation with the following steps:
    1. Resize digital image for desired object height and to PSF resolution.
    2. Convolve with PSF
    3. (Optionally) Resize to lower sensor resolution.
    4. (Optionally) Add shot noise
    5. Quantize


    Images and PSFs should be one of following shape
    - For numpy arrays: (H, W) for grayscale and (H, W, 3) for RGB.
    - For PyTorch tensors: (..., H, W)

    """

    def __init__(
        self,
        object_height,
        scene2mask,
        mask2sensor,
        sensor,
        psf=None,
        output_dim=None,
        snr_db=None,
        max_val=255,
        device_conv="cpu",
        random_shift=False,
        is_torch=False,
        quantize=True,
        **kwargs
    ):
        """
        Parameters
        ----------
        psf : np.ndarray or torch.Tensor, optional.
            Point spread function. If not provided, return image at object plane.
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
        device_conv : str, optional
            Device to use for convolution (when using pytorch), by default "cpu".
        random_shift : bool, optional
            Whether to randomly shift the image, by default False.
        is_torch : bool, optional
            Whether to use pytorch, by default False.
        quantize : bool, optional
            Whether to quantize image, by default True.
        """
        if is_torch:
            self.axes = (-2, -1)
            output_dtype = torch.uint8
        else:
            self.axes = (0, 1)
            output_dtype = np.uint8
        self.is_torch = is_torch

        # for resizing
        self.object_height = object_height
        self.scene2mask = scene2mask
        self.mask2sensor = mask2sensor
        self.sensor = sensor_dict[sensor]
        self.random_shift = random_shift
        self.quantize = quantize

        # for convolution
        if psf is not None:
            self.device_conv = device_conv
            self.set_psf(psf)

            # at sensor
            self.output_dim = output_dim
            self.snr_db = snr_db
            self.max_val = max_val
            self.output_dtype = output_dtype

        else:
            # simply return object / scene plane
            warnings.warn("No PSF provided. Returning image at object plane.")
            self.fft_shape = None
            assert output_dim is not None
            self.conv_dim = np.array(output_dim)

    def set_psf(self, psf):
        """
        Set PSF of simulator.

        Parameters
        ----------
        psf : np.ndarray or torch.Tensor
            Point spread function.
        """

        self.psf = psf
        self.conv_dim = np.array([psf.shape[_ax] for _ax in self.axes])
        self.fft_shape = 2 * np.array(self.conv_dim) - 1
        if torch.is_tensor(psf):
            self.conv = RealFFTConvolve2D(psf, device=self.device_conv)
        else:
            self.H = np.fft.rfft2(psf, s=self.fft_shape, axes=self.axes)
            # -- for removing padding
            self.y_pad_edge = int(
                (self.fft_shape[self.axes[0]] - self.conv_dim[self.axes[0]]) / 2
            )
            self.x_pad_edge = int(
                (self.fft_shape[self.axes[1]] - self.conv_dim[self.axes[1]]) / 2
            )

    def propagate(self, obj, return_object_plane=False):
        """

        Parameters
        ----------
        obj : np.ndarray
            Object to propagate.
        return_object_plane : bool, optional
            Whether to return object plane, by default False.
        """

        if self.is_torch:
            assert torch.is_tensor(obj)

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

        if self.fft_shape is not None:
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

            # 3) (Optionally) Downsample to sensor size
            if self.output_dim is not None:
                if torch.is_tensor(obj):
                    image_plane = resize_torch(image_plane, size=self.output_dim)
                else:
                    image_plane = resize(image_plane, shape=self.output_dim)

            # 4) (Optionally) Add shot noise
            if self.snr_db is not None:
                image_plane = add_shot_noise(image_plane, snr_db=self.snr_db)

            # 5) (Optionaly) Quantize as on sensor
            if self.quantize:
                image_plane = image_plane / image_plane.max()
                image_plane = image_plane * self.max_val
                if torch.is_tensor(image_plane):
                    image_plane = image_plane.to(self.output_dtype)
                else:
                    image_plane = image_plane.astype(self.output_dtype)

            if return_object_plane:
                return image_plane, object_plane
            else:
                return image_plane

        else:
            # return object plane for simulation with PSF at a different stage
            return object_plane
