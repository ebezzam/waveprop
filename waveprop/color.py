"""
Largely inspired by: https://github.com/rafael-fuente/Diffraction-Simulations--Angular-Spectrum-Method/blob/main/diffractsim/colour_functions.py

"""

import numpy as np
from scipy import interpolate
from pathlib import Path
from .util import gamma_correction
import torch
from torchvision.transforms.functional import rgb_to_grayscale


def rgb2gray(rgb, weights=None):
    """
    Convert RGB array to grayscale. Supports PyTorch and NumPy.

    Parameters
    ----------
    rgb : :py:class:`~numpy.ndarray`
        (N_height, N_width, N_channel) image.
    weights : :py:class:`~numpy.ndarray`
        [Optional] (3,) weights to convert from RGB to grayscale.

    Returns
    -------
    img :py:class:`~numpy.ndarray`
        Grayscale image of dimension (height, width).

    """
    if torch.is_tensor(rgb):
        return rgb_to_grayscale(rgb)
    else:
        if weights is None:
            weights = np.array([0.299, 0.587, 0.114])
        assert len(weights) == 3
        return np.tensordot(rgb, weights, axes=((2,), 0))


class ColorSystem:
    def __init__(self, n_wavelength=None, wv=None, color_mapping_txt=None, illumination_txt=None):
        """
        Color conversion class.

        TODO : add option to set XYZ to sRGB matrix.

        Parameters
        ----------
        n_wavelength : int
            Number of wavelengths, sampled uniformly between 380nm and 780nm.
        color_mapping_txt : str, optional
            Path to TXT file containing X,Y,Z coefficients for each wavelength.
        illumination_txt: str, optional
            Path to TXT file containing emittance for each wavelength.
        """
        if n_wavelength is None:
            assert wv is not None
            wv = np.array(wv)
            n_wavelength = len(wv)
        if color_mapping_txt is None:
            color_mapping_txt = Path(__file__).parent / "./lookup/cie-cmf.txt"
        if illumination_txt is None:
            illumination_txt = Path(__file__).parent / "../waveprop/lookup/illuminant_d65.txt"
        self.n_wavelength = n_wavelength

        # XYZ mapping
        cmf = np.loadtxt(color_mapping_txt)
        lookup_wavelength = cmf[:, 0] / 1e9
        min_wv = min(lookup_wavelength)
        max_wv = max(lookup_wavelength)

        if n_wavelength == len(lookup_wavelength) and wv is None:
            self.wv = lookup_wavelength
            self.cie_xyz = cmf[:, 1:]
        else:
            if wv is None:
                self.wv = np.linspace(start=min_wv, stop=max_wv, num=n_wavelength)
            else:
                self.wv = wv
            f = interpolate.interp1d(lookup_wavelength, cmf[:, 1:], axis=0, kind="linear")
            self.cie_xyz = f(self.wv)

        self.d_wv = self.wv[1] - self.wv[0]

        # emittance per wavelength
        emit = np.loadtxt(illumination_txt)
        lookup_wavelength = emit[:, 0]
        min_wv = min(lookup_wavelength)
        max_wv = max(lookup_wavelength)

        if n_wavelength == len(lookup_wavelength):
            self.emit = emit[:, 1:]
        else:
            wv = np.linspace(start=min_wv, stop=max_wv, num=n_wavelength)
            f = interpolate.interp1d(lookup_wavelength, emit[:, 1:], axis=0, kind="linear")
            self.emit = f(wv)

        # http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
        # https://stackoverflow.com/questions/66360637/which-matrix-is-correct-to-map-xyz-to-linear-rgb-for-srgb
        self.xyz_to_srgb = np.array(
            [
                [3.240969941904523, -1.537383177570094, -0.498610760293003],
                [-0.969243636280880, 1.875967501507721, 0.041555057407176],
                [0.055630079696994, -0.203976958888977, 1.056971514242879],
            ]
        )

    @classmethod
    def rgb(cls):
        return cls(wv=np.array([460, 550, 640]) * 1e-9)

    def to_rgb(self, vals, clip=True, gamma=None):
        """

        TODO : flatten inside here

        Parameters
        ----------
        vals : array_like
            (Ny, Nx, n_wavelength) Array of spectrum data at multiple wavelengths.


        Returns
        -------

        """
        assert len(vals.shape) == 3
        assert vals.shape[2] == self.n_wavelength

        xyz = vals.reshape(-1, self.n_wavelength) * self.emit.T @ self.cie_xyz * self.d_wv
        rgb = self.xyz_to_srgb @ xyz.T

        if clip:
            # clipping, add enough white to make all values positive
            # -- http://www.fourmilab.ch/documents/specrend/specrend.c, constrain_rgb
            # -- https://github.com/rafael-fuente/Diffraction-Simulations--Angular-Spectrum-Method/blob/5e82083831acb5729550360c5295447dddb77ca5/diffractsim/colour_functions.py#L78
            rgb_min = np.amin(rgb, axis=0)
            rgb_max = np.amax(rgb, axis=0)
            scaling = np.where(
                rgb_max > 0.0, rgb_max / (rgb_max - rgb_min + 0.00001), np.ones(rgb.shape)
            )
            rgb = np.where(rgb_min < 0.0, scaling * (rgb - rgb_min), rgb)

        if gamma:
            rgb = gamma_correction(rgb, gamma=2.4)

        # reshape back
        return (rgb.T).reshape((vals.shape[0], vals.shape[1], 3))
