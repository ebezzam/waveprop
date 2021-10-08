"""
Largely inspired by: https://github.com/rafael-fuente/Diffraction-Simulations--Angular-Spectrum-Method/blob/main/diffractsim/colour_functions.py

"""

import numpy as np
from scipy import interpolate
from pathlib import Path


class ColorSystem:
    def __init__(self, n_wavelength, color_mapping_txt=None, illumination_txt=None):
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

        if n_wavelength == len(lookup_wavelength):
            self.wv = lookup_wavelength
            self.cie_xyz = cmf[:, 1:]
        else:
            self.wv = np.linspace(start=min_wv, stop=max_wv, num=n_wavelength)
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

    def gamma_correction(self, vals, gamma=2.2):
        """
        Tutorials
        - https://www.cambridgeincolour.com/tutorials/gamma-correction.htm
        - (code, for images) https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
            https://lindevs.com/apply-gamma-correction-to-an-image-using-opencv/
        - (code) http://www.fourmilab.ch/documents/specrend/specrend.c

        Parameters
        ----------
        vals : array_like
            RGB values to gamma correct.

        Returns
        -------

        """

        # simple approach
        # return np.power(vals, 1 / gamma)

        # Rec. 709 gamma correction
        # http://www.fourmilab.ch/documents/specrend/specrend.c
        cc = 0.018
        inv_gam = 1 / gamma
        clip_val = (1.099 * np.power(cc, inv_gam) - 0.099) / cc
        return np.where(vals < cc, vals * clip_val, 1.099 * np.power(vals, inv_gam) - 0.099)

        ## source: https://github.com/rafael-fuente/Diffraction-Simulations--Angular-Spectrum-Method/blob/5e82083831acb5729550360c5295447dddb77ca5/diffractsim/colour_functions.py#L93
        # vals = np.where(
        #     vals <= 0.00304,
        #     12.92 * vals,
        #     1.055 * np.power(vals, 1.0 / 2.4) - 0.055,
        # )
        # rgb_max = np.amax(vals, axis=0) + 0.00001  # avoid division by zero
        # intensity_cutoff = 1.0
        # return np.where(rgb_max > intensity_cutoff, vals * intensity_cutoff / (rgb_max), vals)
