import numpy as np
from waveprop.util import sample_points, ft2, jinc


def fraunhofer(u_in, wv, d1, dz):
    """
    Simulate Fraunhofer propagation.

    Listing 4.1 of "Numerical Simulation of Optical Wave Propagation with Examples in MATLAB"
    (2010).

    Parameters
    ----------
    u_in : :py:class:`~numpy.ndarray`
        Input amplitude distribution, [Ny X Nx].
    wv : float
        Wavelength [m].
    d1 : float or list
        Sampling period along x-dimension and (optionally) y-dimennsion [m].
    dz : float
        Propagation distance [m].

    """

    if isinstance(d1, float) or isinstance(d1, int):
        d1 = [d1, d1]
    assert len(d1) == 2

    Ny, Nx = u_in.shape
    k = 2 * np.pi / wv

    # frequencies of propagating waves
    fx, fy = sample_points(N=[Ny, Nx], delta=[(1 / Ny / d1[0]), (1 / Nx / d1[1])])

    # output coordinates
    x2 = fx * wv * dz
    y2 = fy * wv * dz

    # output distribution
    u_out = (
        np.exp(1j * k * dz)
        * np.exp(1j * k / (2 * dz) * (x2**2 + y2**2))
        / (1j * wv * dz)
        * ft2(u_in, d1)
    )

    return u_out, x2, y2


def fraunhofer_prop_circ_ap(wv, dz, diam, x, y):
    """
    Analytic formula for Fraunhofer diffraction pattern of circular aperture.

    From Listing 4.2 of "Numerical Simulation of Optical Wave Propagation with Examples in MATLAB"
    (2010).

    Parameters
    ----------
    wv : float
        Wavelength [m].
    dz : float
        Propagation distance [m].
    diam : float
        Diameter [m].
    x : :py:class:`~numpy.ndarray`
        [1, Nx] array of x-coordinates [m].
    y : :py:class:`~numpy.ndarray`
        [Ny, 1] array of y-coordinates [m].
    """
    k = 2 * np.pi / wv
    return (
        np.exp(1j * k * dz)
        * np.exp(1j * k / (2 * dz) * (x**2 + y**2))
        / (1j * wv * dz)
        * (diam**2 * np.pi / 4)
        * jinc(diam * np.sqrt(x**2 + y**2) / (wv * dz))
    )


def fraunhofer_prop_rect_ap(wv, dz, x, y, lx, ly=None):
    """
    Analytic formula for Fraunhofer diffraction pattern of a rectangular aperture.

    Derivation can be found in Section 4.4.1 of "Introduction to Fourier Optics" (Second Edition).

    Parameters
    ----------
    wv : float
        Wavelength [m].
    dz : float
        Propagation distance [m].
    x : :py:class:`~numpy.ndarray`
        [1 x Nx] array of x-coordinates [m].
    y : :py:class:`~numpy.ndarray`
        [Ny x 1] array of y-coordinates [m].
    lx : float
        Width in x-dimension [m].
    ly : float
        Width in y-dimension [m].

    """
    if ly is None:
        ly = lx
    k = 2 * np.pi / wv
    return (
        np.exp(1j * k * dz)
        * np.exp(1j * k / (2 * dz) * (x**2 + y**2))
        / (1j * wv * dz)
        * lx
        * ly
        * np.sinc(lx * x / wv / dz)
        * np.sinc(ly * y / wv / dz)
    )
