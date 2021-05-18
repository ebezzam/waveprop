import numpy as np
from numpy.fft import fftshift, fft2, ifftshift, ifft2
from scipy.special import j1


def ft2(g, delta):
    """
    Compute 2D DFT.

    Inspired by Listing 2.5 of "Numerical Simulation of Optical Wave Propagation with Examples in
    MATLAB" (2010).

    Parameters
    ----------
    g : :py:class:`~numpy.ndarray`
        2D input samples.
    delta : float or list
        Sampling period along x-dimension and (optionally) y-dimension [m].
    """
    if isinstance(delta, float) or isinstance(delta, int):
        delta = [delta, delta]
    assert len(delta) == 2
    return fftshift(fft2(fftshift(g))) * delta[0] * delta[1]


def ift2(G, delta_f):
    """
    Compute 2D IDFT.

    Inspired by Listing 2.6 from "Numerical Simulation of Optical Wave Propagation with Examples in
    MATLAB" (2010).

    Parameters
    ----------
    g : :py:class:`~numpy.ndarray`
        2D input samples.
    delta_f : float or list
        Frequency interval along x-dimension and (optionally) y-dimension [Hz].
    """
    if isinstance(delta_f, float) or isinstance(delta_f, int):
        delta_f = [delta_f, delta_f]
    assert len(delta_f) == 2
    return ifftshift(ifft2(ifftshift(G))) * G.shape[0] * G.shape[1] * delta_f[0] * delta_f[1]


def sample_points(N, delta):
    """
    Return sample points in 2D.

    Parameters
    ----------
    N : int or float or list
        Number of sample points
    delta: int or float or list
        Sampling period along x-dimension and (optionally) y-dimension [m].
    """
    if isinstance(N, int):
        N = [N, N]
    assert len(N) == 2
    if isinstance(delta, float) or isinstance(delta, int):
        delta = [delta, delta]
    assert len(delta) == 2
    x = np.arange(-N[0] / 2, N[0] / 2)[np.newaxis, :] * delta[0]
    y = np.arange(-N[1] / 2, N[1] / 2)[:, np.newaxis] * delta[1]
    return x, y


def circ(x, y, diam):
    """
    Sample circle aperture.

    Listing B.3 of "Numerical Simulation of Optical Wave Propagation with Examples in MATLAB"
    (2010).

    Parameters
    ----------
    x : :py:class:`~numpy.ndarray`
        [1 x Nx] array of x-coordinates [m].
    y : :py:class:`~numpy.ndarray`
        [Ny x 1] array of y-coordinates [m].
    diam : float
        Diameter [m].
    """
    r = np.sqrt(x ** 2 + y ** 2)
    z = (r < diam / 2).astype(float)
    z[r == diam / 2] = 0.5
    return z


def rect(x, D=1):
    """
    Sample 1D rectangular function.

    Listing B.1 of "Numerical Simulation of Optical Wave Propagation with Examples in MATLAB"
    (2010).

    Parameters
    ----------
    x : :py:class:`~numpy.ndarray`
        Sample coordinates [m].
    D : float
        Width of rectangular function.

    """
    x = np.abs(x)
    y = (x < D / 2).astype(float)
    y[x == D / 2] = 0.5
    return y


def jinc(x):
    """
    Sample Jinc function.

    Listing B.4 of "Numerical Simulation of Optical Wave Propagation with Examples in MATLAB"
    (2010).

    Parameters
    ----------
    x : :py:class:`~numpy.ndarray`
        Sample coordinates [m].
    """
    y = np.ones_like(x)
    idx = x != 0
    y[idx] = 2.0 * j1(np.pi * x[idx]) / (np.pi * x[idx])
    return y
