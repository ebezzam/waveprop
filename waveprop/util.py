import numpy as np
from numpy.fft import fftshift, fft2, ifftshift, ifft2
from scipy.special import j1
import matplotlib.pyplot as plt


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


def sample_points(N, delta, shift=0):
    """
    Return sample points in 2D.

    Parameters
    ----------
    N : int or list
        Number of sample points
    delta: int or float or list
        Sampling period along x-dimension and (optionally) y-dimension [m].
    shift : int or float or list
        Shift from optical axis
    """
    if isinstance(N, int):
        N = [N, N]
    assert len(N) == 2
    if isinstance(delta, float) or isinstance(delta, int):
        delta = [delta, delta]
    assert len(delta) == 2
    if isinstance(shift, float) or isinstance(shift, int):
        shift = [shift, shift]
    assert len(shift) == 2
    x = np.arange(-N[0] / 2, N[0] / 2)[np.newaxis, :] * delta[0] + shift[0]
    y = np.arange(-N[1] / 2, N[1] / 2)[:, np.newaxis] * delta[1] + shift[1]
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


def rect(x, D):
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


def rect2d(x, y, D):
    """
    Sample 2D rectangular function.

    Parameters
    ----------
    x : :py:class:`~numpy.ndarray`
        Sample coordinates [m].
    D : float or int or list
        Width of rectangular function. Scalar can be provided for square aperture.

    """
    if isinstance(D, float) or isinstance(D, int):
        D = [D, D]
    assert len(D) == 2
    return rect(x, D[0]) * rect(y, D[1])


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


def plot2d(x_vals, y_vals, Z, pcolormesh=True, colorbar=True, title="", ax=None):

    if pcolormesh:
        # define corners of mesh
        dx = x_vals[1] - x_vals[0]
        x_vals -= dx / 2
        x_vals = np.append(x_vals, [x_vals[-1] + dx])

        dy = y_vals[1] - y_vals[0]
        y_vals -= dy / 2
        y_vals = np.append(y_vals, [y_vals[-1] + dy])

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
    X, Y = np.meshgrid(x_vals, y_vals)
    if pcolormesh:
        cp = ax.pcolormesh(X, Y, Z.T)
    else:
        cp = ax.contourf(X, Y, Z.T)
    fig = plt.gcf()
    if colorbar:
        fig.colorbar(cp, ax=ax, orientation="vertical")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    return ax


def bounding_box(ax, start, stop, period, shift=None, pcolormesh=True, **kwargs):
    assert len(start) == 2
    assert len(stop) == 2
    if isinstance(period, int) or isinstance(period, float):
        period = [period, period]
    assert len(period) == 2
    if pcolormesh:
        assert shift is not None
        if isinstance(shift, int) or isinstance(shift, float):
            shift = [shift, shift]
        assert len(shift) == 2
    if shift is None:
        shift = [0, 0]
    ax.axvline(
        x=start[0] - shift[0],
        ymin=0.5 + (start[1] - shift[1]) / period[1],
        ymax=0.5 + (stop[1] - shift[1]) / period[1],
        **kwargs
    )
    ax.axvline(
        x=stop[0] - shift[0],
        ymin=0.5 + (start[1] - shift[1]) / period[1],
        ymax=0.5 + (stop[1] - shift[1]) / period[1],
        **kwargs
    )
    ax.axhline(
        y=start[1] - shift[1],
        xmin=0.5 + (start[0] - shift[0]) / period[0],
        xmax=0.5 + (stop[0] - shift[0]) / period[0],
        **kwargs
    )
    ax.axhline(
        y=stop[1] - shift[1],
        xmin=0.5 + (start[0] - shift[0]) / period[0],
        xmax=0.5 + (stop[0] - shift[0]) / period[0],
        **kwargs
    )
