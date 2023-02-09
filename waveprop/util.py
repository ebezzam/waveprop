import numpy as np
import cv2
from numpy.fft import fftshift, fft2, ifftshift, ifft2
import torch
from scipy.special import j1
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torchvision.transforms.functional import crop as crop_torch
from torchvision.transforms.functional import resize as resize_torch
import torch.nn.functional as F


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
    fact = delta[0] * delta[1]
    if torch.is_tensor(g):
        return torch.fft.fftshift(
            torch.fft.fft2(
                # TODO ifftshift of fftshift?
                torch.fft.fftshift(g * fact)
            )
        )
    else:
        res = fftshift(fft2(fftshift(g))) * fact
        if g.dtype == np.float32 or g.dtype == np.complex64:
            res = res.astype(np.complex64)
        return res


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
    fact = G.shape[0] * G.shape[1] * delta_f[0] * delta_f[1]
    # fact = 1   # TODO : huge difference when we don't apply factor
    if torch.is_tensor(G):
        # fact = pt.tensor([fact], dtype=G.dtype)
        return torch.fft.ifftshift(
            torch.fft.ifft2(
                # TODO ifftshift of fftshift?
                torch.fft.ifftshift(G * fact)
            )
        )
        # * G.shape[0] * G.shape[1] * delta_f[0] * delta_f[1]
    else:
        res = ifftshift(ifft2(ifftshift(G * fact)))
        if G.dtype == np.complex64:
            res = res.astype(np.complex64)
        return res


def sample_points(N, delta, shift=0, pytorch=False):
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
    if pytorch:
        delta = torch.tensor(delta)
        shift = torch.tensor(shift)
        x = torch.arange(-N[1] / 2, N[1] / 2) * delta[1] + shift[1]
        x = torch.unsqueeze(x, 0)
        y = torch.arange(-N[0] / 2, N[0] / 2) * delta[0] + shift[0]
        y = torch.unsqueeze(y, 1)
    else:
        x = np.arange(-N[1] / 2, N[1] / 2)[np.newaxis, :] * delta[1] + shift[1]
        y = np.arange(-N[0] / 2, N[0] / 2)[:, np.newaxis] * delta[0] + shift[0]
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
    r = np.sqrt(x**2 + y**2)
    z = (r < diam / 2).astype(float)
    z[r == diam / 2] = 0.5
    return z


def rect(x, D, offset=0):
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
    x = np.abs(x - offset)
    y = (x < D / 2).astype(float)
    y[x == D / 2] = 0.5
    return y


def rect2d(x, y, D, offset=0):
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
    if isinstance(offset, float) or isinstance(offset, int):
        offset = [offset, offset]
    assert len(offset) == 2
    return rect(x, D[1], offset[1]) * rect(y, D[0], offset[0])


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


def plot2d(x_vals, y_vals, Z, pcolormesh=False, colorbar=True, title="", ax=None):
    """
    pcolormesh doesn't keep square aspect ratio for each pixel
    """

    x_vals = x_vals.squeeze()
    y_vals = y_vals.squeeze()
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
        cp = ax.pcolormesh(X, Y, Z, cmap=cm.gray)
    else:
        if len(Z.shape) == 2 or Z.shape[0] == 1:
            cp = ax.imshow(
                Z[0] if Z.shape[0] == 1 else Z,
                extent=[
                    x_vals.min(),
                    x_vals.max(),
                    y_vals.min(),
                    y_vals.max(),
                ],
                cmap="gray",
                origin="lower",
            )
        else:
            cp = ax.imshow(
                np.transpose(Z, (1, 2, 0)) if Z.shape[0] == 3 else Z,
                extent=[
                    x_vals.min(),
                    x_vals.max(),
                    y_vals.min(),
                    y_vals.max(),
                ],
                origin="lower",
            )
    fig = plt.gcf()
    if colorbar and len(Z.shape) == 2:
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


def rect_tiling(N_in, N_out, L, n_tiles, prop_func):
    """
    Assumes:
    - centered around (0, 0).
    - input and output regions are the same

    :param N_in:
    :param N_out:
    :param L:
    :param n_tiles:
    :return:
    """

    d2 = L / N_out

    # compute offsets for each tile
    offsets_1d = (np.arange(n_tiles) + 1) * L / n_tiles
    offsets_1d -= np.mean(offsets_1d)
    offsets = np.stack(np.meshgrid(offsets_1d, offsets_1d), -1).reshape(-1, 2)

    # loop over tiles
    tiles = []
    for out_shift in offsets:
        u_out = prop_func(out_shift=out_shift)
        tiles.append(u_out)

    # combine tiles
    myorder = [0, 3, 6, 1, 4, 7, 2, 5, 8]
    tiles = [tiles[i] for i in myorder]
    u_out = np.array(tiles).reshape(n_tiles, n_tiles, N_in, N_in)
    u_out = np.transpose(u_out, axes=(0, 2, 1, 3))
    u_out = np.concatenate(u_out, axis=0)
    u_out = np.transpose(u_out, axes=(1, 2, 0))
    u_out = np.concatenate(u_out, axis=0).T
    x2, y2 = sample_points(N=N_out, delta=d2)

    return u_out, x2, y2


def gamma_correction(vals, gamma=2.2):
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


def crop(u, shape, topleft=None, center_shift=None):
    """
    Crop center section of array or tensor (default). Otherwise set `topleft`.

    Parameters
    ----------
    u : array or tensor
        Data to crop.
    shape : tuple
        Target shape (Ny, Nx).

    Returns
    -------

    """
    Ny, Nx = shape
    if topleft is None:
        topleft = (int((u.shape[0] - Ny) / 2), int((u.shape[1] - Nx) / 2))
    if center_shift is not None:
        # subtract (positive) on second column to shift to the right
        topleft = (topleft[0] + center_shift[0], topleft[1] + center_shift[1])
    if torch.is_tensor(u):
        if u.dtype == torch.complex64 or u.dtype == torch.complex128:
            u_out_real = crop_torch(u.real, top=topleft[0], left=topleft[1], height=Ny, width=Nx)
            u_out_imag = crop_torch(u.imag, top=topleft[0], left=topleft[1], height=Ny, width=Nx)
            return torch.complex(u_out_real, u_out_imag)
        else:
            return crop_torch(u, top=topleft[0], left=topleft[1], height=Ny, width=Nx)
    else:
        return u[
            topleft[0] : topleft[0] + Ny,
            topleft[1] : topleft[1] + Nx,
        ]


def _get_dtypes(dtype, is_torch):
    if not is_torch:
        if dtype == np.float32 or dtype == np.complex64:
            return np.complex64, np.complex64
        elif dtype == np.float64 or dtype == np.complex128:
            return np.complex128, np.complex128
        else:
            raise ValueError("Unexpected dtype: ", dtype)
    else:
        if dtype == np.float32 or dtype == np.complex64:
            return torch.complex64, np.complex64
        elif dtype == np.float64 or dtype == np.complex128:
            return torch.complex128, np.complex128
        elif dtype == torch.float32 or dtype == torch.complex64:
            return torch.complex64, np.complex64
        elif dtype == torch.float64 or dtype == torch.complex128:
            return torch.complex128, np.complex128
        else:
            raise ValueError("Unexpected dtype: ", dtype)


def zero_pad(u_in, pad=None):
    Ny, Nx = u_in.shape
    if pad is None:
        y_pad_edge = int(Ny // 2)
        x_pad_edge = int(Nx // 2)
    else:
        y_pad_edge, x_pad_edge = pad

    if torch.is_tensor(u_in):
        pad_width = (
            x_pad_edge + 1 if Nx % 2 else x_pad_edge,
            x_pad_edge,
            y_pad_edge + 1 if Ny % 2 else y_pad_edge,
            y_pad_edge,
        )
        return torch.nn.functional.pad(u_in, pad_width, mode="constant", value=0.0)
    else:
        pad_width = (
            (y_pad_edge + 1 if Ny % 2 else y_pad_edge, y_pad_edge),
            (x_pad_edge + 1 if Nx % 2 else x_pad_edge, x_pad_edge),
        )
        return np.pad(u_in, pad_width=pad_width, mode="constant", constant_values=0)


def resize(img, factor=None, shape=None, interpolation=cv2.INTER_CUBIC, axes=(0, 1)):
    """
    Resize by given factor or to a given shape.

    Parameters
    ----------
    img :py:class:`~numpy.ndarray`
        Downsampled image.
    factor : int or float
        Resizing factor.
    shape : tuple
        (Height, width).
    interpolation : OpenCV interpolation method
        See https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#cv2.resize
    Returns
    -------
    img :py:class:`~numpy.ndarray`
        Resized image.
    """
    min_val = img.min()
    max_val = img.max()
    img_shape = np.array([img.shape[_ax] for _ax in axes])

    if shape is None:
        assert factor is not None
        new_shape = tuple((img_shape * factor).astype(int))
        new_shape = new_shape[::-1]
        resized = cv2.resize(img, dsize=new_shape, interpolation=interpolation)
    else:
        if np.array_equal(img_shape, shape[::-1]):
            return img
        resized = cv2.resize(img, dsize=shape[::-1], interpolation=interpolation)
    return np.clip(resized, min_val, max_val)


def realfftconvolve2d(image, kernel):
    """Convolve image with kernel using real FFT.

    Parameters
    ----------
    image : np.ndarray
        Image.
    kernel : np.ndarray
        Kernel.

    Returns
    -------
    np.ndarray
        Convolved image.
    """
    image_shape = np.array(image.shape)

    fft_shape = image_shape + np.array(kernel.shape) - 1

    H = np.fft.rfft2(kernel, s=fft_shape)
    I = np.fft.rfft2(image, s=fft_shape)
    output = np.fft.irfft2(H * I, s=fft_shape)

    # crop out zero padding
    y_pad_edge = int((fft_shape[0] - image_shape[0]) / 2)
    x_pad_edge = int((fft_shape[1] - image_shape[1]) / 2)
    output = output[
        y_pad_edge : y_pad_edge + image_shape[0], x_pad_edge : x_pad_edge + image_shape[1]
    ]
    return output


def prepare_object_plane(
    obj,
    object_height,
    scene2mask,
    mask2sensor,
    sensor_size,
    sensor_dim,
    random_shift=False,
):
    """
    Prepare object plane for convolution with PSF.

    Parameters
    ----------
    obj : np.ndarray
        Input image (HxWx3).
    object_height : float
        Height of object plane in meters.
    scene2mask : float
        Distance from scene to mask in meters.
    mask2sensor : float
        Distance from mask to sensor in meters.
    sensor_size : tuple
        Size of sensor in meters.
    sensor_dim : tuple
        Dimension of sensor in pixels.
    random_shift : bool
        Randomly shift resized obj in its plane.

    Returns
    -------
    np.ndarray
        Object plane.
    """
    if torch.is_tensor(obj):
        axes = (-2, -1)
    else:
        axes = (0, 1)

    # determine object height in pixels
    input_dim = np.array([obj.shape[_ax] for _ax in axes])
    magnification = mask2sensor / scene2mask
    scene_dim = np.array(sensor_size) / magnification
    object_height_pix = int(np.round(object_height / scene_dim[1] * sensor_dim[1]))
    scaling = object_height_pix / input_dim[1]
    object_dim = tuple((np.round(input_dim * scaling)).astype(int))

    if torch.is_tensor(obj):
        object_plane = resize_torch(obj, size=object_dim)
    else:
        object_plane = resize(obj, shape=object_dim)

    # pad object plane to convolution size
    padding = sensor_dim - object_dim
    left = padding[1] // 2
    right = padding[1] - left
    top = padding[0] // 2
    bottom = padding[0] - top

    if top < 0:
        top = 0
        bottom = 0
    if left < 0:
        left = 0
        right = 0

    if torch.is_tensor(obj):
        object_plane = torch.nn.functional.pad(
            object_plane, pad=(left, right, top, bottom), mode="constant", value=0.0
        )

        object_plane_shape = np.array(object_plane.shape[-2:])

    else:
        pad_width = [(0, 0) for _ in range(len(obj.shape))]
        pad_width[axes[0]] = (top, bottom)
        pad_width[axes[1]] = (left, right)
        pad_width = tuple(pad_width)
        object_plane = np.pad(object_plane, pad_width=pad_width, mode="constant")

        object_plane_shape = np.array(object_plane.shape[:2])

    # remove extra pixels if height extended beyond sensor
    if (object_plane_shape != sensor_dim).any():
        object_plane = crop(object_plane, shape=sensor_dim)

    if random_shift:
        hshift = int(np.random.uniform(low=-left, high=right))
        vshift = int(np.random.uniform(low=-bottom, high=top))
        if torch.is_tensor(obj):
            object_plane = torch.roll(object_plane, shifts=(vshift, hshift), dims=axes)
        else:
            object_plane = np.roll(object_plane, shift=hshift, axis=axes[1])
            object_plane = np.roll(object_plane, shift=vshift, axis=axes[0])

    return object_plane
