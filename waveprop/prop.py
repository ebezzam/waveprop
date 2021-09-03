import numpy as np
from scipy.signal import fftconvolve
from waveprop.util import ft2, ift2, jinc, sample_points
from pyffs import ffsn_sample, ffsn, fs_interpn, ffsn_shift


def free_space_impulse_response(k, x, y, z):
    """
    Impulse response of Rayleigh-Sommerfeld.

    Eq 7 of "Fast-Fourier-transform based numerical integration method for the Rayleigh–Sommerfeld
    diffraction formula" (2006).

    Parameters
    ----------
    k : float
        Wavenumber.
    x : :py:class:`~numpy.ndarray`
        [1 x Nx] array of x-coordinates [m].
    y : :py:class:`~numpy.ndarray`
        [Ny x 1] array of y-coordinates [m].
    z : float
        Propagation distance [m].

    """
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    return 1 / (2 * np.pi) * np.exp(1j * k * r) / r * z / r * (1 / r - 1j * k)


def direct_integration(u_in, wv, d1, dz, x, y):
    """
    Very expensive, brute force approach. But without artifacts of DFT, namely (1) circular
    convolution and (2) requiring bandlimited sequences.

    Eq 9 of "Fast-Fourier-transform based numerical integration method for the Rayleigh–Sommerfeld
    diffraction formula" (2006).

    TODO : add Simpson / Trapezoidal rule

    Parameters
    ----------
    u_in : array_like
        Input amplitude distribution, [Ny, Nx].
    wv : float
        Wavelength [m].
    d1 : float
        Input sampling period for both x-dimension and y-dimension [m].
    x : array_like
        [1 x Nx] array of output x-coordinates [m].
    y : array_like
        [Ny x 1] array of output y-coordinates [m].
    dz : float
        Propagation distance [m].

    """
    if isinstance(d1, float) or isinstance(d1, int):
        d1 = [d1, d1]
    assert len(d1) == 2

    N = u_in.shape
    k = 2 * np.pi / wv

    # source coordinates
    x1, y1 = sample_points(N=N, delta=d1)

    # brute force convolution
    u_out = np.zeros((len(y), len(x)), dtype=complex)
    for i, xm in enumerate(x):
        for j, ym in enumerate(y):
            G = free_space_impulse_response(k, xm - x1, ym - y1, dz)
            tmp = np.multiply(G, u_in)
            u_out[j, i] = np.sum(tmp) * d1[0] * d1[1]
    return u_out


def fft_di(u_in, wv, d1, dz, N_out=None, use_simpson=True):
    """
    Enforces same resolution between input and output.

    Parameters
    ----------
    u_in : array_like
        Input amplitude distribution, [Ny, Nx].
    wv : float
        Wavelength [m].
    d1 : float
        Input sampling period for both x-dimension and y-dimension [m].
    dz : float
        Propagation distance [m].
    N_out : int or list or tuple, optional
        Number of output samples, also determines the size of the output window. Default is same
        number of points, and therefore same area, as the input.
    use_simpson : bool, optional
        Whether to use Simpson's rule to improve calculation accuracy. Note that Simpson's rule can
        only be applied if the dimension is odd. If the dimension is even, trapezoid rule will be
        used instead.

    """
    if isinstance(d1, float) or isinstance(d1, int):
        d1 = [d1, d1]
    assert len(d1) == 2
    if N_out is None:
        N_out = [u_in.shape[1], u_in.shape[0]]
    if isinstance(N_out, float) or isinstance(N_out, int):
        N_out = [N_out, N_out]
    assert len(N_out) == 2

    Ny, Nx = u_in.shape
    k = 2 * np.pi / wv

    # output coordinates
    x2, y2 = sample_points(N=N_out, delta=d1)

    # source coordinates
    x1, y1 = sample_points(N=[Nx, Ny], delta=d1)

    # zero-pad
    Nx_out = N_out[0]
    Ny_out = N_out[1]
    u_in_pad = np.zeros((Ny_out + Ny - 1, Nx_out + Nx - 1))

    if use_simpson:
        # Equation 17
        Bx = np.ones((1, Nx))
        if Nx % 2:
            Bx[0, 1::2] += 3
            Bx[0, 2::2] += 1
            Bx[0, -1] = 1
            Bx /= 3
        else:
            # trapezoidal rule
            Bx[0, 0] = 0.5
            Bx[0, -1] = 0.5

        By = np.ones((Ny, 1))
        if Ny % 2:
            By[1::2] += 3
            By[2::2] += 1
            By[-1] = 1
            By /= 3
        else:
            # trapezoidal rule
            By[0] = 0.5
            By[-1] = 0.5
        W = By @ Bx
        u_in_pad[:Ny, :Nx] = u_in * W
    else:
        u_in_pad[:Ny, :Nx] = u_in

    # compute spatial response, Eq 12
    x1 = np.squeeze(x1)
    y1 = np.squeeze(y1)
    # -- prepare X coord, Eq 13
    X = np.zeros(Nx_out + Nx - 1)
    xin_rev = x1[::-1]
    X[: Nx - 1] = x2[0, 0] - xin_rev[: Nx - 1]
    X[Nx - 1 :] = np.squeeze(x2) - x1[0]
    X = X[np.newaxis, :]
    # -- prepare Y coord, Eq 14
    Y = np.zeros(Ny_out + Ny - 1)
    yin_rev = y1[::-1]
    Y[: Ny - 1] = y2[0, 0] - yin_rev[: Ny - 1]
    Y[Ny - 1 :] = np.squeeze(y2) - y1[0]
    Y = Y[:, np.newaxis]
    # -- get impulse response matrix
    H = free_space_impulse_response(k, X, Y, dz)

    # Eq 10
    S = np.fft.ifft2(np.fft.fft2(u_in_pad) * np.fft.fft2(H)) * d1[0] * d1[1]

    # lower right submatrix
    return S[-Ny_out:, -Nx_out:], x2, y2


def angular_spectrum_ffs(u_in, wv, d1, dz, d2=None, N_out=None, out_shift=0):
    """
    Can control output sampling like shifted Fresnel, but maybe only within period?

    Exactness of Angular spectrum.

    With Fresnel could go output of input region. Here limited by input region?
    """

    if isinstance(d1, float) or isinstance(d1, int):
        d1 = [d1, d1]
    assert len(d1) == 2
    if d2 is None:
        d2 = d1
    if isinstance(d2, float) or isinstance(d2, int):
        d2 = [d2, d2]
    assert len(d2) == 2
    Ny, Nx = u_in.shape
    if N_out is None:
        N_out = [Nx, Ny]
    if isinstance(N_out, int):
        N_out = [N_out, N_out]
    assert len(N_out) == 2

    # output coordinates
    x2, y2 = sample_points(N=N_out, delta=d2, shift=out_shift)

    # determine necessary padding
    x_out_max = np.max(x2)
    y_out_max = np.max(y2)
    x_out_min = np.min(x2)
    y_out_min = np.min(y2)

    # padding_x = max(int(np.ceil(x_out_max * 2 / d1[0])) - Nx, Nx)
    # padding_y = max(int(np.ceil(y_out_max * 2 / d1[1])) - Ny, Ny)
    # x_pad_edge = int(np.ceil(padding_x / 2.0))
    # y_pad_edge = int(np.ceil(padding_y / 2.0))
    # pad_width = ((y_pad_edge, y_pad_edge), (x_pad_edge, x_pad_edge))

    x_pad_neg = int(np.ceil(max(x_out_min / d1[0] - Nx / 2, Nx / 2)))
    x_pad_pos = int(np.ceil(max(x_out_max / d1[0] - Nx / 2, Nx / 2)))
    y_pad_neg = int(np.ceil(max(y_out_min / d1[1] - Ny / 2, Ny / 2)))
    y_pad_pos = int(np.ceil(max(y_out_max / d1[1] - Ny / 2, Ny / 2)))
    pad_width = ((y_pad_neg, y_pad_pos), (x_pad_neg, x_pad_pos))

    #
    # import pudb; pudb.set_trace()

    u_in_pad = np.pad(u_in, pad_width=pad_width, mode="constant", constant_values=0)

    # size of the field
    Ny_pad, Nx_pad = u_in_pad.shape
    Dy, Dx = (d1[1] * float(Ny_pad), d1[0] * float(Nx_pad))

    # compute FS coefficients of input
    # -- reshuffle input for pyFFS
    T = [Dx, Dy]
    T_c = [0, 0]
    N_s = np.array(u_in_pad.shape)
    N_FS = N_s // 2 * 2 - 1  # must be odd
    samp_loc, idx = ffsn_sample(T, N_FS, T_c, N_s)
    u_in_pad_reorder = ffsn_shift(u_in_pad, idx)
    # -- compute coefficients
    U = ffsn(u_in_pad_reorder, T, T_c, N_FS)[: N_FS[1], : N_FS[0]]

    # compute truncated FS coefficients of response
    # -- get corresponding frequencies
    fX = np.arange(-N_FS[0] / 2, N_FS[0] / 2)[np.newaxis, :] / T[0]
    fY = np.arange(-N_FS[1] / 2, N_FS[1] / 2)[:, np.newaxis] / T[1]
    fsq = fX ** 2 + fY ** 2
    # -- compute response
    k = 2 * np.pi / wv
    wv_sq = wv ** 2
    H = np.zeros_like(U).astype(complex)
    prop_waves = fsq <= 1 / wv_sq
    evanescent_waves = np.logical_not(prop_waves)
    H[prop_waves] = np.exp(1j * k * dz * np.sqrt(1 - wv_sq * fsq[prop_waves]))
    H[evanescent_waves] = np.exp(-k * dz * np.sqrt(wv_sq * fsq[evanescent_waves] - 1))
    # -- bandlimit to avoid aliasing
    fx_max = 1 / np.sqrt((2 * dz * (1 / Dx)) ** 2 + 1) / wv
    fy_max = 1 / np.sqrt((2 * dz * (1 / Dy)) ** 2 + 1) / wv
    H_filter = (np.abs(fX) <= fx_max) * (np.abs(fY) < fy_max)
    H *= H_filter

    # use output FS coefficients to interpolate
    output_FS = U * H
    a = [np.min(x2), np.min(y2)]
    b = [np.max(x2), np.max(y2)]
    u_out = fs_interpn(x_FS=output_FS, T=T, a=a, b=b, M=N_out)

    return u_out, x2, y2


def angular_spectrum(u_in, wv, d1, dz, bandlimit=True, out_shift=0, d2=None, N_out=None):
    """
    Band-Limited Angular Spectrum Method for Numerical Simulation of Free-Space Propagation in Far
    and Near Fields (2009)

    TODO : set data type
    TODO : set output sampling
    TODO : padding optional

    Parameters
    ----------
    u_in : :py:class:`~numpy.ndarray`
        Input amplitude distribution, [Ny, Nx].
    wv : float
        Wavelength [m].
    d1 : float or list or tuple
        Input sampling period for both x-dimension and y-dimension [m]. Scalar if the same for both
        dimensions.
    dz : float
        Propagation distance [m].
    bandlimit : bool
        Whether to bandlimit propagation in order to avoid aliasing, as proposed in "Band-Limited
        Angular Spectrum Method for Numerical Simulation of Free-Space Propagation in Far and Near
        Fields" (2009).
    out_shift : array_like
        Shift from optical axis at output, as proposed in "Shifted angular spectrum method for
        off-axis numerical propagation" (2010).
    d2 : float or list or tuple, optional
        Output sampling period for both x-dimension and y-dimension [m]. Scalar if the same for both
        dimensions. Rescale, as proposed in "Band-limited angular spectrum numerical propagation
        method with selective scaling of observation window size and sample number" (2012). Default
        is to use same sampling period as input.
    N_out : int or list or tuple, optional
        Number of output samples for x-dimension and y-dimensions. Scalar if the same for both
        dimensions. Rescale, as proposed in "Band-limited angular spectrum numerical propagation
        method with selective scaling of observation window size and sample number" (2012). Default
        is to use same sampling period as input.
    """
    if isinstance(d1, float) or isinstance(d1, int):
        d1 = [d1, d1]
    assert len(d1) == 2
    if d2 is not None:
        if isinstance(d2, float) or isinstance(d2, int):
            d2 = [d2, d2]
        assert len(d2) == 2
    if N_out is not None:
        assert d2 is not None
        if isinstance(N_out, int):
            N_out = [N_out, N_out]
        assert len(N_out) == 2
        assert [isinstance(val, int) for val in N_out]
    if isinstance(out_shift, float) or isinstance(out_shift, int):
        out_shift = [out_shift, out_shift]
    assert len(out_shift) == 2

    # zero pad to simulate linear convolution
    Ny, Nx = u_in.shape
    Sx = Nx * d1[0]
    Sy = Ny * d1[1]

    u_in_pad = np.zeros((2 * Ny, 2 * Nx), dtype=u_in.dtype)
    u_in_pad[:Ny, :Nx] = u_in

    # y_pad_edge = int(np.ceil(Ny / 2.0))
    # x_pad_edge = int(np.ceil(Nx / 2.0))
    # pad_width = ((y_pad_edge, y_pad_edge), (x_pad_edge, x_pad_edge))
    # u_in_pad = np.pad(u_in, pad_width=pad_width, mode="constant", constant_values=0)

    # size of the padded field
    Ny_pad, Nx_pad = u_in_pad.shape
    Dy, Dx = (d1[1] * float(Ny_pad), d1[0] * float(Nx_pad))

    # frequency coordinates sampling, TODO check (commented is from neural holography)
    # neural holography is probably wrong if you compare with fftfreq
    # fX = np.linspace(-1 / (2 * dx) + 0.5 / (2 * Dx), 1 / (2 * dx) - 0.5 / (2 * Dx), Nx)[:, np.newaxis].T
    # fY = np.linspace(-1 / (2 * dy) + 0.5 / (2 * Dy), 1 / (2 * dy) - 0.5 / (2 * Dy), Ny)[:, np.newaxis]
    dfX = 1.0 / Dx
    dfY = 1.0 / Dy
    fX = np.arange(-Nx_pad / 2, Nx_pad / 2)[np.newaxis, :] * dfX
    fY = np.arange(-Ny_pad / 2, Ny_pad / 2)[:, np.newaxis] * dfY
    fsq = fX ** 2 + fY ** 2

    # compute transfer function (Saleh / Sepand's notes but w/o abs val on distance)
    k = 2 * np.pi / wv
    wv_sq = wv ** 2
    H = np.zeros_like(u_in_pad).astype(complex)
    prop_waves = fsq <= 1 / wv_sq
    evanescent_waves = np.logical_not(prop_waves)
    H[prop_waves] = np.exp(1j * k * dz * np.sqrt(1 - wv_sq * fsq[prop_waves]))
    H[evanescent_waves] = np.exp(
        -k * dz * np.sqrt(wv_sq * fsq[evanescent_waves] - 1)
    )  # evanescent waves

    # shift
    if out_shift[0] or out_shift[1]:
        # Eq 7 of Matsushima (2010)
        H *= np.exp(1j * 2 * np.pi * (out_shift[0] * fX + out_shift[1] * fY))

    # band-limited to avoid aliasing
    # - Eq 13 and 20 of Matsushima et al. (2009)
    # - Table 1 of Matsushima (2010) for generalization to off-axis
    if bandlimit:
        u0, u_width, v0, v_width = bandpass(
            Sx=Sx, Sy=Sy, x0=out_shift[0], y0=out_shift[1], z0=dz, wv=wv
        )
        fx_max = u_width / 2
        fy_max = v_width / 2
        H_filter = (np.abs(fX - u0) <= fx_max) * (np.abs(fY - v0) < fy_max)
        H *= H_filter

    # perform convolution, TODO : bad FFT shifting in neural holography code?
    # U1 = np.fft.fftn(np.fft.ifftshift(Uin_pad), axes=(-2, -1), norm='ortho')
    U1 = ft2(u_in_pad, delta=d1)
    U2 = H * U1

    if d2 is None:
        # no rescaling
        # Uout = np.fft.fftshift(np.fft.ifftn(U2, axes=(-2, -1), norm='ortho'))
        u_out = ift2(U2, delta_f=[dfX, dfY])

        # remove padding
        u_out = u_out[:Ny, :Nx]

        # coordinates
        x2, y2 = sample_points(N=[Nx, Ny], delta=d1, shift=out_shift)
    else:

        if N_out is None:
            N_out = [Nx, Ny]

        # coordinates
        x2, y2 = sample_points(N=N_out, delta=d2, shift=out_shift)
        alpha_x = d2[0] / dfX
        alpha_y = d2[1] / dfY

        # Eq 9 of "Band-limited angular spectrum numerical propagation method with selective scaling
        # of observation window size and sample number" (2012)
        u_out = (
            np.exp(1j * np.pi / alpha_x * x2 ** 2)
            * d2[0]
            * np.exp(1j * np.pi / alpha_y * y2 ** 2)
            * d2[1]
        )

        fX_scaled = alpha_x * fX
        fY_scaled = alpha_y * fY
        B = (
            (1 / alpha_x)
            * (1 / alpha_y)
            * U2
            * np.exp(1j * np.pi / alpha_x * fX_scaled ** 2)
            * np.exp(1j * np.pi / alpha_y * fY_scaled ** 2)
        )
        f = np.exp(-1j * np.pi / alpha_x * fX_scaled ** 2) * np.exp(
            -1j * np.pi / alpha_y * fY_scaled ** 2
        )

        # tmp = fftconvolve(B, f, mode="same")
        tmp = np.fft.ifft2(np.fft.fft2(B) * np.fft.fft2(f))

        import pudb

        pudb.set_trace()

        # import matplotlib.pyplot as plt
        # X, Y = np.meshgrid(np.arange(tmp.shape[0]), np.arange(tmp.shape[1]))
        # plt.pcolormesh(X, Y, np.abs(tmp))
        # plt.pcolormesh(fX_scaled, fY_scaled, np.abs(B))
        # plt.pcolormesh(fX_scaled, fY_scaled, np.abs(f))

        # import pudb
        # pudb.set_trace()

        u_out *= tmp[
            int(Nx - N_out[0] / 2) : int(Nx + N_out[0] / 2),
            int(Ny - N_out[1] / 2) : int(Ny + N_out[1] / 2),
        ]

    return u_out, x2, y2


def bandpass(Sx, Sy, x0, y0, z0, wv):
    """
    Table 1 of "Shifted angular spectrum method for off-axis numerical propagation" (2010).

    :param Sx:
    :param Sy:
    :param x0:
    :param y0:
    :return:
    """

    du = 1 / (2 * Sx)
    u_limit_p = ((x0 + 1 / (2 * du)) ** (-2) * z0 ** 2 + 1) ** (-1 / 2) / wv
    u_limit_n = ((x0 - 1 / (2 * du)) ** (-2) * z0 ** 2 + 1) ** (-1 / 2) / wv
    if Sx < x0:
        u0 = (u_limit_p + u_limit_n) / 2
        u_width = u_limit_p - u_limit_n
    elif x0 <= -Sx:
        u0 = -(u_limit_p + u_limit_n) / 2
        u_width = u_limit_n - u_limit_p
    else:
        u0 = (u_limit_p - u_limit_n) / 2
        u_width = u_limit_p + u_limit_n

    dv = 1 / (2 * Sy)
    v_limit_p = ((y0 + 1 / (2 * dv)) ** (-2) * z0 ** 2 + 1) ** (-1 / 2) / wv
    v_limit_n = ((y0 - 1 / (2 * dv)) ** (-2) * z0 ** 2 + 1) ** (-1 / 2) / wv
    if Sy < y0:
        v0 = (v_limit_p + v_limit_n) / 2
        v_width = v_limit_p - v_limit_n
    elif y0 <= -Sy:
        v0 = -(v_limit_p + v_limit_n) / 2
        v_width = v_limit_n - v_limit_p
    else:
        v0 = (v_limit_p - v_limit_n) / 2
        v_width = v_limit_p + v_limit_n

    return u0, u_width, v0, v_width
