import numpy as np
from util import ft2, ift2, jinc, sample_points
from scipy.special import fresnel


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
    fx, fy = sample_points(N=[Nx, Ny], delta=[(1 / Nx / d1[0]), (1 / Ny / d1[1])])

    # output coordinates
    x2 = fx * wv * dz
    y2 = fy * wv * dz

    # output distribution
    u_out = (
        np.exp(1j * k * dz)
        * np.exp(1j * k / (2 * dz) * (x2 ** 2 + y2 ** 2))
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
        * np.exp(1j * k / (2 * dz) * (x ** 2 + y ** 2))
        / (1j * wv * dz)
        * (diam ** 2 * np.pi / 4)
        * jinc(diam * np.sqrt(x ** 2 + y ** 2) / (wv * dz))
    )


def fraunhofer_prop_rect_ap(wv, dz, x, y, lx, ly):
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
    k = 2 * np.pi / wv
    return (
        np.exp(1j * k * dz)
        * np.exp(1j * k / (2 * dz) * (x ** 2 + y ** 2))
        / (1j * wv * dz)
        * lx
        * ly
        * np.sinc(lx * x / wv / dz)
        * np.sinc(ly * y / wv / dz)
    )


def fresnel_one_step(u_in, wv, d1, dz):
    """
    Fastest approach for Fresnel numerical computation (single FFT) but no control over output
    sampling.

    Listing 6.1 of "Numerical Simulation of Optical Wave Propagation with Examples in MATLAB"
    (2010).

    Parameters
    ----------
    u_in : :py:class:`~numpy.ndarray`
        Input amplitude distribution, [Ny, Nx].
    wv : float
        Wavelength [m].
    d1 : float or list
        Input sampling period, x-dimension and y-dimension (if different) [m].
    dz : float
        Propagation distance [m].
    """
    if isinstance(d1, float) or isinstance(d1, int):
        d1 = [d1, d1]
    assert len(d1) == 2

    Ny, Nx = u_in.shape
    k = 2 * np.pi / wv

    # coordinates
    x1, y1 = sample_points(N=[Nx, Ny], delta=d1)
    x2, y2 = sample_points(N=[Nx, Ny], delta=[1 / Nx / d1[0] * wv * dz, 1 / Ny / d1[1] * wv * dz])

    # evaluate integral
    u_out = (
        np.exp(1j * k * dz)
        / (1j * wv * dz)
        * np.exp(1j * k / (2 * dz) * (x2 ** 2 + y2 ** 2))
        * ft2(u_in * np.exp(1j * k / (2 * dz) * (x1 ** 2 + y1 ** 2)), delta=d1)
    )

    return u_out, x2, y2


def fresnel_two_step(u_in, wv, d1, d2, dz):
    """
    Fresnel numerical computation that gives control over output sampling but at a higher cost of
    two FFTs.

    Listing 6.3 of "Numerical Simulation of Optical Wave Propagation with Examples in MATLAB"
    (2010).

    NB: only works for square sampling, as non-square would result in different magnification
    factors. Moreover d1 != d2. For d1 == d2, use `fresnel_conv`.

    Parameters
    ----------
    u_in : :py:class:`~numpy.ndarray`
        Input amplitude distribution, [Ny, Nx].
    wv : float
        Wavelength [m].
    d1 : float
        Input sampling period for both x-dimension and y-dimension [m].
    d2 : float or list
        Desired output sampling period for both x-dimension and y-dimension [m].
    dz : float
        Propagation distance [m].
    """
    if d1 == d2:
        raise ValueError("Cannot have d1=d2, use `fresnel_conv` instead.")

    N = np.array(u_in.shape)

    # magnification
    m = d2 / d1

    # intermediate plane
    dz1 = dz / (1 - m)
    u_itm, _, _ = fresnel_one_step(u_in, wv, d1, dz1)
    d1a = wv * abs(dz1) / (N * d1)

    # observation plane
    dz2 = dz - dz1
    return fresnel_one_step(u_itm, wv, d1a, dz2)


def fresnel_conv(u_in, wv, d1, d2, dz):
    """
    Fresnel numerical computation (through convolution perspective) that gives control over output
    sampling but at a higher cost of two FFTs.

    Listing 6.5 of "Numerical Simulation of Optical Wave Propagation with Examples in MATLAB"
    (2010).

    NB: only works for square sampling, as non-square would result in different magnification
    factors.

    Parameters
    ----------
    u_in : :py:class:`~numpy.ndarray`
        Input amplitude distribution, [Ny, Nx].
    wv : float
        Wavelength [m].
    d1 : float
        Input sampling period for both x-dimension and y-dimension [m].
    d2 : float or list
        Desired output sampling period for both x-dimension and y-dimension [m].
    dz : float
        Propagation distance [m].

    """
    N = np.array(u_in.shape)
    k = 2 * np.pi / wv

    # source coordinates
    x1, y1 = sample_points(N=N, delta=d1)
    r1sq = x1 ** 2 + y1 ** 2

    # source spatial frequencies
    df1 = 1 / (N * d1)
    fX, fY = sample_points(N=N, delta=df1)
    fsq = fX ** 2 + fY ** 2

    # scaling parameter
    m = d2 / d1

    # observation plane
    x2, y2 = sample_points(N=N, delta=d2)
    r2sq = x2 ** 2 + y2 ** 2

    # quadratic phase factors
    Q1 = np.exp(1j * k / 2 * (1 - m) / dz * r1sq)
    Q2 = np.exp(-1j * np.pi ** 2 * 2 * dz / m / k * fsq)
    Q3 = np.exp(1j * k / 2 * (m - 1) / (m * dz) * r2sq)

    # propagated field
    u_out = Q3 * ift2(Q2 * ft2(Q1 * u_in / m, delta=d1), delta_f=df1)

    return u_out, x2, y2


def fresnel_prop_square_ap(x, y, width, wv, dz):
    """
    Analytically evaluate Fresnel diffraction pattern of square aperture.

    Listing B.5 (Eq. 1.60) of "Numerical Simulation of Optical Wave Propagation with Examples in
    MATLAB" (2010).

    Derivation can be found in Section 4.5.1 of "Introduction to Fourier Optics" (Second Edition).

    TODO : Generalize to rect

    Parameters
    ----------
    x : :py:class:`~numpy.ndarray`
        [1 x Nx] array of x-coordinates [m].
    y : :py:class:`~numpy.ndarray`
        [Ny x 1] array of y-coordinates [m].
    wv : float
        Wavelength [m].
    width : float
        Width of aperture along x- and y-dimension [m].
    dz : float
        Propagation distance [m].

    """

    k = 2 * np.pi / wv
    N_F = (width / 2) ** 2 / (wv * dz)

    bigX = x / np.sqrt(wv * dz)
    bigY = y / np.sqrt(wv * dz)
    alpha1 = -np.sqrt(2) * (np.sqrt(N_F) + bigX)
    alpha2 = np.sqrt(2) * (np.sqrt(N_F) - bigX)
    beta1 = -np.sqrt(2) * (np.sqrt(N_F) + bigY)
    beta2 = np.sqrt(2) * (np.sqrt(N_F) - bigY)

    # Fresnel sine and cosine integrals
    sa1, ca1 = fresnel(alpha1)
    sa2, ca2 = fresnel(alpha2)
    sb1, cb1 = fresnel(beta1)
    sb2, cb2 = fresnel(beta2)

    # observation-plane field, Eq. 1.60
    return (
        np.exp(1j * k * dz)
        / (2 * 1j)
        * ((ca2 - ca1) + 1j * (sa2 - sa1))
        * ((cb2 - cb1) + 1j * (sb2 - sb1))
    )


def free_space_impulse_response(k, x, y, z):
    """
    Impulse response of angular spectrum method.

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

    Parameters
    ----------
    u_in : n:py:class:`~numpy.ndarray`
        Input amplitude distribution, [Ny, Nx].
    wv : float
        Wavelength [m].
    d1 : float
        Input sampling period for both x-dimension and y-dimension [m].
    x : :py:class:`~numpy.ndarray`
        [1 x Nx] array of x-coordinates [m].
    y : :py:class:`~numpy.ndarray`
        [Ny x 1] array of y-coordinates [m].
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


def angular_spectrum(u_in, wv, delta, dz, bandlimit=True):
    """
    Band-Limited Angular Spectrum Method for Numerical Simulation of Free-Space Propagation in Far
    and Near Fields (2009)

    Code inspiration: https://github.com/computational-imaging/neural-holography/blob/d2e399014aa80844edffd98bca34d2df80a69c84/propagation_ASM.py#L22

    TODO : set data type
    TODO : set output sampling
    TODO : padding optional

    Parameters
    ----------
    u_in : :py:class:`~numpy.ndarray`
        Input amplitude distribution, [Ny, Nx].
    wv : float
        Wavelength [m].
    delta : float
        Input sampling period for both x-dimension and y-dimension (if different) [m].
    dz : float
        Propagation distance [m].
    bandlimit : bool
        Whether to bandlimit propagation in order to avoid aliasing, as proposed in "Band-Limited
        Angular Spectrum Method for Numerical Simulation of Free-Space Propagation in Far and Near
        Fields" (2009).
    """
    if isinstance(delta, float) or isinstance(delta, int):
        delta = [delta, delta]
    assert len(delta) == 2

    # zero pad to simulate linear convolution
    Ny, Nx = u_in.shape
    y_pad_edge = int(np.ceil(Ny / 2.0))
    x_pad_edge = int(np.ceil(Nx / 2.0))
    pad_width = ((y_pad_edge, y_pad_edge), (x_pad_edge, x_pad_edge))
    u_in_pad = np.pad(u_in, pad_width=pad_width, mode="constant", constant_values=0)

    # size of the field
    Ny, Nx = u_in_pad.shape
    Dy, Dx = (delta[1] * float(Ny), delta[0] * float(Nx))

    # frequency coordinates sampling, TODO check (commented is from neural holography)
    # neural holography is probably wrong if you compare with fftfreq
    # fX = np.linspace(-1 / (2 * dx) + 0.5 / (2 * Dx), 1 / (2 * dx) - 0.5 / (2 * Dx), Nx)[:, np.newaxis].T
    # fY = np.linspace(-1 / (2 * dy) + 0.5 / (2 * Dy), 1 / (2 * dy) - 0.5 / (2 * Dy), Ny)[:, np.newaxis]
    dfX = 1.0 / Dx
    dfY = 1.0 / Dy
    fX = np.arange(-Nx / 2, Nx / 2)[:, np.newaxis].T * dfX
    fY = np.arange(-Ny / 2, Ny / 2)[:, np.newaxis] * dfY
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

    # band-limited to avoid aliasing - Eq 13 and 20 of Matsushima et al. (2009)
    if bandlimit:
        fx_max = 1 / np.sqrt((2 * dz * (1 / Dx)) ** 2 + 1) / wv
        fy_max = 1 / np.sqrt((2 * dz * (1 / Dy)) ** 2 + 1) / wv
        H_filter = (np.abs(fX) <= fx_max) * (np.abs(fY) < fy_max)
        H *= H_filter

    # perform convolution, TODO : bad FFT shifting in neural holography code?
    # U1 = np.fft.fftn(np.fft.ifftshift(Uin_pad), axes=(-2, -1), norm='ortho')
    U1 = ft2(u_in_pad, delta=delta)
    U2 = H * U1

    # Uout = np.fft.fftshift(np.fft.ifftn(U2, axes=(-2, -1), norm='ortho'))
    u_out = ift2(U2, delta_f=[dfX, dfY])

    # remove padding
    u_out = u_out[pad_width[0][0] : -pad_width[0][0], pad_width[1][0] : -pad_width[1][1]]

    # coordinates
    Ny, Nx = u_out.shape
    x2, y2 = sample_points(N=[Nx, Ny], delta=delta)
    # x2, y2 = np.meshgrid(np.arange(-Nx / 2, Nx / 2) * delta[0], np.arange(-Ny / 2, Ny / 2) * delta[1])

    return u_out, x2, y2


def fresnel_prop_circ_ap(wv, dz, diam, x, y):
    raise NotImplementedError


def fresnel_multi_step(u_in, wv, delta1, deltan, z):
    N = u_in.shape[0]  # assume square grid
    k = 2 * np.pi / wv

    src = np.arange(-N / 2, N / 2)
    nx, ny = np.meshgrid(src, src)

    # super-Gaussian absorbing boundary
    nsq = nx ** 2 + ny ** 2
    w = 0.47 * N
    sg = np.exp(-(nsq ** 8) / w ** 16)

    z = np.r_[0, z]  # propagation plane locations
    n = len(z)

    # propagation distances
    Delta_z = z[1:] - z[: n - 1]
    # grid spacings
    alpha = z / z[-1]
    delta = (1 - alpha) * delta1 + alpha * deltan
    m = delta[1:] / delta[: n - 1]
    x1 = nx * delta[0]
    y1 = ny * delta[0]
    r1sq = x1 ** 2 + y1 ** 2

    Q1 = np.exp(1j * k / 2 * (1 - m[0]) / Delta_z[0] * r1sq)
    u_in = u_in * Q1
    for idx in range(n - 1):
        # spatial frequencies of ith plane
        deltaf = 1 / (N * delta[idx])
        fX = nx * deltaf
        fY = ny * deltaf
        fsq = fX ** 2 + fY ** 2
        Z = Delta_z[idx]  # propagation distance

        # quadratic phase factor
        Q2 = np.exp(-1j * np.pi ** 2 * 2 * Z / m[idx] / k * fsq)

        # compute propagated field
        u_in = sg * ift2(Q2 * ft2(u_in / m[idx], delta[idx]), deltaf)

    # observation plane coordinates
    xn = nx * delta[n - 1]
    yn = ny * delta[n - 1]
    rnsq = xn ** 2 + yn ** 2
    Q3 = np.exp(1j * k / 2 * (m[n - 2] - 1) / (m[n - 2] * Z) * rnsq)
    Uout = Q3 * u_in

    return Uout, xn, yn
