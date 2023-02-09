import numpy as np
import torch
from scipy.special import fresnel
from waveprop.util import sample_points, ft2, ift2, _get_dtypes, zero_pad, crop


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
    x1, y1 = sample_points(N=[Ny, Nx], delta=d1)
    x2, y2 = sample_points(N=[Ny, Nx], delta=[1 / Ny / d1[0] * wv * dz, 1 / Nx / d1[1] * wv * dz])

    # evaluate integral
    u_out = (
        np.exp(1j * k * dz)
        / (1j * wv * dz)
        * np.exp(1j * k / (2 * dz) * (x2**2 + y2**2))
        * ft2(u_in * np.exp(1j * k / (2 * dz) * (x1**2 + y1**2)), delta=d1)
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


def fresnel_conv(u_in, wv, d1, dz, device=None, dtype=None, d2=None, pad=True):
    """
    Fresnel numerical computation (through convolution perspective) that gives
    control over output sampling but at a higher cost of two FFTs.

    Based off of Listing 6.5 of "Numerical Simulation of Optical Wave
    Propagation with Examples in MATLAB" (2010). Added zero-padding and support
    for PyTorch.

    NB: only works for square sampling, as non-square would result in different
    magnification factors.

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
    pad : bool
        Whether or not to zero-pad to linearize circular convolution. If the
        original signal has enough padding, this may not be necessary.
    device : "cpu" or "gpu"
        If using PyTorch, required. Device on which to perform computations.
    dtype : float32 or float 64
        Data type to use.

    """
    if torch.is_tensor(u_in) or torch.is_tensor(dz):
        is_torch = True
    else:
        is_torch = False
    if is_torch:
        assert device is not None, "Set device for PyTorch"
        if torch.is_tensor(u_in):
            u_in = u_in.to(device)
        if torch.is_tensor(dz):
            dz = dz.to(device)
    assert isinstance(d1, float)
    if d2 is None:
        d2 = d1
    else:
        assert isinstance(d2, float)
    if dtype is None:
        dtype = u_in.dtype
    ctype, ctype_np = _get_dtypes(dtype, is_torch)

    if pad:
        N_orig = np.array(u_in.shape)
        u_in = zero_pad(u_in)
    N = np.array(u_in.shape)
    k = 2 * np.pi / wv

    # source coordinates
    x1, y1 = sample_points(N=N, delta=d1)
    r1sq = x1**2 + y1**2

    # source spatial frequencies
    df1 = 1 / (N * d1)
    fX, fY = sample_points(N=N, delta=df1)
    fsq = fX**2 + fY**2

    # scaling parameter
    m = d2 / d1

    # observation plane
    x2, y2 = sample_points(N=N, delta=d2)
    r2sq = x2**2 + y2**2

    # quadratic phase factors
    Q2 = np.exp(-1j * np.pi**2 * 2 * dz / m / k * fsq).astype(ctype_np)
    if is_torch:
        Q2 = torch.tensor(Q2, dtype=ctype).to(device)
    if m == 1:
        Q1 = 1
        Q3 = 1
    else:
        Q1 = np.exp(1j * k / 2 * (1 - m) / dz * r1sq).astype(ctype_np)
        Q3 = np.exp(1j * k / 2 * (m - 1) / (m * dz) * r2sq).astype(ctype_np)
        if is_torch:
            Q1 = torch.tensor(Q1, dtype=ctype).to(device)
            Q3 = torch.tensor(Q3, dtype=ctype).to(device)

    # propagated field
    u_out = Q3 * ift2(Q2 * ft2(Q1 * u_in / m, delta=d1), delta_f=df1)

    if pad:
        u_out = crop(u_out, shape=N_orig, topleft=(int(N_orig[0] // 2), int(N_orig[1] // 2)))

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


def shifted_fresnel(u_in, wv, d1, dz, d2=None, out_shift=0):
    """
    "Shifted Fresnel diffraction for computational holography." (2007)

    https://www.davidhbailey.com/dhbpapers/fracfft.pdf

    Control over output spacing and shift from axis of propagation.

    Section 3.

    TODO : check different Nx, Ny
    TODO : fix/check amplitude scaling

    Limitation
    - Number of discrete element in target image must be equal to number of discrete elements in
    source image.

    Parameters
    ----------
    u_in : array_like
        Input amplitude distribution, [Ny, Nx].
    wv : float
        Wavelength [m].
    d1 : float or list or tuple
        Input sampling period for both x-dimension and y-dimension [m]. Scalar if the same for both
        dimensions.
    dz : float
        Propagation distance [m].
    d2 : float or list or tuple
        Output sampling period for both x-dimension and y-dimension [m]. Scalar if the same for both
        dimensions.
    out_shift : array_like
        Shift from optical axis at output.

    """
    if isinstance(d1, float) or isinstance(d1, int):
        d1 = [d1, d1]
    if d2 is None:
        d2 = d1
    if isinstance(d2, float) or isinstance(d2, int):
        d2 = [d2, d2]

    Ny, Nx = u_in.shape
    k = 2 * np.pi / wv

    # output coordinates, same number as input
    x_m, y_n = sample_points(N=[Ny, Nx], delta=d2, shift=out_shift)
    m = np.arange(Nx)[np.newaxis, :]
    n = np.arange(Ny)[:, np.newaxis]
    x0 = np.min(x_m)
    y0 = np.min(y_n)

    # source coordinates
    Ny, Nx = u_in.shape
    x_p, y_q = sample_points(N=[Ny, Nx], delta=d1)
    p = np.arange(Nx)[np.newaxis, :]
    q = np.arange(Ny)[:, np.newaxis]
    x0_in = np.min(x_p)
    y0_in = np.min(y_q)

    # constants in front of sum in Eq 13
    fact_fresnel = (
        np.exp(1j * k * dz)
        / (1j * wv * dz)
        * np.exp(1j * np.pi * (x_m**2 + y_n**2) / wv / dz)
        * np.exp(-1j * 2 * np.pi * (x0_in * m * d2[0] + y0_in * n * d2[1]) / wv / dz)
    )

    # modulated input, second line of Eq 13
    h = (
        u_in
        * np.exp(1j * np.pi * (x_p**2 + y_q**2) / wv / dz)
        * np.exp(-1j * 2 * np.pi * (x_p * x0 + y_q * y0) / wv / dz)
    )

    # scaled Discrete Fourier transform (e.g. fractional FT), Eq 15 but for 2D
    s = d1[0] * d2[0] / wv / dz
    t = d1[1] * d2[1] / wv / dz
    fact_sdft = np.exp(-1j * np.pi * s * m**2) * np.exp(-1j * np.pi * t * n**2)
    # -- Eq 16, Eq 13 of Bailey
    f = h * np.exp(-1j * np.pi * s * p**2) * np.exp(-1j * np.pi * t * q**2)
    # -- Eq 17, Eq 14 of Bailey
    g = np.exp(1j * np.pi * s * p**2) * np.exp(1j * np.pi * t * q**2)

    # -- pad 2-D sequence, Eq 15-16 of Bailey
    pad_size = (2 * Ny - 1, 2 * Nx - 1)
    f_pad = np.zeros(pad_size, dtype=f.dtype)
    f_pad[:Ny, :Nx] = f
    g_pad = np.zeros(pad_size, dtype=g.dtype)
    g_pad[:Ny, :Nx] = g
    p_pad = np.arange(pad_size[1] - Nx, pad_size[1])[np.newaxis, :]
    g_pad[:Ny, pad_size[1] - Nx :] = np.exp(1j * np.pi * s * (p_pad - pad_size[1]) ** 2) * np.exp(
        1j * np.pi * t * q**2
    )
    q_pad = np.arange(pad_size[0] - Ny, pad_size[0])[:, np.newaxis]
    g_pad[pad_size[0] - Ny :, :Nx] = np.exp(1j * np.pi * s * p**2) * np.exp(
        1j * np.pi * t * (q_pad - pad_size[0]) ** 2
    )
    g_pad[pad_size[0] - Ny :, pad_size[1] - Nx :] = np.exp(
        1j * np.pi * s * (p_pad - pad_size[1]) ** 2
    ) * np.exp(1j * np.pi * t * (q_pad - pad_size[0]) ** 2)

    # fractional FT
    tmp = (
        np.fft.ifft2(np.fft.fft2(f_pad) * np.fft.fft2(g_pad))[:Ny, :Nx] / pad_size[0] / pad_size[1]
    )

    return fact_fresnel * fact_sdft * tmp, x_m, y_n


def fresnel_prop_circ_ap(wv, dz, diam, x, y):
    raise NotImplementedError


def fresnel_multi_step(u_in, wv, delta1, deltan, z):
    N = u_in.shape[0]  # assume square grid
    k = 2 * np.pi / wv

    src = np.arange(-N / 2, N / 2)
    nx, ny = np.meshgrid(src, src)

    # super-Gaussian absorbing boundary
    nsq = nx**2 + ny**2
    w = 0.47 * N
    sg = np.exp(-(nsq**8) / w**16)

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
    r1sq = x1**2 + y1**2

    Q1 = np.exp(1j * k / 2 * (1 - m[0]) / Delta_z[0] * r1sq)
    u_in = u_in * Q1
    for idx in range(n - 1):
        # spatial frequencies of ith plane
        deltaf = 1 / (N * delta[idx])
        fX = nx * deltaf
        fY = ny * deltaf
        fsq = fX**2 + fY**2
        Z = Delta_z[idx]  # propagation distance

        # quadratic phase factor
        Q2 = np.exp(-1j * np.pi**2 * 2 * Z / m[idx] / k * fsq)

        # compute propagated field
        u_in = sg * ift2(Q2 * ft2(u_in / m[idx], delta[idx]), deltaf)

    # observation plane coordinates
    xn = nx * delta[n - 1]
    yn = ny * delta[n - 1]
    rnsq = xn**2 + yn**2
    Q3 = np.exp(1j * k / 2 * (m[n - 2] - 1) / (m[n - 2] * Z) * rnsq)
    Uout = Q3 * u_in

    return Uout, xn, yn
