import numpy as np
import torch
from waveprop.pytorch_util import fftconvolve as fftconvolve_torch
import warnings
from scipy.signal import fftconvolve
from waveprop.util import ft2, ift2, sample_points, crop, _get_dtypes, zero_pad
from pyffs import ffsn, fs_interpn, ffs_shift


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
    r = np.sqrt(x**2 + y**2 + z**2)
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


def angular_spectrum_np(
    u_in,
    wv,
    d1,
    dz,
    bandlimit=True,
    out_shift=0,
    d2=None,
    N_out=None,
    pyffs=False,
    in_shift=None,
    weights=None,
    dtype=None,
    return_H=False,
    pad=True,
):
    """
    Band-Limited Angular Spectrum Method for Numerical Simulation of Free-Space Propagation in Far
    and Near Fields (2009)

    TODO : set data type
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
        if isinstance(N_out, int):
            N_out = [N_out, N_out]
        assert len(N_out) == 2
        assert [isinstance(val, int) for val in N_out]
    if isinstance(out_shift, float) or isinstance(out_shift, int):
        out_shift = [out_shift, out_shift]
    assert len(out_shift) == 2
    if d2 is None and N_out is None and pyffs:
        warnings.warn("Defaulting to standard BLAS as no need for pyFFS interpolation.")
        pyffs = False
    if in_shift is not None:
        if weights is not None:
            assert len(weights) == len(in_shift)
        else:
            weights = np.ones(len(in_shift))
    if dtype is None:
        # TODO : use it!
        dtype = u_in.dtype
    if dtype == np.float32:
        ctype = np.complex64
    else:
        ctype = np.complex128

    # TODO : check to support multiple input shifts which get added
    # TODO : implement for when d2 and N_out are not None

    # if isinstance(in_shift, float) or isinstance(in_shift, int):
    #     in_shift = [in_shift, in_shift]
    # assert len(in_shift) == 2

    if pad:
        # zero pad to simulate linear convolution
        Ny, Nx = u_in.shape
        u_in_pad = zero_pad(u_in)
    else:
        u_in_pad = u_in

    # size of the padded field
    Ny_pad, Nx_pad = u_in_pad.shape
    Dy, Dx = (d1[0] * float(Ny_pad), d1[1] * float(Nx_pad))

    # frequency coordinates sampling
    dfX = 1.0 / Dx
    dfY = 1.0 / Dy
    fX = np.arange(-Nx_pad / 2, Nx_pad / 2)[np.newaxis, :] * dfX
    fY = np.arange(-Ny_pad / 2, Ny_pad / 2)[:, np.newaxis] * dfY
    fsq = fX**2 + fY**2

    # compute transfer function (Saleh / Sepand's notes but w/o abs val on distance)
    k = 2 * np.pi / wv
    wv_sq = wv**2
    # H = np.zeros_like(u_in_pad).astype(complex)
    H = np.zeros((fY.shape[0], fX.shape[1]), dtype=ctype)
    prop_waves = fsq <= 1 / wv_sq
    evanescent_waves = np.logical_not(prop_waves)
    H[prop_waves] = np.exp(1j * k * dz * np.sqrt(1 - wv_sq * fsq[prop_waves]))
    # evanescent waves
    H[evanescent_waves] = np.exp(-k * dz * np.sqrt(wv_sq * fsq[evanescent_waves] - 1))

    # shift
    if (out_shift[0] or out_shift[1]) and not pyffs:
        # Eq 7 of Matsushima (2010)
        H *= np.exp(1j * 2 * np.pi * (out_shift[1] * fX + out_shift[0] * fY))

    # band-limited to avoid aliasing
    # - Eq 13 and 20 of Matsushima et al. (2009)
    # - Table 1 of Matsushima (2010) for generalization to off-axis
    if bandlimit:
        # H = _bandpass(
        #     H, fX, fY, Sx=Nx * d1[1], Sy=Ny * d1[0], x0=out_shift[1], y0=out_shift[0], z0=dz, wv=wv
        # )
        H = _bandpass(
            H,
            fX,
            fY,
            Sx=Nx_pad * d1[1],
            Sy=Ny_pad * d1[0],
            x0=out_shift[1],
            y0=out_shift[0],
            z0=dz,
            wv=wv,
        )

    if return_H:
        return H

    if d2 is None and N_out is None:

        # perform convolution
        U1 = ft2(u_in_pad, delta=d1)
        if in_shift is not None:
            shift_terms = np.zeros_like(U1)

            # y_mod = np.exp(-1j * 2 * np.pi * fY @ in_shift[:, 0][np.newaxis, :])
            # x_mod = np.exp(-1j * 2 * np.pi * in_shift[:, 1][:, np.newaxis] @ fX)
            # for i in range(len(in_shift)):
            #     shift_terms += y_mod[:, i][:, np.newaxis] @ x_mod[i, :][np.newaxis, :]

            # import pudb; pudb.set_trace()

            ## -- this approach is faster, can be precomputed?
            for i, shift in enumerate(in_shift):
                _shift = np.ones_like(U1)
                if shift[0]:
                    _shift *= np.exp(-1j * 2 * np.pi * fY * shift[0])
                if shift[1]:
                    _shift *= np.exp(-1j * 2 * np.pi * fX * shift[1])
                shift_terms += _shift * weights[i]

            U1 *= shift_terms
        U2 = H * U1

        # back to spatial domain
        u_out = ift2(U2, delta_f=[dfY, dfX])

        # remove padding
        if pad:
            y_pad_edge = int(Ny // 2)
            x_pad_edge = int(Nx // 2)
            u_out = u_out[
                y_pad_edge : y_pad_edge + Ny,
                x_pad_edge : x_pad_edge + Nx,
            ]

        # output coordinates
        x2, y2 = sample_points(N=[Ny, Nx], delta=d1, shift=out_shift)

    else:
        if N_out is None:
            N_out = [Ny, Nx]
        if d2 is None:
            d2 = d1
        # output coordinates
        x2, y2 = sample_points(N=N_out, delta=d2, shift=out_shift)

        if pyffs:
            # compute FS coefficients of input
            # -- reshuffle input for pyFFS
            T = [Dy, Dx]
            T_c = [0, 0]
            N_s = np.array(u_in_pad.shape)
            N_FS = [ns if ns % 2 else ns // 2 * 2 - 1 for ns in N_s]  # must be odd
            u_in_pad_reorder = ffs_shift(u_in_pad)

            # -- compute coefficients
            U1 = ffsn(u_in_pad_reorder, T, T_c, N_FS)[: N_FS[0], : N_FS[1]]
            H = H[: N_FS[0], : N_FS[1]]

            # convolution
            U2 = H * U1

            # output coordinates
            # TODO: if d2 = d1, N_out=N_in revert to standard ASM
            x2, y2 = sample_points(N=N_out, delta=d2, shift=out_shift)

            # use output FS coefficients to interpolate
            a = [np.min(y2), np.min(x2)]
            b = [np.max(y2), np.max(x2)]
            u_out = fs_interpn(x_FS=U2, T=T, a=a, b=b, M=N_out)

        else:

            # perform convolution
            U1 = ft2(u_in_pad, delta=d1)
            U2 = H * U1

            # -- rescaled BLAS
            alpha_x = d2[1] / dfX
            alpha_y = d2[0] / dfY

            # Eq 9 of "Band-limited angular spectrum numerical propagation method with selective scaling
            # of observation window size and sample number" (2012)
            u_out = (
                np.exp(1j * np.pi / alpha_x * x2**2)
                * d2[1]
                * np.exp(1j * np.pi / alpha_y * y2**2)
                * d2[0]
            ).astype(ctype)

            fX_scaled = alpha_x * fX
            fY_scaled = alpha_y * fY
            B = (
                U2
                * (1 / alpha_x)
                * (1 / alpha_y)
                * np.exp(1j * np.pi / alpha_x * fX_scaled**2)
                * np.exp(1j * np.pi / alpha_y * fY_scaled**2)
            )
            f = np.exp(-1j * np.pi / alpha_x * fX_scaled**2) * np.exp(
                -1j * np.pi / alpha_y * fY_scaled**2
            )
            tmp = fftconvolve(B, f, mode="same")
            u_out *= tmp[
                int(Ny - N_out[0] / 2) : int(Ny + N_out[0] / 2),
                int(Nx - N_out[1] / 2) : int(Nx + N_out[1] / 2),
            ]

    return u_out, x2, y2


def angular_spectrum(
    u_in,
    wv,
    d1,
    dz,
    bandlimit=True,
    out_shift=0,
    d2=None,
    N_out=None,
    pyffs=False,
    aperture=None,
    aperture_ft=None,
    in_shift=None,
    weights=None,
    dtype=None,
    return_H=False,
    return_H_exp=False,
    return_U1=False,
    H=None,
    H_exp=None,
    U1=None,
    device=None,
    pad=True,
):
    """

    Parameters
    ----------
    u_in
    wv
    d1
    dz
    bandlimit
    out_shift
    d2
    N_out
    pyffs
    aperture
    in_shift
    weights
    dtype
    return_H
    return_H_exp
    return_U1
    H
    H_exp
    U1
    device

    Returns
    -------

    """
    if isinstance(d1, float) or isinstance(d1, int):
        d1 = [d1, d1]
    assert len(d1) == 2
    if d2 is not None:
        if isinstance(d2, float) or isinstance(d2, int):
            d2 = [d2, d2]
        assert len(d2) == 2
    if N_out is not None:
        if isinstance(N_out, int):
            N_out = [N_out, N_out]
        assert len(N_out) == 2
        assert [isinstance(val, int) for val in N_out]
    if isinstance(out_shift, float) or isinstance(out_shift, int):
        out_shift = [out_shift, out_shift]
    assert len(out_shift) == 2
    if d2 is None and N_out is None and pyffs:
        warnings.warn("Defaulting to standard BLAS as no need for pyFFS interpolation.")
        pyffs = False
    if torch.is_tensor(u_in) or torch.is_tensor(dz):
        is_torch = True
    else:
        is_torch = False
    if in_shift is not None:
        assert weights is not None
        # if shifting u_in aperture as for an SLM
        # TODO distinguish between amplitude and phase weights
        assert len(weights) == len(in_shift)
        if torch.is_tensor(weights):
            is_torch = True
    if is_torch:
        assert device is not None, "Set device for PyTorch"
        if torch.is_tensor(u_in):
            u_in = u_in.to(device)
        if torch.is_tensor(dz):
            dz = dz.to(device)
        if weights is not None and torch.is_tensor(weights):
            weights = weights.to(device)
    if dtype is None:
        dtype = u_in.dtype
    ctype, ctype_np = _get_dtypes(dtype, is_torch)

    Ny, Nx = u_in.shape
    # pad input to linearize convolution
    if pad:
        u_in_pad = zero_pad(u_in)
    else:
        u_in_pad = u_in

    # size of the padded field
    Ny_pad, Nx_pad = u_in_pad.shape
    Dy, Dx = (d1[0] * float(Ny_pad), d1[1] * float(Nx_pad))

    # frequency coordinates sampling
    dfX = 1.0 / Dx
    dfY = 1.0 / Dy
    fX = np.arange(-Nx_pad / 2, Nx_pad / 2)[np.newaxis, :] * dfX
    fY = np.arange(-Ny_pad / 2, Ny_pad / 2)[:, np.newaxis] * dfY

    # compute FT of input
    if U1 is None:
        if not return_H and not return_H_exp:
            if pyffs:
                if is_torch:
                    raise ValueError("No PyTorch support for pyFFS")

                # compute FS coefficients of input
                # -- reshuffle input for pyFFS
                T = [Dy, Dx]
                T_c = [0, 0]
                N_s = np.array(u_in_pad.shape)
                N_FS = [ns if ns % 2 else ns // 2 * 2 - 1 for ns in N_s]  # must be odd
                u_in_pad_reorder = ffs_shift(u_in_pad)

                # -- compute coefficients
                U1 = ffsn(u_in_pad_reorder, T, T_c, N_FS)[: N_FS[0], : N_FS[1]]
                # TODO
            # else:
            #     U1 = ft2(u_in_pad, delta=d1)
            #     if not torch.is_tensor(U1):
            #         U1 = U1.astype(ctype_np)

            if in_shift is not None:
                # input field goes through SLM mask, take into account deadspace
                # TODO : precompute shift terms? would take a lot of memory...
                if aperture is None and aperture_ft is None:
                    # use input field as aperture that we shift
                    AP = ft2(u_in_pad, delta=d1)
                else:
                    # we have separate input field that we need to multiply with mask in time domain
                    if aperture_ft is None:
                        assert aperture.shape == u_in.shape
                        aperture_pad = zero_pad(aperture)
                        AP = ft2(aperture_pad, delta=d1)
                    else:
                        AP = aperture_ft

                if aperture_ft is None:
                    AP = AP.astype(ctype_np)
                    if is_torch:
                        AP = torch.tensor(AP, dtype=ctype).to(device)

                # TODO: matrix-free, will `index_select` work for differentiation??
                if is_torch:
                    shift_terms = torch.zeros_like(AP)
                else:
                    shift_terms = np.zeros_like(AP)
                for i, shift in enumerate(in_shift):
                    _shift = np.ones(AP.shape, dtype=ctype_np)
                    if shift[0]:
                        _shift *= np.exp(-1j * 2 * np.pi * fY * shift[0])
                    if shift[1]:
                        _shift *= np.exp(-1j * 2 * np.pi * fX * shift[1])
                    # if is_torch:
                    #     # TODO will overwriting _shift be a problem for backprop?
                    #     _shift = torch.tensor(_shift.astype(ctype_np), dtype=ctype).to(device)

                    if is_torch:
                        index_tensor = torch.tensor([i], dtype=torch.int, device=device)
                        shift_terms += torch.tensor(_shift.astype(ctype_np), dtype=ctype).to(
                            device
                        ) * torch.index_select(weights, 0, index_tensor)
                    else:
                        shift_terms += _shift * weights[i]

                if aperture is None:
                    U1 = AP * shift_terms
                else:
                    # go back to time domain to multiply with mask
                    # TODO option to do without shifting in freq domain for faster output
                    AP *= shift_terms
                    u_in_masked = u_in_pad * ift2(AP, delta_f=[dfY, dfX])
                    U1 = ft2(u_in_masked, delta=d1)

            else:
                U1 = ft2(u_in_pad, delta=d1)
                if not torch.is_tensor(U1):
                    U1 = U1.astype(ctype_np)

    if return_U1:
        return U1

    # compute transfer function
    if H is None:
        H = _form_transfer_function(
            u_in_pad,
            d1,
            wv,
            dz,
            out_shift,
            bandlimit,
            H_exp,
            dtype,
            pyffs,
            return_H_exp,
            device,
            is_torch,
        )
    if return_H_exp:
        return H
    if return_H:
        return H

    # compute output, convolution in frequency domain
    if is_torch:
        if not torch.is_tensor(U1):
            U1 = torch.tensor(U1.astype(ctype_np), dtype=ctype).to(device)
        if not torch.is_tensor(H):
            H = torch.tensor(H.astype(ctype_np), dtype=ctype).to(device)
    U2 = H * U1
    if d2 is None and N_out is None:

        # back to spatial domain
        u_out = ift2(U2, delta_f=[dfY, dfX])

        # remove padding
        if pad:
            u_out = crop(u_out, shape=(Ny, Nx), topleft=(int(Ny // 2), int(Nx // 2)))

        # output coordinates
        x2, y2 = sample_points(N=[Ny, Nx], delta=d1, shift=out_shift)

    else:
        if N_out is None:
            N_out = [Ny, Nx]
        if d2 is None:
            d2 = d1

        # output coordinates
        x2, y2 = sample_points(N=N_out, delta=d2, shift=out_shift)

        if pyffs:
            # output coordinates
            # TODO: if d2 = d1, N_out=N_in revert to standard ASM
            x2, y2 = sample_points(N=N_out, delta=d2, shift=out_shift)

            # use output FS coefficients to interpolate
            a = [np.min(y2), np.min(x2)]
            b = [np.max(y2), np.max(x2)]
            u_out = fs_interpn(x_FS=U2, T=T, a=a, b=b, M=N_out)

        else:
            # -- rescaled BLAS
            # Eq 9 of "Band-limited angular spectrum numerical propagation method with selective scaling
            # of observation window size and sample number" (2012)
            alpha_x = d2[1] / dfX
            alpha_y = d2[0] / dfY
            u_out = (
                np.exp(1j * np.pi / alpha_x * x2**2)
                * d2[1]
                * np.exp(1j * np.pi / alpha_y * y2**2)
                * d2[0]
            ).astype(ctype_np)
            fX_scaled = alpha_x * fX
            fY_scaled = alpha_y * fY
            f = np.exp(-1j * np.pi / alpha_x * fX_scaled**2) * np.exp(
                -1j * np.pi / alpha_y * fY_scaled**2
            )

            if is_torch:
                u_out = torch.tensor(u_out, dtype=ctype).to(device)
                mod_term = (
                    np.exp(1j * np.pi / alpha_x * fX_scaled**2)
                    * np.exp(1j * np.pi / alpha_y * fY_scaled**2)
                    * (1 / alpha_x)
                    * (1 / alpha_y)
                )
                mod_term = torch.tensor(mod_term.astype(ctype_np), dtype=ctype).to(device)
                f = torch.tensor(f.astype(ctype_np), dtype=ctype).to(device)
                B = U2 * mod_term
                tmp = fftconvolve_torch(B, f, mode="same")
            else:
                B = (
                    U2
                    * (1 / alpha_x)
                    * (1 / alpha_y)
                    * np.exp(1j * np.pi / alpha_x * fX_scaled**2)
                    * np.exp(1j * np.pi / alpha_y * fY_scaled**2)
                )
                tmp = fftconvolve(B, f, mode="same")
            u_out *= crop(
                tmp, shape=N_out, topleft=(int(Ny - N_out[0] / 2), int(Nx - N_out[1] / 2))
            )

    return u_out, x2, y2


def _form_transfer_function(
    u_in_pad,
    d1,
    wv,
    dz,
    out_shift,
    bandlimit=True,
    H_exp=None,
    dtype=None,
    pyffs=False,
    return_H_exp=False,
    device=None,
    is_torch=False,
):
    """

    Return optical transfer function (OTF) for free space propagation. Handles
    both numpy and pytorch tensors.

    Note that if using Tensor and optimizing for distance `dz`, you may observe
    quantization effects due to quantizing before multiplying distance with rest
    of complex argument.

    Parameters
    ----------
    u_in_pad : numpy array or Tensor
        Padded input plane. Determine output type, e.g. numpy array or pytorch
        tensor.
    d1 : list or tuple
        Input sampling.
    wv : float
        wavelength
    dz : float or Tensor
        Propagation distance. If Tensor is provided, it is an indication that
        we want to optimize for the distance and a Tensor is return
    out_shift : tuple or list
        Output shift.
    bandlimit : bool
        Whether to bandlimit propagation
    H_exp : Tensor
        Precomputed complex argument (before multiplying with distance) when
        optimizing for propagation distance.
    dtype : dtype from numpy or pytorch
        To indicate whether working in float32 or float 64.
    pyffs : bool
        Whether pyffs will be used for output interpolation.
    is_torch

    Returns
    -------
    numpy array or Tensor
        If ` u_in_pad` or `dz` is Tensor, output will be Tensor. Otherwise
        numpy array.
    """
    assert len(d1) == 2
    assert len(out_shift) == 2

    if torch.is_tensor(u_in_pad):
        is_torch = True
        if device is None:
            device = u_in_pad.device

    if torch.is_tensor(dz):
        is_torch = True
        optimize_z = True
        if device is None:
            device = dz.device
        else:
            dz = dz.to(device)
    else:
        optimize_z = False

    if dtype is None:
        dtype = u_in_pad.dtype

    ctype, ctype_np = _get_dtypes(dtype, is_torch)

    # size of the padded field
    Ny_pad, Nx_pad = u_in_pad.shape
    Dy, Dx = (d1[0] * float(Ny_pad), d1[1] * float(Nx_pad))

    # frequency coordinates sampling
    dfX = 1.0 / Dx
    dfY = 1.0 / Dy
    fX = np.arange(-Nx_pad / 2, Nx_pad / 2)[np.newaxis, :] * dfX
    fY = np.arange(-Ny_pad / 2, Ny_pad / 2)[:, np.newaxis] * dfY

    # - compute transfer function
    fsq = fX**2 + fY**2
    k = 2 * np.pi / wv
    wv_sq = wv**2

    if optimize_z or return_H_exp or H_exp is not None:
        # need to cast to tensor before multiplying with z inside complex exp
        if H_exp is None:

            # distinguish propagation and evanescent waves
            prop = (fsq <= 1 / wv_sq).astype(np.uint8)
            evan = np.logical_not(prop)
            sqrt_arg = np.abs(1 - wv_sq * fsq)
            H_exp = (prop * 1j * k * np.sqrt(sqrt_arg) - evan * k * np.sqrt(sqrt_arg)).astype(
                ctype_np
            )
            if is_torch:
                H_exp = torch.tensor(H_exp, dtype=ctype).to(device)

        if return_H_exp:
            return H_exp

        if is_torch:
            H = torch.exp(H_exp * dz)
        else:
            H = np.exp(H_exp * dz)

        if (out_shift[0] or out_shift[1]) and not pyffs:
            # Eq 7 of Matsushima (2010)
            H_shift = np.exp(1j * 2 * np.pi * (out_shift[1] * fX + out_shift[0] * fY))
            if is_torch:
                H_shift = torch.tensor(H_shift.astype(ctype_np), dtype=ctype).to(device)

            H *= H_shift

        if bandlimit:
            H = _bandpass(
                H,
                fX,
                fY,
                Sx=Nx_pad * d1[1],
                Sy=Ny_pad * d1[0],
                x0=out_shift[1],
                y0=out_shift[0],
                z0=dz.cpu().numpy()[0] if torch.is_tensor(dz) else dz,
                wv=wv,
            )
    else:

        # form H completely in numpy and then cast to tensor
        H = np.zeros((fY.shape[0], fX.shape[1]), dtype=ctype_np)
        prop_waves = fsq <= 1 / wv_sq
        evanescent_waves = np.logical_not(prop_waves)
        H[prop_waves] = np.exp(1j * k * dz * np.sqrt(1 - wv_sq * fsq[prop_waves]))
        # evanescent waves
        H[evanescent_waves] = np.exp(-k * dz * np.sqrt(wv_sq * fsq[evanescent_waves] - 1))

        # shift
        if (out_shift[0] or out_shift[1]) and not pyffs:
            # Eq 7 of Matsushima (2010)
            H *= np.exp(1j * 2 * np.pi * (out_shift[1] * fX + out_shift[0] * fY))

        # band-limited to avoid aliasing
        # - Eq 13 and 20 of Matsushima et al. (2009)
        # - Table 1 of Matsushima (2010) for generalization to off-axis
        if bandlimit:
            H = _bandpass(
                H,
                fX,
                fY,
                Sx=Nx_pad * d1[1],
                Sy=Ny_pad * d1[0],
                x0=out_shift[1],
                y0=out_shift[0],
                z0=dz,
                wv=wv,
            )

        # cast to tensor
        if is_torch:
            H = torch.tensor(H, dtype=ctype).to(device)

    return H


def _bandpass(H, fX, fY, Sx, Sy, x0, y0, z0, wv):
    """
    Table 1 of "Shifted angular spectrum method for off-axis numerical propagation" (2010).

    :param Sx:
    :param Sy:
    :param x0:
    :param y0:
    :return:
    """
    du = 1 / (2 * Sx)
    u_limit_p = ((x0 + 1 / (2 * du)) ** (-2) * z0**2 + 1) ** (-1 / 2) / wv
    u_limit_n = ((x0 - 1 / (2 * du)) ** (-2) * z0**2 + 1) ** (-1 / 2) / wv
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
    v_limit_p = ((y0 + 1 / (2 * dv)) ** (-2) * z0**2 + 1) ** (-1 / 2) / wv
    v_limit_n = ((y0 - 1 / (2 * dv)) ** (-2) * z0**2 + 1) ** (-1 / 2) / wv
    if Sy < y0:
        v0 = (v_limit_p + v_limit_n) / 2
        v_width = v_limit_p - v_limit_n
    elif y0 <= -Sy:
        v0 = -(v_limit_p + v_limit_n) / 2
        v_width = v_limit_n - v_limit_p
    else:
        v0 = (v_limit_p - v_limit_n) / 2
        v_width = v_limit_p + v_limit_n

    fx_max = u_width / 2
    fy_max = v_width / 2
    if torch.is_tensor(H):
        H_filter = torch.tensor(
            ((np.abs(fX - u0) < fx_max) & (np.abs(fY - v0) < fy_max)).astype(np.uint8),
            dtype=H.dtype,
        ).to(H.device)
    else:
        H_filter = (np.abs(fX - u0) <= fx_max) * (np.abs(fY - v0) < fy_max)
    return H * H_filter
