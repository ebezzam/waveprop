import numpy as np
import math
import torch
from waveprop.util import sample_points, _get_dtypes
from waveprop.pytorch_util import fftconvolve as fftconvolve_torch
from scipy.signal import fftconvolve


def spherical_prop(
    u_in=None,
    in_shape=None,
    d1=None,
    wv=None,
    dz=None,
    return_psf=False,
    psf=None,
    dtype=None,
    is_torch=False,
    device="cpu",
):
    """

    Similar to: https://github.com/vsitzmann/deepoptics/blob/defbb975309a6a3f3d2a86b92e82d02156ab213e/src/layers/optics.py#L1010

    Parameters
    ----------
    u_in
    d1
    wv
    dz
    return_psf
    psf
    dtype

    Returns
    -------
    n_wv x n_x x n_y

    """

    if isinstance(wv, float):
        wv = np.array([wv])
    if u_in is not None:
        assert len(u_in.shape) == 3
        if torch.is_tensor(u_in):
            is_torch = True
            device = u_in.device
        else:
            is_torch = False
        in_shape = u_in.shape[1:]
        if dtype is None:
            dtype = u_in.dtype
    else:
        assert return_psf, "Require input field."
        assert in_shape is not None
        assert dtype is not None
    ctype, ctype_np = _get_dtypes(dtype, is_torch)

    if psf is None:
        assert d1 is not None
        assert wv is not None
        assert dz is not None

        x1, y1 = sample_points(N=in_shape, delta=d1)
        k = (2 * math.pi / wv)[:, np.newaxis, np.newaxis]
        curvature = np.sqrt(x1**2 + y1**2 + dz**2)[np.newaxis, :]
        psf = np.exp(1j * k * curvature).astype(ctype_np)
        if is_torch:
            psf = torch.tensor(psf, dtype=ctype, device=device)

    if return_psf:
        return psf
    else:
        assert u_in is not None

    if is_torch:
        return fftconvolve_torch(u_in, psf, axes=(-2, -1))
    else:
        return fftconvolve(u_in, psf, mode="same", axes=(-2, -1))
