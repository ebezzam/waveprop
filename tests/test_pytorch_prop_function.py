import torch
import numpy as np
from waveprop.pytorch_util import compute_numpy_error
from waveprop.util import rect2d, sample_points, plot2d
from waveprop.rs import angular_spectrum_np, angular_spectrum
from pprint import pprint
from scipy import special

# constants
N = 512  # number of grid points per size
L = 1e-2  # total size of grid
wv = 635e-9  # wavelength (red light)
verbose = False

# check if cuda available
if torch.cuda.is_available():
    DEVICES = ["cpu", "cuda"]
else:
    DEVICES = ["cpu"]



def _test_angular_spectrum(
    diam,
    dz,
    N_out=None,
    output_scaling=None,
    out_shift_fact=None,
    device="cpu",
    dtype=np.float32,
    bandlimit=True,
    optimize_z=False,
    plot=False,
    verbose=True,
):

    d1 = L / N  # source-plane grid spacing
    if output_scaling:
        d2 = output_scaling * d1
    else:
        d2 = None
    if out_shift_fact:
        out_shift = d1 * N / out_shift_fact
    else:
        out_shift = 0

    if optimize_z:
        dz_torch = torch.tensor([dz], dtype=dtype)
        dz_torch.requires_grad = False
    else:
        dz_torch = dz

    if dtype == torch.float32:
        dtype_np = np.float32
    else:
        dtype_np = np.float64

    err_dict = dict()

    """ create rectangular aperture """
    x1, y1 = sample_points(N=N, delta=d1)
    u_in = rect2d(x1, y1, diam).astype(dtype_np)
    if plot:
        plot2d(x1.squeeze(), y1.squeeze(), u_in, title="Aperture")

    """ to PyTorch / GPU """
    u_in_tensor = torch.from_numpy(u_in)

    """ Angular spectrum """
    H_exp_np = angular_spectrum(
        u_in=u_in,
        wv=wv,
        d1=d1,
        dz=dz,
        bandlimit=bandlimit,
        out_shift=out_shift,
        d2=d2,
        N_out=N_out,
        return_H_exp=True,
    )

    H_exp_torch = angular_spectrum(
        u_in=u_in_tensor,
        wv=wv,
        d1=d1,
        dz=dz_torch,
        bandlimit=bandlimit,
        out_shift=out_shift,
        d2=d2,
        N_out=N_out,
        return_H_exp=True,
        device=device,
    )
    if verbose:
        print("\n-- H_exp")
    err_dict["H_exp_err"] = compute_numpy_error(H_exp_torch, H_exp_np, verbose=verbose)

    H_np = angular_spectrum(
        u_in=u_in,
        wv=wv,
        d1=d1,
        dz=dz,
        bandlimit=bandlimit,
        out_shift=out_shift,
        d2=d2,
        N_out=N_out,
        H_exp=H_exp_np,
        return_H=True,
    )
    H_torch = angular_spectrum(
        u_in=u_in_tensor,
        wv=wv,
        d1=d1,
        dz=dz_torch,
        bandlimit=bandlimit,
        out_shift=out_shift,
        d2=d2,
        N_out=N_out,
        H_exp=H_exp_torch,
        return_H=True,
        device=device,
    )
    if verbose:
        print("\n-- H")
    err_dict["H_err"] = compute_numpy_error(H_torch, H_np, verbose=verbose)

    U1_np = angular_spectrum(
        u_in=u_in,
        wv=wv,
        d1=d1,
        dz=dz,
        bandlimit=bandlimit,
        out_shift=out_shift,
        d2=d2,
        N_out=N_out,
        return_U1=True,
    )
    U1_torch = angular_spectrum(
        u_in=u_in_tensor,
        wv=wv,
        d1=d1,
        dz=dz_torch,
        bandlimit=bandlimit,
        out_shift=out_shift,
        d2=d2,
        N_out=N_out,
        return_U1=True,
        device=device,
    )
    if verbose:
        print("\n-- U1")
    err_dict["U1_err"] = compute_numpy_error(U1_torch, U1_np, verbose=verbose)

    u_out_np, x2, y2 = angular_spectrum(
        u_in=u_in,
        wv=wv,
        d1=d1,
        dz=dz_torch,
        bandlimit=bandlimit,
        out_shift=out_shift,
        d2=d2,
        N_out=N_out,
        device=device,
    )
    if optimize_z:
        # TODO : move outside or another tests?
        assert torch.is_tensor(u_out_np)

    u_out_np, x2, y2 = angular_spectrum(
        u_in=u_in,
        wv=wv,
        d1=d1,
        dz=dz,
        bandlimit=bandlimit,
        out_shift=out_shift,
        d2=d2,
        N_out=N_out,
    )

    u_out_torch, x2, y2 = angular_spectrum(
        u_in=u_in_tensor,
        wv=wv,
        d1=d1,
        dz=dz_torch,
        bandlimit=bandlimit,
        out_shift=out_shift,
        d2=d2,
        N_out=N_out,
        device=device,
    )
    if verbose:
        print("\n-- u_out")
    err_dict["u_out_err"] = compute_numpy_error(u_out_torch, u_out_np, verbose=verbose)

    if plot:
        plot2d(x2.squeeze(), x2.squeeze(), np.abs(u_out_torch.cpu()) ** 2, title="AS (pytorch)")
        plot2d(x2.squeeze(), x2.squeeze(), np.abs(u_out_np) ** 2, title="AS (numpy partner)")

    """ Compare with numpy"""
    u_out_asm_np, x_asm, y_asm = angular_spectrum_np(
        u_in=u_in, wv=wv, d1=d1, dz=dz, bandlimit=bandlimit, out_shift=out_shift, d2=d2, N_out=N_out
    )
    if plot:
        plot2d(x_asm.squeeze(), y_asm.squeeze(), np.abs(u_out_asm_np) ** 2, title="AS (numpy)")

    if verbose:
        print("\n-- u_out (angular_spectrum_np)")
    err_dict["u_out_err_np"] = compute_numpy_error(u_out_torch, u_out_asm_np, verbose=verbose)

    err_dict["u_out_err_np_np"] = np.mean(np.abs((u_out_asm_np - u_out_np) ** 2))
    if verbose:
        print("\n-- u_out (angular_spectrum_np), numpy companion")
        print("angular_spectrum_np", u_out_asm_np.dtype)
        print("angular_spectrum", u_out_np.dtype)
        print("MSE wrt numpy", err_dict["u_out_err_np_np"])

    return u_out_torch, err_dict


def test_blas_no_shift_no_rescale():
    TOL = 1e-15
    TOL_optimize_z = 1e-6
    diam = 3e-4  # diameter of aperture [m]

    for dtype in [torch.float32, torch.float64]:

        for device in DEVICES:

            for dz in [0.1, 0.01]:

                for optimize_z in [True, False]:

                    print("\n", dtype, device, dz, optimize_z)
                    u_out_torch, err_dict = _test_angular_spectrum(
                        diam=diam,
                        dz=dz,
                        dtype=dtype,
                        device=device,
                        optimize_z=optimize_z,
                        verbose=verbose,
                    )
                    pprint(err_dict)
                    # TODO : when optimizing z, error increases quite a bit due to quantization before exponential
                    assert err_dict["u_out_err"] < TOL_optimize_z if optimize_z else TOL
                    assert err_dict["u_out_err_np"] < TOL_optimize_z if optimize_z else TOL
                    assert err_dict["u_out_err_np_np"] == 0
                    assert torch.abs(u_out_torch).dtype == dtype


def test_blas_off_axis():
    TOL = 1e-15
    TOL_optimize_z = 1e-5
    diam = 2e-3  # diameter of aperture [m]
    dz = 0.5  # distance [m]
    out_shift_fact = 2

    for dtype in [torch.float32, torch.float64]:

        for device in DEVICES:

            for dz in [0.1, 0.01]:

                for optimize_z in [True, False]:

                    print("\n", dtype, device, dz, optimize_z)
                    u_out_torch, err_dict = _test_angular_spectrum(
                        diam=diam,
                        dz=dz,
                        dtype=dtype,
                        device=device,
                        optimize_z=optimize_z,
                        verbose=verbose,
                        out_shift_fact=out_shift_fact,
                    )
                    pprint(err_dict)
                    # TODO : when optimizing z, error increases quite a bit due to quantization before exponential
                    assert err_dict["u_out_err"] < TOL_optimize_z if optimize_z else TOL
                    assert err_dict["u_out_err_np"] < TOL_optimize_z if optimize_z else TOL
                    assert err_dict["u_out_err_np_np"] == 0
                    assert torch.abs(u_out_torch).dtype == dtype


def test_blas_rescaling():
    TOL = 1e-15
    TOL_optimize_z = 1e-4
    diam = 2e-3  # diameter of aperture [m]
    out_shift_fact = 10
    output_scaling = 1 / 4

    for dtype in [torch.float32, torch.float64]:

        for device in DEVICES:

            for dz in [0.1, 0.01]:

                for optimize_z in [True, False]:

                    print("\n", dtype, device, dz, optimize_z)
                    u_out_torch, err_dict = _test_angular_spectrum(
                        diam=diam,
                        dz=dz,
                        dtype=dtype,
                        device=device,
                        optimize_z=optimize_z,
                        verbose=verbose,
                        out_shift_fact=out_shift_fact,
                        output_scaling=output_scaling,
                    )
                    pprint(err_dict)
                    # TODO : when optimizing z, error increases quite a bit due to quantization before exponential
                    assert err_dict["u_out_err"] < TOL_optimize_z if optimize_z else TOL
                    assert err_dict["u_out_err_np"] < TOL_optimize_z if optimize_z else TOL
                    assert err_dict["u_out_err_np_np"] == 0
                    assert torch.abs(u_out_torch).dtype == dtype


# size even
N_EVEN = 50
# size odd
N_ODD = 51

# lens diameter, in microns
D = 50
wl = 0.633 # wavelength in microns


lens_radius = D/2
F = 10*D # focal length

# even/odd grid
x_even = np.linspace(-D/2, D/2, N_EVEN)
y_even = x_even
xx_even, yy_even = np.meshgrid(x_even, y_even)
rr_even = np.sqrt(xx_even**2 + yy_even**2)
kAlphaSinTheta_even = 2*np.pi/wl*lens_radius*np.arctan(rr_even/F)

x_odd = np.linspace(-D/2, D/2, N_ODD)
y_odd = x_odd
xx_odd, yy_odd = np.meshgrid(x_odd, y_odd)
rr_odd =np.sqrt(xx_odd **2 + yy_odd**2)
kAlphaSinTheta_odd = 2*np.pi/wl*lens_radius*np.arctan(rr_odd/F)

# mask so that airy function is circularly symmetric
mask_even = np.zeros_like(xx_even)
mask_odd = np.zeros_like(xx_odd)
mask_even[xx_even ** 2 + yy_even ** 2 <= lens_radius ** 2] = 1
mask_odd[xx_odd ** 2 + yy_odd ** 2 <= lens_radius ** 2] = 1

# standard lens phase
phi_even = np.mod(2*np.pi/wl * (np.sqrt(xx_even ** 2 + yy_even ** 2 + F ** 2) - F), 2 * np.pi) * mask_even
phi_odd = np.mod(2*np.pi/wl * (np.sqrt(xx_odd ** 2 + yy_odd ** 2 + F ** 2) - F), 2 * np.pi) * mask_odd

# nearfield electric field
e_nearfield_even = np.exp( - 1j * phi_even)
e_nearfield_odd =  np.exp( - 1j * phi_odd)


def bessel_j1_over_r(r):
    """
    bessel_j1(r)/r
        \lim_{r->0} J1(r)/r = 0.5

    Parameters
    ----------
    r : float or numpy array
        radii

    Returns
    -------
        J1(r)/r.

    """
    tol = 1e-7
    ret = np.zeros_like(r)
    ret[np.abs(r) < tol] = 0.5
    ret[np.abs(r) > tol] = special.j1(r[np.abs(r) > tol]) / r[np.abs(r) > tol]
    return ret


airy_even = (2 * bessel_j1_over_r(kAlphaSinTheta_even)) ** 2
airy_odd = (2 * bessel_j1_over_r(kAlphaSinTheta_odd)) ** 2

def test_asm_even():
    # near to far field propagation
    ff = angular_spectrum(e_nearfield_even,
                          wl, x_even[1] - x_even[0],
                          F,
                          bandlimit=False,
                          pad=False)[0]
    # farfield intensity
    ff_I = np.abs( ff ) ** 2

    # airy disk is defined to be equal to 1 at r = 0
    # since even grids do not hit the r = 0 point, normalize farfield
    # normalize farfield to max(ff_I) = max(airy_even)
    ff_I = ff_I / np.max(ff_I) * np.max(airy_even)

    # calcualate error between analytic and
    # numerical solution. In case of off by 1 error, expect error to be of order ~0.5
    err = np.max(np.abs(ff_I - airy_even))
    assert err < 0.05

    # expect maxima in the center
    ci = int(N_EVEN / 2)
    ff_max_val = np.max(ff_I)
    assert np.abs(ff_I[ci,ci] - ff_max_val) < 1e-5
    assert np.abs(ff_I[ci-1, ci] - ff_max_val) < 1e-5
    assert np.abs(ff_I[ci, ci-1] - ff_max_val) < 1e-5
    assert np.abs(ff_I[ci-1, ci-1] - ff_max_val) < 1e-5



def test_asm_odd():
    # near to far field propagation
    ff = angular_spectrum(e_nearfield_odd,
                          wl, x_odd[1] - x_odd[0],
                          F,
                          bandlimit=False,
                          pad=False)[0]
    # farfield intensity
    ff_I = np.abs(ff) ** 2

    # airy disk is defined to be equal to 1 at r = 0
    ff_I = ff_I / np.max(ff_I)

    # calcualate error between analytic and
    # numerical solution. In case of off by 1 error, expect error to be of order ~0.5
    err = np.max(np.abs(ff_I - airy_odd))
    assert err < 0.05

    ci = int(N_ODD / 2)
    ff_max_val = np.max(ff_I)
    # expect max value at the center
    assert np.abs(ff_I[ci, ci] - 1) < 1e-5




if __name__ == "__main__":

    print("\n----- BLAS, no shift nor rescaling -----")
    test_blas_no_shift_no_rescale()

    print("\n----- BLAS, off-axis-----")
    test_blas_off_axis()

    print("\n----- BLAS, rescaling-----")
    test_blas_rescaling()
