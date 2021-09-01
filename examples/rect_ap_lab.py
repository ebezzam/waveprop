"""
Simulating rectangular aperture for our lab setup.

"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import click

from waveprop.util import rect2d, sample_points, plot2d
from waveprop.prop import (
    direct_integration,
    angular_spectrum,
    angular_spectrum_ffs,
)
from waveprop.fresnel import fresnel_prop_square_ap, shifted_fresnel
from waveprop.fraunhofer import fraunhofer_prop_rect_ap
from waveprop.condition import (
    fraunhofer_schmidt,
    fraunhofer_goodman,
    fraunhofer_saleh,
    fresnel_saleh,
    fresnel_goodman,
    fraunhofer_valid_output_region,
    fresnel_valid_output_region,
)


@click.command()
@click.option("--dz", type=float, default=45e-2)
@click.option("--wid", type=float, default=40e-6)  # 40e-6 Holoeye SLM, 394e-6 Nokia display
@click.option("--r_out", type=float, default=2.5e-2)  # radius of output plane
@click.option("--n_grid", type=int, default=512)  # number of grid points per side
@click.option("--grid_len", type=float, default=1e-2)  # length of grid per side
@click.option("--wv", type=float, default=635e-9)  # wavelength [m]
@click.option("--di", is_flag=True, help="Whether to compare with direct integration.")
def prop(dz, wid, r_out, n_grid, grid_len, wv, di):

    d1 = grid_len / n_grid  # source-plane grid spacing

    # output coordinates
    delta = 2 * r_out / n_grid
    x2, y2 = sample_points(n_grid, delta)

    print("\nPROPAGATION DISTANCE : {} m".format(dz))

    """ discretize aperture """
    x1, y1 = sample_points(N=n_grid, delta=d1)
    u_in = rect2d(x1, y1, wid)

    """ Fraunhofer propagation """

    print("\nFraunhofer propagation")
    print("-" * 30)

    # Fraunhofer simulation
    u_out_fraun = fraunhofer_prop_rect_ap(wv, dz, x2, y2, wid, wid)

    # check condition
    fraunhofer_schmidt(wv, dz, wid)
    fraunhofer_goodman(wv, dz, x1=wid / 2, y1=wid / 2, x2=x2, y2=y2)
    fraunhofer_saleh(wv, dz, x1=wid / 2, y1=wid / 2, x2=x2, y2=y2)

    """ Fresnel propagation """

    # Fresnel theoretical
    u_out_th_fres = fresnel_prop_square_ap(x=x2, y=y2, width=wid, wv=wv, dz=dz)

    # check condition
    print("\nFresnel propagation")
    print("-" * 30)
    fresnel_goodman(wv, dz, x1=wid / 2, y1=wid / 2, x2=x2, y2=y2)
    fresnel_saleh(wv, dz, x=x2, y=y2)

    """ Angular spectrum """
    u_out_asm_bl, x_asm, y_asm = angular_spectrum(u_in=u_in, wv=wv, delta=d1, dz=dz, bandlimit=True)

    """ Angular spectrum with FS coefficients """
    N_out = n_grid * 2
    # d2 = d1 / 2
    # out_shift = d2 * N_out / 2
    d2 = d1
    out_shift = 0
    u_out_ffs, x2_ffs, y2_ffs = angular_spectrum_ffs(
        u_in, wv, d1, dz, N_out=N_out, d2=d2, out_shift=out_shift
    )

    """ Shifted Fresnel """
    output_scaling = 15
    out_shift = 0
    # out_shift = output_scaling * d1 * N / 2
    u_out_sfres, x2_sfres, y2_sfres = shifted_fresnel(
        u_in, wv, d1, dz, d2=output_scaling * d1, out_shift=out_shift
    )

    """ Direct integration (ground truth) """
    if di:
        u_out_di = direct_integration(u_in, wv, d1, dz, x=x2[0], y=[0])

    """ Plot """
    r_out_fraun = fraunhofer_valid_output_region(wv, dz, tol=1) * 1e3
    r_out_fres = fresnel_valid_output_region(wv, dz, tol=1) * 1e3

    # plot y2 = 0 cross-section
    idx = y2[:, 0] == 0
    plt.figure()

    plt.axvline(
        r_out_fraun,
        label="fraunhofer boundary - {:.2f} mm".format(r_out_fraun),
        color="green",
        linestyle="dashed",
    )
    plt.plot(
        x2[0] * 1e3, np.abs(u_out_fraun[:, idx]), color="green", label="fraunhofer (theoretical)"
    )
    plt.axvline(
        r_out_fres,
        label="fresnel boundary - {:.2f} mm".format(r_out_fres),
        color="blue",
        linestyle="dashed",
    )
    plt.plot(
        x2[0] * 1e3, np.abs(u_out_th_fres[idx][0]), color="blue", label="fresnel (theoretical)"
    )
    if di:
        plt.plot(x2[0] * 1e3, np.abs(u_out_di[0]), color="red", label="direct integration")
    plt.plot(x_asm[0] * 1e3, np.abs(u_out_asm_bl[:, idx]), color="orange", label="angular spectrum")
    plt.xlabel("x [mm]")
    plt.legend()
    plt.title("log amplitude, y2 = 0")
    plt.yscale("log")
    plt.xlim([0, r_out * 1e3])

    # plot input
    ax = plot2d(x1.squeeze(), y1.squeeze(), u_in)
    ax.set_title("Aperture")

    # plot output
    ax = plot2d(x2.squeeze(), y2.squeeze(), np.abs(u_out_fraun))
    ax.set_title("Fraunhofer (theoretical)")
    # ax.set_xlim([-xlim, xlim])
    # ax.set_ylim([-ylim, ylim])

    ax = plot2d(x2.squeeze(), y2.squeeze(), np.abs(u_out_th_fres))
    ax.set_title("Fresnel (theoretical)")
    # ax.set_xlim([-xlim, xlim])
    # ax.set_ylim([-ylim, ylim])

    ax = plot2d(x_asm.squeeze(), y_asm.squeeze(), np.abs(u_out_asm_bl))
    ax.set_title("Angular spectrum")
    ax.set_xlim([np.min(x2_ffs), np.max(x2_ffs)])
    ax.set_ylim([np.min(y2_ffs), np.max(y2_ffs)])

    ax = plot2d(x2_ffs.squeeze(), y2_ffs.squeeze(), np.abs(u_out_ffs))
    ax.set_title("Angular spectrum (FFS)")

    ax = plot2d(x2.squeeze(), y2.squeeze(), np.abs(u_out_th_fres))
    ax.set_title("Fresnel (theoretical)")
    ax.set_xlim([np.min(x2_ffs), np.max(x2_ffs)])
    ax.set_ylim([np.min(y2_ffs), np.max(y2_ffs)])

    ax = plot2d(x2_sfres.squeeze(), y2_sfres.squeeze(), np.abs(u_out_sfres))
    ax.set_title("Shifted Fresnel")

    plt.show()


if __name__ == "__main__":
    prop()
