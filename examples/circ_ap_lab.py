"""
Simulating circular aperture for our lab setup.

"""

import click
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker

from waveprop.util import circ, sample_points
from waveprop.prop import (
    fraunhofer_prop_circ_ap,
    fresnel_two_step,
    angular_spectrum,
    direct_integration,
)
from waveprop.condition import (
    fraunhofer_schmidt,
    fraunhofer_goodman,
    fraunhofer_saleh,
    fresnel_goodman,
    fresnel_saleh,
    fraunhofer_valid_output_region,
    fresnel_valid_output_region,
)


@click.command()
@click.option("--dz", type=float, default=45e-2)
@click.option(
    "--r_in", type=float, default=30e-6
)  # radius of input plane, 40e-6 Holoeye SLM, 394e-6 Nokia display
@click.option("--r_out", type=float, default=1e-2)  # radius of output plane
@click.option("--n_grid", type=int, default=512)  # number of grid points per side
@click.option("--grid_len", type=float, default=1e-2)  # length of grid per side
@click.option("--wv", type=float, default=635e-9)  # wavelength [m]
def prop(dz, r_in, r_out, n_grid, grid_len, wv):

    d1 = grid_len / n_grid  # source-plane grid spacing
    diam = 2 * r_in

    # dz_fraun = distance_from_output_region(wv, r_out=1e-2, tol=1)
    delta = 2 * r_out / n_grid
    x2, y2 = sample_points(n_grid, delta)

    print("\nPROPAGATION DISTANCE : {} m".format(dz))

    """ discretize aperture """
    x1, y1 = sample_points(N=n_grid, delta=d1)
    u_in = circ(x=x1, y=y1, diam=diam)

    """ Fraunhofer propagation """

    # Fraunhofer theoretical
    u_out_fraun = fraunhofer_prop_circ_ap(wv, dz, diam, x2, y2)

    # check condition
    print("\nFraunhofer propagation")
    print("-" * 30)
    fraunhofer_schmidt(wv, dz, diam)
    fraunhofer_goodman(wv, dz, x1=diam / 2, y1=0, x2=x2, y2=y2)
    fraunhofer_saleh(wv, dz, x1=diam / 2, y1=0, x2=x2, y2=y2)

    """ Fresnel approximation """
    u_out_fres, x2_fres, y2_fres = fresnel_two_step(u_in=u_in, wv=wv, d1=d1, d2=delta, dz=dz)

    # check condition
    print("\nFresnel propagation")
    print("-" * 30)
    fresnel_goodman(wv, dz, x1=diam / 2, y1=0, x2=x2, y2=y2)
    fresnel_saleh(wv, dz, x=x2, y=y2)

    """ Angular spectrum """
    u_out_asm_bl, x_asm, y_asm = angular_spectrum(u_in=u_in, wv=wv, delta=d1, dz=dz, bandlimit=True)

    """ Direct integration (ground truth) """
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
    plt.plot(x2_fres[0] * 1e3, np.abs(u_out_fres[:, idx]), color="blue", label="fresnel (two step)")
    plt.plot(x2[0] * 1e3, np.abs(u_out_di[0]), color="red", label="direct integration")
    plt.plot(x_asm[0] * 1e3, np.abs(u_out_asm_bl[:, idx]), color="orange", label="angular spectrum")
    plt.xlabel("x [mm]")
    plt.legend()
    plt.title("log amplitude, y2 = 0")
    plt.yscale("log")
    plt.xlim([0, r_out * 1e3])

    # plot input
    X1, Y1 = np.meshgrid(x1, y1)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    cp = ax.contourf(X1, Y1, u_in)
    fig = plt.gcf()
    fig.colorbar(cp, ax=ax, orientation="vertical")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Aperture")

    # plot outputs
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    X2, Y2 = np.meshgrid(x2, y2)
    cp = ax.contourf(X2, Y2, np.abs(u_out_fraun), locator=ticker.LogLocator())
    fig = plt.gcf()
    fig.colorbar(cp, ax=ax, orientation="vertical")
    ax.set_title("Fraunhofer diffraction pattern (theoretical)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    cp = ax.contourf(X2, Y2, np.abs(u_out_fres), locator=ticker.LogLocator())
    fig = plt.gcf()
    fig.colorbar(cp, ax=ax, orientation="vertical")
    ax.set_title("Fresnel diffraction pattern (numerical)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    X2, Y2 = np.meshgrid(x_asm, y_asm)
    cp = ax.contourf(X2, Y2, np.abs(u_out_asm_bl), locator=ticker.LogLocator())
    fig = plt.gcf()
    fig.colorbar(cp, ax=ax, orientation="vertical")
    ax.set_title("Angular spectrum diffraction pattern (numerical)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    plt.show()


if __name__ == "__main__":
    prop()
