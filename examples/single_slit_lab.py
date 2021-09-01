"""
Simulating rectangular aperture for our lab setup.

"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import click

from waveprop.util import rect2d, sample_points
from waveprop.prop import angular_spectrum
from waveprop.fresnel import fresnel_two_step
from waveprop.fraunhofer import fraunhofer, fraunhofer_prop_rect_ap
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
@click.option("--length", type=float, default=1e-2)
@click.option("--width", type=float, default=40e-6)
@click.option("--r_in", type=float, default=None)
@click.option("--r_out", type=float, default=5e-3)
@click.option("--n_grid", type=int, default=1024)  # number of grid points per side
@click.option("--wv", type=float, default=635e-9)  # wavelength [m]
def prop(dz, length, width, r_in, r_out, n_grid, wv):

    if r_in is None:
        r_in = length / 2
    d1 = 2 * r_in / n_grid  # source-plane grid spacing

    print("\nPROPAGATION DISTANCE : {} m".format(dz))

    # dz_fraun = distance_from_output_region(wv, r_out=1e-2, tol=1)
    delta = 2 * r_out / n_grid
    x2, y2 = sample_points(n_grid, delta)

    """ discretize aperture """
    x1, y1 = sample_points(N=n_grid, delta=d1)
    u_in = rect2d(x1, y1, D=[width, length])

    """ Fraunhofer propagation """

    print("\nFraunhofer propagation")
    print("-" * 30)
    u_out_fraun = fraunhofer_prop_rect_ap(wv, dz, x2, y2, width, length)
    # u_out_fraun_num, x2_fraun, y2_fraun = fraunhofer(u_in, wv, d1, dz)

    # check condition
    fraunhofer_schmidt(wv, dz, length)
    fraunhofer_goodman(wv, dz, x1=length / 2, y1=length / 2, x2=x2, y2=y2)
    fraunhofer_saleh(wv, dz, x1=length / 2, y1=length / 2, x2=x2, y2=y2)

    """ Fresnel propagation """
    print("\nFresnel propagation")
    print("-" * 30)
    d2 = x2[0][1] - x2[0][0]
    u_out_fres, x2_fres, y2_fres = fresnel_two_step(u_in=u_in, wv=wv, d1=d1, d2=d2, dz=dz)

    # check condition
    fresnel_goodman(wv, dz, x1=x1, y1=y1, x2=x2, y2=y2)
    fresnel_saleh(wv, dz, x=x2, y=y2)

    """ Angular spectrum """
    u_out_asm_bl, x_asm, y_asm = angular_spectrum(u_in=u_in, wv=wv, delta=d1, dz=dz, bandlimit=True)

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
    # plt.plot(
    #     x2_fraun[0] * 1e3, np.abs(u_out_fraun_num[:, idx]), color="purple",
    #     label="fraunhofer (numerical)"
    # )
    plt.axvline(
        r_out_fres,
        label="fresnel boundary - {:.2f} mm".format(r_out_fres),
        color="blue",
        linestyle="dashed",
    )
    plt.plot(x2[0] * 1e3, np.abs(u_out_fres[idx][0]), color="blue", label="fresnel (numerical)")
    plt.plot(x_asm[0] * 1e3, np.abs(u_out_asm_bl[:, idx]), color="orange", label="angular spectrum")
    plt.xlim([0, r_out * 1e3])

    plt.xlabel("x [mm]")
    plt.yscale("log")
    plt.legend()

    # plot input
    X1, Y1 = np.meshgrid(x1, y1)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    cp = ax.contourf(X1, Y1, u_in)
    fig = plt.gcf()
    fig.colorbar(cp, ax=ax, orientation="vertical")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    # plot output
    X2, Y2 = np.meshgrid(x2, y2)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    cp = ax.contourf(X2, Y2, np.abs(u_out_fraun), locator=ticker.LogLocator())
    fig = plt.gcf()
    fig.colorbar(cp, ax=ax, orientation="vertical")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Fraunhofer (theoretical)")

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    cp = ax.contourf(X2, Y2, np.abs(u_out_fres), locator=ticker.LogLocator())
    fig = plt.gcf()
    fig.colorbar(cp, ax=ax, orientation="vertical")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Fresnel (numerical)")

    fig = plt.figure()
    X2, Y2 = np.meshgrid(x_asm, y_asm)
    ax = fig.add_subplot(1, 1, 1)
    cp = ax.contourf(X2, Y2, np.abs(u_out_asm_bl), locator=ticker.LogLocator())
    fig = plt.gcf()
    fig.colorbar(cp, ax=ax, orientation="vertical")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Angular spectrum")

    plt.show()


if __name__ == "__main__":
    prop()
