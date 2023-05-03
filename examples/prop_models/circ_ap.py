"""
Circular aperture pattern in Fraunhofer regime
----------------------------------------------

Similar to Listing 4.2 from "Numerical Simulation of Optical Wave Propagation with Examples in
MATLAB" (2010).

Added Fresnel propagation for comparison.

"""

import hydra
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from waveprop.util import circ, sample_points, plot2d
from waveprop.fresnel import fresnel_one_step, shifted_fresnel
from waveprop.fraunhofer import fraunhofer, fraunhofer_prop_circ_ap
from waveprop.condition import fraunhofer_schmidt, fraunhofer_goodman, fraunhofer_saleh


@hydra.main(version_base=None, config_path="../configs", config_name="circ_ap")
def circ_ap(config):

    matplotlib.rc("font", **config.plot.font)

    # simulation parameters
    N = config.sim.N  # number of grid points per size
    L = config.sim.L  # total size of grid
    diam = config.sim.diam  # diameter of aperture [m]
    wv = config.sim.wv  # wavelength
    dz = config.sim.dz  # distance [m]
    d1 = L / N  # source-plane grid spacing

    # shift fresnel parameters
    output_scaling = config.shifted_fresnel.output_scaling
    # out_shift = output_scaling * d1 * N / 2
    out_shift = output_scaling * L * config.shifted_fresnel.relative_shift

    print("\nPROPAGATION DISTANCE : {} m".format(dz))

    """ discretize aperture """
    x1, y1 = sample_points(N=N, delta=d1)
    u_in = circ(x=x1, y=y1, diam=diam)

    """ Fraunhofer propagation """

    # Fraunhofer simulation
    u_out_fraun, x2, y2 = fraunhofer(u_in, wv, d1, dz)

    # Fraunhofer theoretical
    u_out_fraun_th = fraunhofer_prop_circ_ap(wv, dz, diam, x2, y2)

    # check condition
    fraunhofer_schmidt(wv, dz, diam)
    fraunhofer_goodman(wv, dz, x1, y1, x2, y2)
    fraunhofer_saleh(wv, dz, x1, y1, x2, y2)

    """ Fresnel approximation """
    u_out_fres, x2_fres, y2_fres = fresnel_one_step(u_in, wv, d1, dz)

    """ Shifted Fresnel """
    u_out_sfres, x2_sfres, y2_sfres = shifted_fresnel(
        u_in, wv, d1, dz, d2=output_scaling * d1, out_shift=out_shift
    )
    u_out_sfres /= np.max(np.abs(u_out_sfres))
    u_out_sfres *= np.max(np.abs(u_out_fres))

    """ Plot """
    # plot y2 = 0 cross-section
    idx = y2[:, 0] == 0

    idx_sf = y2_sfres[:, 0] == 0
    plt.figure()
    plt.plot(
        x2[0],
        np.abs(u_out_fraun[:, idx]),
        marker="o",
        label="fraunhofer (numerical)",
        alpha=config.plot.alpha,
    )
    plt.plot(
        x2[0],
        np.abs(u_out_fraun_th[:, idx]),
        marker="x",
        label="fraunhofer (theoretical)",
        alpha=config.plot.alpha,
    )
    plt.plot(x2_fres[0], np.abs(u_out_fres[:, idx]), label="fresnel (numerical)")
    plt.plot(x2_sfres[0], np.abs(u_out_sfres[:, idx_sf]), label="shifted fresnel")
    plt.xlabel("x[m]")
    plt.title("amplitude, y = 0")

    plt.legend()
    plt.yscale("log")
    if config.plot.xlim is not None:
        plt.xlim(config.plot.xlim)

    plt.savefig("2_prop_cross_section.png", dpi=config.plot.dpi)

    # plot input
    plot2d(x1.squeeze(), y1.squeeze(), u_in, title="Aperture")
    plt.savefig("1_input.png", dpi=config.plot.dpi)

    # plot outputs
    ax = plot2d(x2.squeeze(), y2.squeeze(), np.abs(u_out_fraun), title="Fraunhofer")
    plt.savefig("3_fraunhofer.png", dpi=config.plot.dpi)

    ax.set_xlim([np.min(x2_sfres), np.max(x2_sfres)])
    ax.set_ylim([np.min(y2_sfres), np.max(y2_sfres)])
    plot2d(x2_sfres.squeeze(), y2_sfres.squeeze(), np.abs(u_out_sfres), title="Shifted Fresnel")
    plt.savefig("4_shifted_fresnel.png", dpi=config.plot.dpi)


if __name__ == "__main__":
    circ_ap()
