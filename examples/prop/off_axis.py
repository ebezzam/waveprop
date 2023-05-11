"""
Off-axis optical wave propagation.

Show that shift Fresnel is not valid in near-field.

"""

import hydra
import os
import numpy as np
import matplotlib.pyplot as plt
import time
from waveprop.util import rect2d, sample_points, plot2d
from waveprop.fresnel import fresnel_one_step, shifted_fresnel
from waveprop.rs import angular_spectrum
from waveprop.condition import fresnel_goodman, fresnel_saleh
import matplotlib


@hydra.main(version_base=None, config_path="../configs", config_name="off_axis")
def off_axis(config):

    # plotting parameters
    matplotlib.rc("font", **config.plot.font)

    # simulation parameters
    N = config.sim.N  # number of grid points per size
    L = config.sim.L  # total size of grid
    wv = config.sim.wv  # wavelength
    d1 = L / N  # source-plane grid spacing

    diam = config.sim.diam  # side length of aperture [m]
    dz_vals = config.sim.dz_vals  # distance [m]
    out_shift = d1 * N * config.sim.relative_shift

    """ discretize aperture """
    x1, y1 = sample_points(N=N, delta=d1)
    u_in = rect2d(x1, y1, diam)
    plot2d(x1.squeeze(), y1.squeeze(), u_in, title="Aperture")
    plt.savefig("1_input.png", dpi=config.plot.dpi)

    """ loop through distances """
    fig, ax_cross = plt.subplots(ncols=len(dz_vals))
    for i, dz in enumerate(dz_vals):

        print("\nPROPAGATION DISTANCE : {} m".format(dz))

        """ Shifted Fresnel """
        start_time = time.time()
        u_out_sfres, x2_sfres, y2_sfres = shifted_fresnel(
            u_in, wv, d1, dz, d2=d1, out_shift=out_shift
        )
        print("Fresnel : {} s".format(time.time() - start_time))
        u_out_fres, _, _ = fresnel_one_step(u_in, wv, d1, dz)  # TODO fix/remove scaling
        u_out_sfres /= np.max(np.abs(u_out_sfres))
        u_out_sfres *= np.max(np.abs(u_out_fres))
        print("-- ", end="")
        fresnel_goodman(wv, dz, x1, y1, x2_sfres, y2_sfres)
        print("-- ", end="")
        fresnel_saleh(wv, dz, x2_sfres, y2_sfres)

        """ Shifted Angular Spectrum"""
        start_time = time.time()
        u_out_asm, x_asm, y_asm = angular_spectrum(
            u_in=u_in, wv=wv, d1=d1, dz=dz, bandlimit=True, out_shift=out_shift
        )
        print("BLAS : {} s".format(time.time() - start_time))

        """ Plot """
        # plot y2 = 0 cross-section
        idx_sf = y2_sfres[:, 0] == 0
        idx_asm = y_asm[:, 0] == 0

        ax_cross[i].plot(x2_sfres[0], np.abs(u_out_sfres[:, idx_sf]), label="fresnel")
        ax_cross[i].plot(x_asm[0], np.abs(u_out_asm[:, idx_asm]), label="blas")
        ax_cross[i].set_xlabel("x[m]")
        ax_cross[i].set_title("y = 0, {} m".format(dz))
        ax_cross[i].set_yscale("log")
        if i == len(dz_vals) - 1:
            ax_cross[i].legend()

        # plot outputs
        _, ax_2d = plt.subplots(ncols=2)
        plot2d(
            x2_sfres.squeeze(),
            y2_sfres.squeeze(),
            np.abs(u_out_sfres),
            ax=ax_2d[0],
            title="Fresnel {} m".format(dz),
        )
        plot2d(
            x_asm.squeeze(),
            y_asm.squeeze(),
            np.abs(u_out_asm),
            ax=ax_2d[1],
            title="BLAS {} m".format(dz),
        )
        plt.savefig(f"2_outputs_dz={dz}.png", dpi=config.plot.dpi)

    # save fig
    fig.savefig("3_cross_sections.png", dpi=config.plot.dpi)

    print(f"\nSaved figures to {os.getcwd()}")


if __name__ == "__main__":
    off_axis()
