"""
Compare different propagation methods for a square aperture, as the propagation distance is increases.

Store comparison as GIF.


"""

import hydra
import os
import imageio
import progressbar
import numpy as np
import time
from waveprop.util import sample_points, plot2d, rect2d
from waveprop.fresnel import shifted_fresnel
from waveprop.rs import angular_spectrum, fft_di
from waveprop.fraunhofer import fraunhofer
import matplotlib.pyplot as plt


@hydra.main(version_base=None, config_path="../configs", config_name="square_ap_gif")
def square_ap_gif(config):

    N = config.sim.N  # number of grid points per size
    L = config.sim.L  # total size of grid
    diam = config.sim.diam  # diameter of aperture [m]
    wv = config.sim.wv  # wavelength
    gif_duration = config.gif_duration  
    dz_vals = (
        # list(np.arange(start=1, stop=10, step=2, dtype=int) * 1e-3) +
        list(np.arange(start=1, stop=10, step=1) * 1e-2)
        + list(np.arange(start=1, stop=10, step=1) * 1e-1)
        + list(np.arange(start=1, stop=11, step=1, dtype=int))
    )
    d1 = L / N  # source-plane grid spacing

    """ discretize aperture """
    x1, y1 = sample_points(N=N, delta=d1)
    u_in = rect2d(x1, y1, diam)
    plot2d(x1, y1, u_in, title="Aperture")

    # -- propagate with angular spectrum (pyFFS)
    fraunhofer_time = []
    blas_time = []
    as_time = []
    fresnel_time = []
    fft_di_time = []
    _, ax = plt.subplots(ncols=5, figsize=(25, 5))
    dz_vals = np.around(dz_vals, decimals=3)

    bar = progressbar.ProgressBar()
    filenames = []
    frames = []
    for dz in bar(dz_vals):
        start_time = time.time()
        # apply Fraunhofer
        u_out_fraun, x_fraun, y_raun = fraunhofer(u_in=u_in, wv=wv, d1=d1, dz=dz)
        fraunhofer_time.append(time.time() - start_time)

        # propagate with angular spectrum
        start_time = time.time()
        u_out_asm, x_asm, y_asm = angular_spectrum(u_in=u_in, wv=wv, d1=d1, dz=dz, bandlimit=False)
        as_time.append(time.time() - start_time)

        start_time = time.time()
        u_out_blas, x_blas, y_blas = angular_spectrum(
            u_in=u_in, wv=wv, d1=d1, dz=dz, bandlimit=True
        )
        blas_time.append(time.time() - start_time)

        # -- propagate with shifted Fresnel
        start_time = time.time()
        u_out_sfres, x2_sfres, y2_sfres = shifted_fresnel(u_in=u_in, wv=wv, d1=d1, dz=dz)
        fresnel_time.append(time.time() - start_time)

        # -- FFT-DI
        start_time = time.time()
        u_out_di, x_di, y_di = fft_di(u_in=u_in, wv=wv, d1=d1, dz=dz)
        fft_di_time.append(time.time() - start_time)

        # update plot
        plot2d(
            x_fraun,
            y_raun,
            np.abs(u_out_fraun),
            title="Fraunhofer {} m".format(dz),
            ax=ax[0],
            colorbar=False,
        )
        ax[0].set_xlim([x_asm.min(), x_asm.max()])
        ax[0].set_ylim([y_asm.min(), y_asm.max()])
        plot2d(
            x2_sfres,
            y2_sfres,
            np.abs(u_out_sfres),
            title="Fresnel {} m".format(dz),
            ax=ax[1],
            colorbar=False,
        )
        plot2d(
            x_di, y_di, np.abs(u_out_di), title="FFT-DI {} m".format(dz), ax=ax[2], colorbar=False
        )
        plot2d(
            x_asm, y_asm, np.abs(u_out_asm), title="AS {} m".format(dz), ax=ax[3], colorbar=False
        )
        plot2d(
            x_blas,
            y_blas,
            np.abs(u_out_blas),
            title="BLAS {} m".format(dz),
            ax=ax[4],
            colorbar=False,
        )
        for _ax in ax[1:]:
            _ax.set_ylabel("")
            _ax.set_yticks([])
            _ax.set_yticklabels([])

        filename = f"{dz}.png"
        filenames.append(filename)
        plt.savefig(filename)
        frames.append(imageio.imread(filename))

    plt.figure()
    plt.plot(dz_vals, fraunhofer_time, label="Fraunhofer")
    plt.plot(dz_vals, fresnel_time, label="Fresnel")
    plt.plot(dz_vals, as_time, label="ASM")
    plt.plot(dz_vals, blas_time, label="BLAS")
    plt.plot(dz_vals, fft_di_time, label="FFT-DI")
    plt.title("Computation time")
    plt.xlabel("Distance [m]")
    plt.ylabel("Seconds")
    plt.xscale("log")
    plt.legend()
    plt.savefig("computation_time.png")

    imageio.mimsave("square_ap.gif", frames, "GIF", duration=gif_duration)
    for filename in set(filenames):
        os.remove(filename)

    print(f"\nSaved figures to {os.getcwd()}")


if __name__ == "__main__":
    square_ap_gif()
