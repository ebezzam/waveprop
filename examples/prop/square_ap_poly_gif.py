"""
A square aperture is propagated with the band-limited angular spectrum method.

A polychromatic source is used, and the resulting intensity is computed and
stored in a GIF file as the propagation distance increases.


"""

import hydra
import os
import imageio
import progressbar
import time
import numpy as np
from waveprop.util import sample_points, plot2d, rect2d
from waveprop.rs import angular_spectrum
import matplotlib.pyplot as plt
from waveprop.color import ColorSystem
import torch
from joblib import Parallel, delayed
import multiprocessing


@hydra.main(version_base=None, config_path="../configs", config_name="square_ap_poly_gif")
def square_ap_poly_gif(config):

    # simulation parameters
    N = config.sim.N  # number of grid points per size
    L = config.sim.L  # total size of grid
    diam = config.sim.diam  # side length of aperture [m]
    gain = config.sim.gain
    n_wavelength = config.sim.n_wavelength
    dz_vals = (
        list(np.arange(start=1, stop=10, step=1, dtype=int) * 1e-3)
        + list(np.arange(start=1, stop=10, step=1) * 1e-2)
        + list(np.arange(start=1, stop=11, step=1) * 1e-1)
    )
    d1 = L / N  # source-plane grid spacing

    # visualization parameters
    gamma = config.plot.gamma
    plot_int = config.plot.intensity  # or amplitude
    gif_duration = config.plot.gif_duration  # duration of gif [s]

    # compute hardware
    if config.use_cuda:
        use_cuda = torch.cuda.is_available()
    else:
        use_cuda = False
    if use_cuda:
        device = "cuda:0"
    else:
        device = "cpu"
    print("Using device: {}".format(device))

    # parallelization
    n_jobs = min(multiprocessing.cpu_count(), n_wavelength)

    """ prepare color system """
    cs = ColorSystem(n_wavelength)

    """ discretize aperture """
    x, y = sample_points(N=N, delta=d1)
    u_in = torch.tensor(rect2d(x, y, diam).astype(np.float32), device=device)

    """ loop over distance """
    start_time_tot = time.time()
    _, ax = plt.subplots()

    dz_vals = np.around(dz_vals, decimals=3)

    def simulate(i):
        u_out_wv, _, _ = angular_spectrum(
            u_in=u_in * gain, wv=cs.wv[i], d1=d1, dz=dz, device=device
        )
        res = torch.abs(u_out_wv)
        if plot_int:
            res = res**2
        return res

    bar = progressbar.ProgressBar()
    filenames = []
    frames = []
    for dz in bar(dz_vals):

        """parallelize over wavelengths"""
        u_out = Parallel(n_jobs=n_jobs)(delayed(simulate)(i) for i in range(cs.n_wavelength))
        u_out = torch.stack(u_out).permute(1, 2, 0)

        # convert to RGB
        rgb = cs.to_rgb(u_out.cpu().numpy(), clip=True, gamma=gamma)

        plot2d(x, y, rgb, title="BLAS {} m".format(dz), ax=ax)

        filename = f"{dz}.png"
        filenames.append(filename)
        plt.savefig(filename)
        frames.append(imageio.imread(filename))

    imageio.mimsave("square_poly.gif", frames, "GIF", duration=gif_duration)
    for filename in set(filenames):
        os.remove(filename)

    print(f"Total computation time: {time.time() - start_time_tot}")

    print(f"\nSaved figures to {os.getcwd()}")


if __name__ == "__main__":
    square_ap_poly_gif()
