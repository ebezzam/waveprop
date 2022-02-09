"""

TODO : check clipping and gamma correction code
- give warning for clipping

"""

import os
import imageio
import progressbar
import time
import numpy as np
from waveprop.util import sample_points, plot2d, rect2d
from waveprop.rs import angular_spectrum
import matplotlib.pyplot as plt
from waveprop.color import ColorSystem

N = 128  # number of grid points per size
L = 1e-2  # total size of grid
diam = 2e-3  # diameter of aperture [m]
gain = 1e9
gamma = 2.4
plot_int = False  # or amplitude
build_gif = True
pyffs = True  # doesn't make a difference if same input/output resolution
n_wavelength = 10
dz_vals = (
    # list(np.arange(start=1, stop=10, step=2, dtype=int) * 1e-3) +
    list(np.arange(start=1, stop=10, step=1) * 1e-2)
    + list(np.arange(start=1, stop=10, step=1) * 1e-1)
    + list(np.arange(start=1, stop=11, step=1, dtype=int))
)

# diam = 1.55e-6 * 3040 * 0.6  # diameter of aperture [m]
# L = 10 * diam  # total size of grid
# print(diam)
# dz_vals = [0.01]

d1 = L / N  # source-plane grid spacing

""" prepare color system """
cs = ColorSystem(n_wavelength)
# cs = ColorSystem(wv=[600e-9, 635e-9])

""" discretize aperture """
x1, y1 = sample_points(N=N, delta=d1)
u_in = rect2d(x1, y1, diam)

""" loop over distance """
fig, ax = plt.subplots()
plot_pause = 0.01
dz_vals = np.around(dz_vals, decimals=3)
if build_gif:
    filenames = []
    frames = []
for dz in dz_vals:
    """loop over wavelengths for simulation"""
    # TODO : easily grow too large! Break into partitions??
    u_out = np.zeros((u_in.shape[0], u_in.shape[1], len(cs.wv)), dtype=np.float32)
    bar = progressbar.ProgressBar()
    start_time = time.time()
    for i in bar(range(cs.n_wavelength)):
        # -- propagate with angular spectrum (pyFFS)
        u_out_wv, x2, y2 = angular_spectrum(
            u_in=u_in * gain, wv=cs.wv[i], d1=d1, dz=dz, pyffs=pyffs
        )
        if plot_int:
            res = np.real(u_out_wv * np.conjugate(u_out_wv))
        else:
            res = np.abs(u_out_wv)
        u_out[:, :, i] = res

    # convert to RGB
    rgb = cs.to_rgb(u_out, clip=True, gamma=gamma)

    print(f"Computation time: {time.time() - start_time}")

    plot2d(x2, y2, rgb, title="BLAS {} m".format(dz), ax=ax)
    plt.draw()
    plt.pause(plot_pause)

    if build_gif:
        filename = f"{dz}.png"
        filenames.append(filename)
        plt.savefig(filename)
        frames.append(imageio.imread(filename))

if build_gif:
    imageio.mimsave("square_poly.gif", frames, "GIF", duration=0.3)
    for filename in set(filenames):
        os.remove(filename)

plt.show()
