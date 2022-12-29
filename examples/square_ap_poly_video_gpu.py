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

N = 512  # number of grid points per size
L = 1e-2  # total size of grid
diam = 2e-3  # diameter of aperture [m]
gain = 1e9
gamma = 2.2
plot_int = False  # or amplitude
build_gif = True
fps = 20
n_wavelength = 20
dz_vals = list(np.arange(start=1, stop=100, step=1) * 1e-2)
n_jobs = min(multiprocessing.cpu_count(), n_wavelength)

# diam = 1.55e-6 * 3040 * 0.6  # diameter of aperture [m]
# L = 10 * diam  # total size of grid
# print(diam)
# dz_vals = [0.01]

d1 = L / N  # source-plane grid spacing

use_cuda = torch.cuda.is_available()
if use_cuda:
    device = "cuda:0"
else:
    device = "cpu"

""" prepare color system """
cs = ColorSystem(n_wavelength)
# cs = ColorSystem(wv=[600e-9, 635e-9])

""" discretize aperture """
x, y = sample_points(N=N, delta=d1)
u_in = torch.tensor(rect2d(x, y, diam).astype(np.float32), device=device)

""" loop over distance """
start_time_tot = time.time()
fig, ax = plt.subplots()
plot_pause = 0.01
dz_vals = np.around(dz_vals, decimals=3)
if build_gif:
    filenames = []
    frames = []

def simulate(i):
    u_out_wv, _, _ = angular_spectrum(
        u_in=u_in * gain, wv=cs.wv[i], d1=d1, dz=dz, device=device
    )
    if plot_int:
        res = torch.real(u_out_wv * np.conjugate(u_out_wv))
    else:
        res = torch.abs(u_out_wv)
    return res

for dz in dz_vals:
    """loop over wavelengths for simulation"""

    start_time = time.time()
    
    u_out = Parallel(n_jobs=n_jobs)(delayed(simulate)(i) for i in range(cs.n_wavelength))
    u_out = torch.stack(u_out).permute(1, 2, 0)

    # u_out = torch.zeros((u_in.shape[0], u_in.shape[1], len(cs.wv))).to(u_in)

    # bar = progressbar.ProgressBar()
    # start_time = time.time()

    # for i in bar(range(cs.n_wavelength)):
    #     # -- propagate with angular spectrum (pyFFS)
    #     u_out_wv, x2, y2 = angular_spectrum(
    #         u_in=u_in * gain, wv=cs.wv[i], d1=d1, dz=dz, device=device
    #     )
    #     if plot_int:
    #         res = torch.real(u_out_wv * np.conjugate(u_out_wv))
    #     else:
    #         res = torch.abs(u_out_wv)
    #     u_out[:, :, i] = res

    # convert to RGB
    rgb = cs.to_rgb(u_out.cpu().numpy(), clip=True, gamma=gamma)

    print(f"Computation time: {time.time() - start_time}")

    plot2d(x, y, rgb, title="BLAS {} m".format(dz), ax=ax)
    plt.draw()
    plt.pause(plot_pause)

    if build_gif:
        filename = f"{dz}.png"
        filenames.append(filename)
        plt.savefig(filename)
        frames.append(imageio.imread(filename))

if build_gif:
    imageio.mimsave("square_poly.gif", frames, "GIF", fps=fps)
    for filename in set(filenames):
        os.remove(filename)

print(f"Total computation time: {time.time() - start_time_tot}")

plt.show()
