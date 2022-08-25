"""
Apply iterative Fourier Transform algorithm (Gerchberg Saxton) for determining 
phase pattern.

Propagate this pattern to see how it converges to target amplitude and then
diverges. Use multiprocessing and PyTorch for propagation so that it can be 
parallelized and on GPUs if available.

"""

import numpy as np
import os
import imageio
import time
import matplotlib.pyplot as plt
import torch
from joblib import Parallel, delayed
import multiprocessing
from waveprop.util import sample_points, plot2d, gamma_correction, resize, zero_pad
from waveprop.rs import angular_spectrum
from waveprop.io import load_image
from waveprop.holography import gerchberg_saxton


# -- input parameters
fp = "data/lcav.png"
invert = True  # black to white
N = 400  # number of grid points per size

# -- phase retrieval parameters
pad = (N // 2, N // 2)  # linearize convolution
n_iter = 100

# -- propagation param
device = "cpu"  # None to automatically detect if GPU and use
N_sim = 2400

L = 30e-3  # total size of grid
pattern_size = 10e-3
f_lens = 50e-2
wv = 532 * 1e-9
gain = 0.15
dz_vals = np.around(np.arange(start=0, stop=100, step=5) * 1e-2, decimals=2)
n_jobs = 15

# # shorter/smaller propagation
# L = 10e-3
# pattern_size = 3e-3
# f_lens = 50e-3
# wv = 532 * 1e-9
# gain = 0.15
# dz_vals = np.around(np.arange(start=0, stop=100, step=5) * 1e-3, decimals=3)
# n_jobs = 15

# -- plotting param
fps = 10
gamma = 2.2

bn = os.path.basename(fp).split(".")[0]

""" Gerchberg-Saxton """
# prepare target image
target_amp = load_image(fp, size=(N, N), invert=invert, grayscale=True)
target_amp = np.pad(target_amp, ((pad[1], pad[1]), (pad[0], pad[0])), "constant")

# get phase
start_time = time.time()
source_phase = gerchberg_saxton(target_amp=target_amp, n_iter=n_iter)
print(f"Gerchberg-Saxton computation time: {time.time() - start_time}")


""" Propagate """

if device is None:
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = "cuda:0"
    else:
        device = "cpu"

# resize and pad
N_pattern = int(np.round(pattern_size / L * N_sim))
source_phase = resize(img=source_phase, shape=(N_pattern, N_pattern))
u_in = np.exp(1j * source_phase) * gain
pad_amount = ((N_sim - N_pattern) // 2, (N_sim - N_pattern) // 2)
u_in = zero_pad(u_in, pad=pad_amount)

# cast to Tensor
u_in = torch.tensor(u_in.astype(np.complex64), device=device)

# lens transmission function
d1 = L / N_sim  # source-plane grid spacing
x, y = sample_points(N=N_sim, delta=d1)
lens_t = torch.tensor(
    np.exp(-1j * np.pi / (wv * f_lens) * (x**2 + y**2)).astype(np.complex64), device=device
)

print(f"\nUsing {n_jobs} jobs for propagation sweep")
filenames = []
frames = []
fig, ax = plt.subplots()


def simulate(dz):

    # propagate
    u_out_wv, _, _ = angular_spectrum(
        u_in=u_in * lens_t, wv=wv, d1=d1, dz=dz, device=device, pad=False
    )
    res = torch.abs(u_out_wv)

    u_out = np.zeros((N_sim, N_sim, 3))
    u_out[:, :, 1] = res.cpu().numpy()

    if gamma is not None:
        u_out = gamma_correction(u_out, gamma=gamma)

    if np.max(u_out) > 1:
        print(f"Clipping for {dz}m : {np.max(u_out)}")
        u_out[u_out > 1.0] = 1.0

    plot2d(x, y, u_out, title="BLAS {} m".format(dz), ax=ax)
    plt.draw()

    filename = f"{dz}.png"

    plt.savefig(filename)


# parallelize propagation
start_time = time.time()
Parallel(n_jobs=n_jobs)(delayed(simulate)(dz) for dz in dz_vals)
print(f"Propagation computation time: {time.time() - start_time}")


""" Create GIF """
for dz in dz_vals:

    filename = f"{dz}.png"
    filenames.append(filename)
    if dz == f_lens:
        for _ in range(fps):
            frames.append(imageio.imread(filename))
    else:
        frames.append(imageio.imread(filename))


gif_fp = f"{bn}.gif"
imageio.mimsave(gif_fp, frames, "GIF", fps=fps)
for filename in set(filenames):
    os.remove(filename)
print(f"GIF saved to : {gif_fp}")
