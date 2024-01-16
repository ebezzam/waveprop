"""
Apply iterative Fourier Transform algorithm (Gerchberg Saxton) for determining phase pattern.

Propagate this pattern to see how it converges to target amplitude and then diverges.
Use multiprocessing and PyTorch for propagation so that it can be parallelized and on
GPUs if available.

Example usage:
```
python examples/holography.py --target data/lcav.png --invert --n_jobs 15
```

Propagating over smaller distances:
```
python examples/holography.py --target data/lcav.png --invert --n_jobs 15 \
--sim_size 0.01 --pattern_size 0.003 --f_lens 0.05 --z_step 0.001
```

If only interested in the holography pattern at a single distance, e.g. the focal plane,
the following command can be run, which will produce a GIF with a single image
```
python examples/holography.py --target data/lcav.png --invert \
--f_lens 0.5 --z_start 0.5 --nz 1
```

TODO
- support non-square shape
- parallelize distances over GPU

"""

import click
import numpy as np
import os
import imageio
import time
import matplotlib.pyplot as plt
import torch
from joblib import Parallel, delayed
from waveprop.util import sample_points, plot2d, gamma_correction, resize, zero_pad
from waveprop.rs import angular_spectrum
from waveprop.io import load_image
from waveprop.holography import gerchberg_saxton


@click.command()
@click.option("--target", type=str, default="data/lcav.png", help="File path of target image.")
@click.option(
    "--target_dim",
    type=int,
    default=400,
    help="Resize target to the square dimension with this side length.",
)
@click.option("--wv", type=float, default=532, help="Wavelength in [nm]")
@click.option(
    "--invert",
    is_flag=True,
    help="Whether to invert black and white, e.g. for an image with white background.",
)
@click.option(
    "--n_iter",
    type=int,
    default=100,
    help="Number of iterations for Gerchberg-Saxton phase retrieval algorithm.",
)
@click.option(
    "--device",
    type=str,
    default=None,
    help="PyTorch device to use for propagation. Defaults to GPU if available. Otherwise pass 'cpu'.",
)
@click.option("--fps", type=int, default=10, help="Frames per second for GIF.")
@click.option("--gamma", type=float, default=2.2, help="Gamma correction.")
@click.option(
    "--n_jobs",
    type=int,
    default=1,
    help="Number of CPU jobs for parallelizing simulation over distance.",
)
@click.option(
    "--sim_size", type=float, default=30e-3, help="Side length of simulation square in [m]."
)
@click.option(
    "--sim_dim", type=int, default=2400, help="Side length of simulation square in pixels."
)
@click.option(
    "--pattern_size", type=float, default=10e-3, help="Side length of pattern square in [m]."
)
@click.option("--f_lens", type=float, default=0.5, help="Focal length of lens in [m].")
@click.option("--gain", type=float, default=0.15, help="Input gain.")
@click.option("--z_start", type=float, default=0, help="Start of propagation sweep in [m].")
@click.option("--z_step", type=float, default=1e-2, help="Propagation distance step in [m].")
@click.option("--nz", type=int, default=100, help="Number of propagation steps from `z_start`.")
def holography(
    target,
    target_dim,
    wv,
    invert,
    n_iter,
    device,
    fps,
    gamma,
    n_jobs,
    sim_size,
    sim_dim,
    pattern_size,
    f_lens,
    gain,
    z_start,
    z_step,
    nz,
):

    # -- phase retrieval parameters
    pad = (target_dim // 2, target_dim // 2)  # linearize convolution

    # -- propagation param
    wv = wv * 1e-9
    if nz == 1:
        dz_vals = np.array([z_start])
    else:
        dz_vals = np.around(np.arange(start=z_start, stop=nz, step=1) * z_step, decimals=3)

    bn = os.path.basename(target).split(".")[0]

    """ Gerchberg-Saxton """
    # prepare target image
    target_amp = load_image(target, size=(target_dim, target_dim), invert=invert, grayscale=True)
    target_amp = np.pad(target_amp, ((pad[1], pad[1]), (pad[0], pad[0])), "constant")

    # get phase
    start_time = time.time()
    print(f"\nApplying Gerchberg-Saxton for {n_iter} iterations...")
    source_phase = gerchberg_saxton(target_amp=target_amp, n_iter=n_iter)
    print(f"-- Computation time: {time.time() - start_time}")

    """ Propagate """

    if device is None:
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            print("Using GPU...")
            device = "cuda:0"
        else:
            device = "cpu"

    # resize and pad
    N_pattern = int(np.round(pattern_size / sim_size * sim_dim))
    source_phase = resize(img=source_phase, shape=(N_pattern, N_pattern))
    u_in = np.exp(1j * source_phase) * gain
    pad_amount = ((sim_dim - N_pattern) // 2, (sim_dim - N_pattern) // 2)
    u_in = zero_pad(u_in, pad=pad_amount)

    # cast to Tensor
    u_in = torch.tensor(u_in.astype(np.complex64), device=device)

    # lens transmission function
    d1 = sim_size / sim_dim  # source-plane grid spacing
    x, y = sample_points(N=sim_dim, delta=d1)
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

        u_out = np.zeros((sim_dim, sim_dim, 3))
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
    print(f"-- Computation time: {time.time() - start_time}")

    """ Create GIF """
    for dz in dz_vals:

        filename = f"{dz}.png"
        filenames.append(filename)
        if dz == f_lens:
            for _ in range(fps):
                frames.append(imageio.v2.imread(filename))
        else:
            frames.append(imageio.v2.imread(filename))

    gif_fp = f"{bn}.gif"
    imageio.mimsave(gif_fp, frames, "GIF", fps=fps)
    for filename in set(filenames):
        os.remove(filename)
    print(f"\nGIF saved to : {gif_fp}")


if __name__ == "__main__":
    holography()
