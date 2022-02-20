"""
python examples/incoherent_source_adafruit_slm.py --deadspace --pytorch \
--slm_pattern data/slm_pattern_20200802.npy

# TODO : rotation of SLM

"""


import numpy as np
import time
import progressbar
import torch
import os
from waveprop.util import sample_points, plot2d, rect2d, ft2
from waveprop.rs import angular_spectrum, _zero_pad
from waveprop.slm import get_active_pixel_dim, get_slm_mask
import matplotlib.pyplot as plt
from waveprop.dataset_util import load_dataset, Datasets
from waveprop.spherical import spherical_prop
from waveprop.pytorch_util import fftconvolve as fftconvolve_torch
from scipy.signal import fftconvolve
from waveprop.color import ColorSystem, rgb2gray
import click


@click.command()
@click.option(
    "--dataset",
    type=click.Choice([Datasets.MNIST, Datasets.CIFAR10, Datasets.FLICKR8k]),
    default=Datasets.MNIST,
    help="Dataset to use as input data.",
)
@click.option("--idx", type=int, default=50, help="Index from dataset.")
@click.option(
    "--input_pad",
    type=float,
    default=2,
    help="How many times to pad input scene along both dimensions.",
)
@click.option("--down", type=int, default=6, help="Downsample factor.")
@click.option("--deadspace", is_flag=True, help="Whether to model deadspace.")
@click.option("--pytorch", is_flag=True, help="Whether to use PyTorch, or NumPy.")
@click.option(
    "--device",
    default="cuda",
    type=click.Choice(["cpu", "cuda"]),
    help="Which device to use with PyTorch.",
)
@click.option("--grayscale", is_flag=True, help="Whether output should be grayscale.")
@click.option(
    "--noise_std", type=float, default=0.001, help="Gaussian noise standard deviation at sensor."
)
@click.option("--snr", type=float, default=40, help="Signal-to-noise ratio at camera.")
@click.option(
    "--crop_fact",
    type=float,
    default=0.7,
    help="Fraction of sensor that is left uncropped, centered.",
)
@click.option("--d", type=float, default=0.4, help="Scene to SLM/mask distance in meters.")
@click.option("--z", type=float, default=0.004, help="SLM/mask to sensor distance in meters.")
@click.option(
    "--slm_pattern",
    type=str,
    help="Filepath to SLM pattern (optional), otherwise randomly generate one.",
)
@click.option(
    "--pattern_shift",
    default=None,
    nargs=2,
    type=int,
    help="Shift from center of SLM pattern that is read, in case SLM is not centered in setup.",
)
@click.option(
    "--first_color",
    default=0,
    type=click.Choice([0, 1, 2]),
    help="Color of first row of SLM, R:0, G:1, or B:2.",
)
def incoherent_simulation(
    dataset,
    idx,
    input_pad,
    down,
    deadspace,
    pytorch,
    device,
    grayscale,
    noise_std,
    snr,
    crop_fact,
    d,
    z,
    slm_pattern,
    pattern_shift,
    first_color,
):
    assert crop_fact > 0
    assert crop_fact < 1
    assert d > 0
    assert z > 0
    if slm_pattern is not None:
        assert os.path.exists(slm_pattern)

    # SLM parameters (Adafruit screen)
    slm_dim = [128 * 3, 160]
    slm_pixel_dim = np.array([0.06e-3, 0.18e-3])  # RGB sub-pixel
    slm_size = [28.03e-3, 35.04e-3]

    # RPi HQ camera datasheet: https://www.arducam.com/sony/imx477/#imx477-datasheet
    rpi_dim = [3040, 4056]
    rpi_pixel_dim = [1.55e-6, 1.55e-6]

    # polychromatric
    wv = np.array([460, 550, 640]) * 1e-9  # restricted to these wavelengths when using RGB images
    cs = ColorSystem(wv=wv)

    if pytorch:
        dtype = torch.float32
        ctype = torch.complex64
    else:
        dtype = np.float32
        ctype = np.complex64

    N = np.array([rpi_dim[0] // down, rpi_dim[1] // down])
    target_dim = N.tolist()

    """ determining overlapping region and number of SLM pixels """
    rpi_dim_m = np.array(rpi_dim) * np.array(rpi_pixel_dim)
    overlapping_mask_size, overlapping_mask_dim, n_active_slm_pixels = get_active_pixel_dim(
        sensor_dim=rpi_dim,
        sensor_pixel_size=rpi_pixel_dim,
        sensor_crop=crop_fact,
        slm_size=slm_size,
        slm_dim=slm_dim,
        slm_pixel_size=slm_pixel_dim,
    )
    print("RPi sensor dimensions [m] :", rpi_dim_m)
    print("Overlapping SLM dimensions [m] :", overlapping_mask_size)
    print("Number of overlapping SLM pixels :", overlapping_mask_dim)
    print("Number of active SLM pixels :", n_active_slm_pixels)

    """ load input image """
    d1 = np.array(overlapping_mask_size) / N
    x1, y1 = sample_points(N=N, delta=d1)

    # load dataset
    ds = load_dataset(
        dataset,
        target_dim=target_dim,
        device=device,
        pad=input_pad,
        grayscale=False,
        vflip=True,
        # for Flickr8
        root_dir="/home/bezzam/Documents/Datasets/Flickr8k/images",
        captions_file="/home/bezzam/Documents/Datasets/Flickr8k/captions.txt",
    )

    # get image
    input_image = ds[idx][0]
    print("\n-- Input image")
    print("label", ds[idx][1])
    if pytorch:
        print("device", input_image.device)
    else:
        input_image = input_image.cpu().numpy()
    print("shape", input_image.shape)
    print("dtype", input_image.dtype)
    print("minimum : ", input_image.min().item())
    print("maximum : ", input_image.max().item())

    # plot
    if pytorch:
        plot2d(x1.squeeze(), y1.squeeze(), input_image.cpu(), title="input")
    else:
        plot2d(x1.squeeze(), y1.squeeze(), input_image, title="input")

    """ propagate free space, far-field """
    spherical_wavefront = spherical_prop(input_image, d1, cs.wv, d, return_psf=True)
    print("shape", spherical_wavefront.shape)
    print("dtype", spherical_wavefront.dtype)

    # check PSF for closest to red
    red_idx = np.argmin(np.abs(cs.wv - 640e-9))
    plot_title = f"{d}m spherical wavefront (phase), wv={1e9 * cs.wv[red_idx]:.2f}nm"
    if pytorch:
        plot2d(
            x1.squeeze(),
            y1.squeeze(),
            np.angle(spherical_wavefront[red_idx].cpu()),
            title=plot_title,
        )
    else:
        plot2d(x1.squeeze(), y1.squeeze(), np.angle(spherical_wavefront[red_idx]), title=plot_title)

    """ discretize aperture (some SLM pixels will overlap due to coarse sampling) """
    mask = get_slm_mask(
        slm_dim,
        slm_size,
        slm_pixel_dim,
        rpi_dim,
        rpi_pixel_dim,
        crop_fact,
        N,
        slm_pattern=slm_pattern,
        deadspace=deadspace,
        pattern_shift=pattern_shift,
        pytorch=pytorch,
        device=device,
        dtype=dtype,
        first_color=first_color,
    )

    # plot input
    print("\n-- aperture")
    print(mask.shape)
    print(mask.dtype)
    if torch.is_tensor(mask):
        plot2d(x1.squeeze(), y1.squeeze(), mask.detach().cpu(), title="Aperture")
    else:
        plot2d(x1.squeeze(), y1.squeeze(), mask, title="Aperture")

    """ after mask / aperture """
    print("\n-- after aperture")
    u_in = mask * spherical_wavefront
    print(u_in.shape)
    print(u_in.dtype)
    plot_title = f"After mask/aperture (phase), wv={1e9 * cs.wv[red_idx]:.2f}nm"
    if pytorch:
        plot2d(x1.squeeze(), y1.squeeze(), np.angle(u_in[red_idx].cpu().detach()), title=plot_title)
    else:
        plot2d(x1.squeeze(), y1.squeeze(), np.angle(u_in[red_idx]), title=plot_title)

    """ mask-to-sensor simulation """
    if pytorch:
        psfs = torch.zeros(u_in.shape, dtype=ctype).to(device)
    else:
        psfs = np.zeros(u_in.shape, dtype=ctype)
    bar = progressbar.ProgressBar()
    start_time = time.time()

    if deadspace:
        # precompute aperture FT
        u_in_cent = rect2d(x1, y1, slm_pixel_dim)
        aperture_pad = _zero_pad(u_in_cent)
        aperture_ft = ft2(aperture_pad, delta=d1).astype(np.complex64)
        if pytorch:
            aperture_ft = torch.tensor(aperture_ft, dtype=ctype).to(device)

    for i in bar(range(cs.n_wavelength)):
        if deadspace:
            # # shift same aperture in the frequency domain, not enough memory..
            # psf_wv, x2, y2 = angular_spectrum(
            #     u_in=spherical_wavefront[i],
            #     wv=cs.wv[i],
            #     d1=d1,
            #     dz=dz,
            #     aperture_ft=aperture_ft,
            #     in_shift=centers,
            #     weights=mask_flat,
            #     dtype=dtype,
            #     device=device,
            # )
            # not perfect squares
            psf_wv, x2, y2 = angular_spectrum(
                u_in=u_in[i], wv=cs.wv[i], d1=d1, dz=z, dtype=dtype, device=device
            )
        else:
            psf_wv, x2, y2 = angular_spectrum(
                u_in=u_in[i], wv=cs.wv[i], d1=d1, dz=z, dtype=dtype, device=device
            )
        psfs[i] = psf_wv

    print(f"Computation time (wavelength simulations) [s]: {time.time() - start_time}")

    plot_title = f"At sensor (phase), wv={1e9 * cs.wv[red_idx]:.2f}nm"
    if pytorch:
        plot2d(x1.squeeze(), y1.squeeze(), np.angle(psfs[red_idx].cpu().detach()), title=plot_title)
    else:
        plot2d(x1.squeeze(), y1.squeeze(), np.angle(psfs[red_idx]), title=plot_title)

    plot_title = f"At sensor (intensity PSF)"
    if pytorch:
        psfs_int = torch.square(torch.abs(psfs))
        plot2d(x1.squeeze(), y1.squeeze(), psfs_int.cpu().detach(), title=plot_title)
    else:
        psfs_int = np.abs(psfs) ** 2
        plot2d(x1.squeeze(), y1.squeeze(), psfs_int, title=plot_title)

    print("\n--intensity PSF info")
    print("SHAPE : ", psfs_int.shape)
    print("DTYPE : ", psfs_int.dtype)
    print("MINIMUM : ", psfs_int.min().item())
    print("MAXIMUM : ", psfs_int.max().item())

    """ convolve with intensity PSF """
    if pytorch:
        out = fftconvolve_torch(input_image, psfs_int, axes=(-2, -1)) / (
            torch.numel(input_image) / 3
        )
        out = torch.clip(out, min=0)
    else:
        out = fftconvolve(input_image, psfs_int, axes=(-2, -1), mode="same") / (
            input_image.size / 3
        )
        out = np.clip(out, a_min=0, a_max=None)

    # adding noise
    if noise_std > 0:
        # -- measured black level: # https://www.strollswithmydog.com/pi-hq-cam-sensor-performance/
        black_level = 256.3 / 4095  # TODO: use as mean?
        noise = np.random.normal(loc=black_level, scale=noise_std, size=out.shape).astype(
            np.float32
        )
        if pytorch:
            noise = torch.tensor(noise.astype(np.float32), dtype=dtype, device=device)
            signal_var = torch.linalg.norm(out, axis=(1, 2))
            noise_var = torch.linalg.norm(noise, axis=(1, 2))
            signal_fact = noise_var * 10 ** (snr / 10) / signal_var
            signal_fact = signal_fact.unsqueeze(1).unsqueeze(2)
        else:
            signal_var = np.linalg.norm(out, axis=(1, 2))
            noise_var = np.linalg.norm(noise, axis=(1, 2))
            signal_fact = (noise_var * 10 ** (snr / 10) / signal_var)[:, np.newaxis, np.newaxis]
        out = signal_fact * out + noise

    if grayscale:
        out = rgb2gray(out)

    print("\n--out info")
    print("SHAPE : ", out.shape)
    print("DTYPE : ", out.dtype)
    print("MINIMUM : ", out.min().item())
    maxval = out.max().item()
    print("MAXIMUM : ", maxval)

    if pytorch:

        """Test backward"""
        t = torch.ones_like(out)

        loss = torch.nn.MSELoss().to(device)
        loss_val = loss(out, t)
        loss_val.backward()

    # plot normalize
    plot_title = "At sensor (intensity of image)"
    out /= maxval
    if pytorch:
        plot2d(x1.squeeze(), y1.squeeze(), out.cpu().detach(), title=plot_title)
    else:
        plot2d(x1.squeeze(), y1.squeeze(), out, title=plot_title)

    plt.show()


if __name__ == "__main__":
    incoherent_simulation()
