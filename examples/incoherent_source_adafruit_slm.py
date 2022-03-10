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
from waveprop.util import sample_points, plot2d
from waveprop.rs import angular_spectrum
from waveprop.fresnel import fresnel_conv
from waveprop.slm import get_active_pixel_dim, get_slm_mask
import matplotlib.pyplot as plt
from waveprop.dataset_util import load_dataset, Datasets
from waveprop.spherical import spherical_prop
from waveprop.pytorch_util import fftconvolve as fftconvolve_torch
from scipy.signal import fftconvolve
from waveprop.color import ColorSystem, rgb2gray
import click
from waveprop.devices import SLMOptions, slm, SensorOptions, SensorParam, sensor


@click.command()
@click.option(
    "--dataset",
    type=click.Choice([Datasets.MNIST, Datasets.CIFAR10, Datasets.FLICKR8k]),
    default=Datasets.MNIST,
    help="Dataset to use as input data.",
)
@click.option("--idx", type=int, default=50, help="Index from dataset.")
@click.option(
    "--object_height",
    type=float,
    default=5e-2,
    help="Height of object in meters.",
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
@click.option(
    "--fresnel", is_flag=True, help="Whether to use Fresnel approximation for second propagation."
)
@click.option(
    "--full_fresnel",
    is_flag=True,
    help="Whether to use Fresnel approximation for full propagation.",
)
def incoherent_simulation(
    dataset,
    idx,
    object_height,
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
    fresnel,
    full_fresnel,
):
    assert crop_fact > 0
    assert crop_fact < 1
    assert d > 0
    assert z > 0
    if slm_pattern is not None:
        assert os.path.exists(slm_pattern)

    # SLM parameters (Adafruit screen)
    slm_config = slm[SLMOptions.ADAFRUIT.value]

    # RPi HQ camera datasheet: https://www.arducam.com/sony/imx477/#imx477-datasheet
    sensor_config = sensor[SensorOptions.RPI_HQ.value]
    target_dim = sensor_config[SensorParam.SHAPE] // down

    # polychromatric
    cs = ColorSystem.rgb()
    red_idx = np.argmin(np.abs(cs.wv - 640e-9))

    if pytorch:
        dtype = torch.float32
        ctype = torch.complex64
    else:
        dtype = np.float32
        ctype = np.complex64

    """ determining overlapping region and number of SLM pixels """
    overlapping_mask_size, overlapping_mask_dim, n_active_slm_pixels = get_active_pixel_dim(
        sensor_config=sensor_config,
        sensor_crop=crop_fact,
        slm_config=slm_config,
    )

    print("Sensor dimensions [m] :", sensor_config[SensorParam.SIZE])
    print("Overlapping SLM dimensions [m] :", overlapping_mask_size)
    print("Number of overlapping SLM pixels :", overlapping_mask_dim)
    print("Number of active SLM pixels :", n_active_slm_pixels)

    """ load input image """
    d1 = np.array(overlapping_mask_size) / target_dim
    x1, y1 = sample_points(N=target_dim, delta=d1)

    # load dataset
    ds = load_dataset(
        dataset,
        scene2mask=d,
        mask2sensor=z,
        sensor_dim=sensor_config[SensorParam.SIZE],
        object_height=object_height,
        target_dim=target_dim.tolist(),
        device=device,
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

    """ discretize aperture (some SLM pixels will overlap due to coarse sampling) """
    mask = get_slm_mask(
        slm_config=slm_config,
        sensor_config=sensor_config,
        crop_fact=crop_fact,
        target_dim=target_dim,
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

    """ propagate free space, far-field """
    if full_fresnel:
        # TODO
        print("TODO!!")
        first_prop = 1  # See Sepand's notes
        u_in = mask
    else:
        spherical_wavefront = spherical_prop(
            u_in=input_image, d1=d1, wv=cs.wv, dz=d, return_psf=True
        )
        print("shape", spherical_wavefront.shape)
        print("dtype", spherical_wavefront.dtype)

        # check PSF for closest to red
        plot_title = f"{d}m spherical wavefront (phase), wv={1e9 * cs.wv[red_idx]:.2f}nm"
        if pytorch:
            plot2d(
                x1.squeeze(),
                y1.squeeze(),
                np.angle(spherical_wavefront[red_idx].cpu()),
                title=plot_title,
            )
        else:
            plot2d(
                x1.squeeze(), y1.squeeze(), np.angle(spherical_wavefront[red_idx]), title=plot_title
            )

        """ after mask / aperture """
        print("\n-- after aperture")
        u_in = mask * spherical_wavefront
        print(u_in.shape)
        print(u_in.dtype)
        plot_title = f"After mask/aperture (phase), wv={1e9 * cs.wv[red_idx]:.2f}nm"
        if pytorch:
            plot2d(
                x1.squeeze(), y1.squeeze(), np.angle(u_in[red_idx].cpu().detach()), title=plot_title
            )
        else:
            plot2d(x1.squeeze(), y1.squeeze(), np.angle(u_in[red_idx]), title=plot_title)

    """ mask-to-sensor simulation """
    if pytorch:
        psfs = torch.zeros(u_in.shape, dtype=ctype).to(device)
    else:
        psfs = np.zeros(u_in.shape, dtype=ctype)
    bar = progressbar.ProgressBar()
    start_time = time.time()

    for i in bar(range(cs.n_wavelength)):
        if fresnel:
            psf_wv, x2, y2 = fresnel_conv(
                u_in=u_in[i], wv=cs.wv[i], d1=d1[0], dz=z, dtype=dtype, device=device
            )
        elif full_fresnel:
            dz = 1 / d + 1 / z
            psf_wv, x2, y2 = fresnel_conv(
                u_in=u_in[i], wv=cs.wv[i], d1=d1[0], dz=dz, dtype=dtype, device=device
            )
            psf_wv /= 4 * cs.wv[i] ** 2 * d * z
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
