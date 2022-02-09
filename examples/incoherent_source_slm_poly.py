import numpy as np
import time
import progressbar
import torch
from waveprop.util import sample_points, plot2d, rect2d, ft2
from waveprop.rs import angular_spectrum, _zero_pad
from waveprop.slm import get_centers, get_deadspace, get_active_pixel_dim
import matplotlib.pyplot as plt
from waveprop.dataset_util import FlickrDataset, CIFAR10Dataset, MNISTDataset
from waveprop.spherical import spherical_prop
import cv2
import torch.nn.functional as F
from waveprop.pytorch_util import fftconvolve as fftconvolve_torch
from scipy.signal import fftconvolve
from waveprop.color import ColorSystem, rgb2gray


idx = 50
slm_dim = [128 * 3, 160]
slm_pixel_dim = np.array([0.06e-3, 0.18e-3])  # RGB sub-pixel
slm_size = [28.03e-3, 35.04e-3]
downsample_factor = 8
deadspace = True  # TODO : not using freq approach as too slow
pytorch = True
dataset = "MNIST"
device = "cuda"  # "cpu" or "cuda"
grayscale = True  # convert to grayscale at the end
input_pad = 2  # fraction wrt to input image

# polychromatric
wv = np.array([460, 550, 640]) * 1e-9  # restricted to these wavelengths when using RGB images
# n_wavelength = 100
cs = ColorSystem(
    # n_wavelength=n_wavelength,
    wv=wv
)

# RPi HQ camera datasheet: https://www.arducam.com/sony/imx477/#imx477-datasheet
rpi_dim = [3040, 4056]
rpi_pixel_dim = [1.55e-6, 1.55e-6]
source_distance = 0.4  # [m]
dz = 0.005  # mask to sensor
sensor_crop_fraction = 0.85

if pytorch:
    dtype = torch.float32
    ctype = torch.complex64
else:
    dtype = np.float32
    ctype = np.complex64

N = np.array([rpi_dim[0] // downsample_factor, rpi_dim[1] // downsample_factor])
target_dim = N.tolist()

# rough estimate of dead space between pixels
dead_space_pix = get_deadspace(slm_size, slm_dim, slm_pixel_dim)
pixel_pitch = slm_pixel_dim + dead_space_pix

""" determining overlapping region and number of SLM pixels """
rpi_dim_m = np.array(rpi_dim) * np.array(rpi_pixel_dim)
overlapping_mask_size, overlapping_mask_dim, n_active_slm_pixels = get_active_pixel_dim(
    sensor_dim=rpi_dim,
    sensor_pixel_size=rpi_pixel_dim,
    sensor_crop=sensor_crop_fraction,
    slm_size=slm_size,
    slm_dim=slm_dim,
    slm_pixel_size=slm_pixel_dim,
    deadspace=deadspace,
)
print("RPi sensor dimensions [m] :", rpi_dim_m)
print("Overlapping SLM dimensions [m] :", overlapping_mask_size)
print("Number of overlapping SLM pixels :", overlapping_mask_dim)
print("Number of active SLM pixels :", n_active_slm_pixels)

""" load input image """
d1 = np.array(overlapping_mask_size) / N
x1, y1 = sample_points(N=N, delta=d1)

if dataset == "MNIST":
    """MNIST - 60'000 examples of 28x28"""
    ds = MNISTDataset(target_dim=target_dim, device=device, pad=input_pad, grayscale=False)

elif dataset == "CIFAR":
    """CIFAR10 - 50;000 examples of 32x32"""
    ds = CIFAR10Dataset(target_dim=target_dim, device=device, pad=input_pad, grayscale=False)

elif dataset == "FLICKR":
    """Flickr8k - varied, around 400x500"""
    ds = FlickrDataset(
        root_dir="/home/bezzam/Documents/Datasets/Flickr8k/images",
        captions_file="/home/bezzam/Documents/Datasets/Flickr8k/captions.txt",
        target_dim=target_dim,
        device=device,
        pad=input_pad,
        grayscale=False,
    )

else:
    raise ValueError("Not supported dataset...")

input_image = ds[idx][0].squeeze()
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
spherical_wavefront = spherical_prop(input_image, d1, cs.wv, source_distance, return_psf=True)
print("shape", spherical_wavefront.shape)
print("dtype", spherical_wavefront.dtype)

# check PSF for closest to red
red_idx = np.argmin(np.abs(cs.wv - 640e-9))
plot_title = f"{source_distance}m spherical wavefront (phase), wv={1e9 * cs.wv[red_idx]:.2f}nm"
if pytorch:
    plot2d(
        x1.squeeze(), y1.squeeze(), np.angle(spherical_wavefront[red_idx].cpu()), title=plot_title
    )
else:
    plot2d(x1.squeeze(), y1.squeeze(), np.angle(spherical_wavefront[red_idx]), title=plot_title)

""" discretize aperture (some SLM pixels will overlap due to coarse sampling) """
if deadspace:
    u_in = np.zeros((3, len(y1), x1.shape[1]), dtype=np.float32)
    mask = np.random.rand(*n_active_slm_pixels).astype(np.float32)
    mask_flat = mask.reshape(-1)
    if pytorch:
        mask_flat = torch.tensor(mask_flat, dtype=dtype, device=device, requires_grad=True)
        u_in = torch.tensor(u_in, dtype=dtype, device=device)

    centers, cf = get_centers(
        n_active_slm_pixels, pixel_pitch=pixel_pitch, return_color_filter=True
    )
    for i, _center in enumerate(centers):
        ap = rect2d(x1, y1, slm_pixel_dim, offset=_center).astype(np.float32)
        ap = np.tile(ap, (3, 1, 1)) * cf[:, i][:, np.newaxis, np.newaxis]
        if pytorch:
            # TODO : is pytoroch autograd compatible with this??
            u_in += torch.tensor(ap, dtype=dtype, device=device) * mask_flat[i]
        else:
            u_in += ap * mask_flat[i]

else:
    u_in = np.zeros((3,) + tuple(overlapping_mask_dim), dtype=np.float32)
    for i in range(n_active_slm_pixels[0]):
        u_in[i % 3, i, : n_active_slm_pixels[1]] = 1
    shift = ((np.array(overlapping_mask_dim) - np.array(n_active_slm_pixels)) / 2).astype(int)
    u_in = np.roll(u_in, shift=shift, axis=(1, 2))

    if pytorch:
        u_in = torch.tensor(u_in.astype(np.float32), dtype=dtype, device=device)
        mask = torch.rand(*overlapping_mask_dim, dtype=dtype, device=device, requires_grad=True)
        u_in *= mask
        u_in = F.interpolate(
            u_in.unsqueeze(0).unsqueeze(0), size=(3,) + tuple(N.tolist()), mode="nearest"
        )
        u_in = u_in.squeeze()
    else:
        mask = np.random.rand(*overlapping_mask_dim).astype(np.float32)
        u_in *= mask
        u_in = cv2.resize(
            np.transpose(u_in, (1, 2, 0)), dsize=(N[1], N[0]), interpolation=cv2.INTER_NEAREST
        )
        u_in = np.transpose(u_in, (2, 0, 1))

# plot input
print("\n-- aperture")
print(u_in.shape)
print(u_in.dtype)
if torch.is_tensor(u_in):
    plot2d(x1.squeeze(), y1.squeeze(), u_in.detach().cpu(), title="Aperture")
else:
    plot2d(x1.squeeze(), y1.squeeze(), u_in, title="Aperture")

""" after mask / aperture """
print("\n-- after aperture")
u_in = u_in * spherical_wavefront
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
        # shift same aperture in the frequency domain
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
            u_in=u_in[i], wv=cs.wv[i], d1=d1, dz=dz, dtype=dtype, device=device
        )
    else:
        psf_wv, x2, y2 = angular_spectrum(
            u_in=u_in[i], wv=cs.wv[i], d1=d1, dz=dz, dtype=dtype, device=device
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

""" convolve with intensity PSF """
if pytorch:
    out = fftconvolve_torch(input_image, psfs_int, axes=(-2, -1))
    out = torch.clip(out, min=0)
else:
    out = fftconvolve(input_image, psfs_int, axes=(-2, -1), mode="same")
    out = np.clip(out, a_min=0, a_max=None)

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

plot_title = "At sensor (intensity of image)"
out /= maxval
if pytorch:
    plot2d(x1.squeeze(), y1.squeeze(), out.cpu().detach(), title=plot_title)
else:
    plot2d(x1.squeeze(), y1.squeeze(), out, title=plot_title)

plt.show()
