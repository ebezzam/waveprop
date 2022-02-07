import numpy as np
import torch
import torch.nn.functional as F
from waveprop.util import sample_points, plot2d, rect2d
from waveprop.rs import angular_spectrum
from waveprop.slm import get_centers, get_deadspace, get_active_pixel_dim
import matplotlib.pyplot as plt
import cv2


slm_dim = [128 * 3, 160]
slm_pixel_dim = np.array([0.06e-3, 0.18e-3])  # RGB sub-pixel
slm_size = [28.03e-3, 35.04e-3]
downsample_factor = 16
wv = 640e-9  # red wavelength
dz = 0.005
sensor_crop_fraction = 0.7
deadspace = True  # model deadspace (much slower for many wavelengths)
pytorch = True
device = "cuda"  # "cpu" or "cuda"
seed = 11

# RPi HQ camera datasheet: https://www.arducam.com/sony/imx477/#imx477-datasheet
rpi_dim = [3040, 4056]
rpi_pixel_dim = [1.55e-6, 1.55e-6]

if pytorch:
    dtype = torch.float32
else:
    dtype = np.float32

N = np.array([rpi_dim[0] // downsample_factor, rpi_dim[1] // downsample_factor])

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

""" discretize aperture (some SLM pixels will overlap due to coarse sampling) """
d1 = np.array(overlapping_mask_size) / N
x1, y1 = sample_points(N=N, delta=d1)

if seed is not None:
    rng = np.random.seed(seed)
    mask_init = np.random.rand(*overlapping_mask_dim)
else:
    mask_init = np.random.rand(*overlapping_mask_dim)

if deadspace:
    # mask = np.random.rand(*n_active_slm_pixels)
    # to have equivalent initialization as no deadspace
    shift = ((np.array(overlapping_mask_dim) - np.array(n_active_slm_pixels)) / 2).astype(int)
    mask = np.roll(mask_init, shift=-shift, axis=(0, 1))[
        : n_active_slm_pixels[0], : n_active_slm_pixels[1]
    ]
    mask_flat = mask.reshape(-1).astype(np.float32)
    if pytorch:
        mask_flat = torch.tensor(mask_flat, dtype=dtype, device=device, requires_grad=True)
    u_in = np.zeros((len(y1), x1.shape[1])).astype(np.float32)

    centers = get_centers(n_active_slm_pixels, pixel_pitch=pixel_pitch)
    for i, _center in enumerate(centers):
        if pytorch:
            u_in += (
                rect2d(x1, y1, slm_pixel_dim, offset=_center) * mask_flat.detach().cpu().numpy()[i]
            )
        else:
            u_in += rect2d(x1, y1, slm_pixel_dim, offset=_center) * mask_flat[i]

else:
    u_in = np.zeros(overlapping_mask_dim)
    u_in[: n_active_slm_pixels[0], : n_active_slm_pixels[1]] = 1
    shift = ((np.array(overlapping_mask_dim) - np.array(n_active_slm_pixels)) / 2).astype(int)
    u_in = np.roll(u_in, shift=shift, axis=(0, 1))
    if pytorch:
        u_in = torch.tensor(u_in.astype(np.float32), dtype=dtype, device=device)
        mask = torch.tensor(
            mask_init.astype(np.float32), dtype=dtype, device=device, requires_grad=True
        )
        # mask = torch.rand(*overlapping_mask_dim, dtype=dtype, device=device, requires_grad=True)
        u_in *= mask
        u_in = F.interpolate(u_in.unsqueeze(0).unsqueeze(0), size=N.tolist(), mode="nearest")
        u_in = u_in.squeeze()
    else:
        mask = mask_init.astype(dtype)
        u_in *= mask
        u_in = cv2.resize(u_in, dsize=(N[1], N[0]), interpolation=cv2.INTER_NEAREST)

# plot input
print("-- u_in")
print(u_in.shape)
print(u_in.dtype)
if torch.is_tensor(u_in):
    plot2d(x1.squeeze(), y1.squeeze(), u_in.detach().cpu(), title="Aperture")
else:
    plot2d(x1.squeeze(), y1.squeeze(), u_in, title="Aperture")

""" simulate """
if deadspace:
    # shift same aperture in the frequency domain
    u_in_cent = rect2d(x1, y1, slm_pixel_dim)

    u_out, x2, y2 = angular_spectrum(
        u_in=u_in_cent,
        wv=wv,
        d1=d1,
        dz=dz,
        in_shift=centers,
        weights=mask_flat,
        dtype=dtype,
        device=device,
    )
else:

    u_out, x2, y2 = angular_spectrum(u_in=u_in, wv=wv, d1=d1, dz=dz, dtype=dtype, device=device)

print("--u_out info")
print(u_out.shape)
print(u_out.dtype)
if pytorch:

    """Test backward"""
    I = torch.abs(u_out) ** 2
    t = torch.ones_like(I)

    loss = torch.nn.MSELoss().to(device)
    loss_val = loss(I, t)
    loss_val.backward()
    # Get warning from backward for deadspace = False:
    # [W Copy.cpp:240] Warning: Casting complex values to real discards the imaginary part (function operator())

    """ to numpy for plotting """
    u_out = u_out.detach().cpu().numpy()

if deadspace:
    _title = "BLAS {} m, superimposed Fourier".format(dz)
else:
    _title = "BLAS {} m".format(dz)
plot2d(x2, y2, np.abs(u_out) ** 2, title=_title)

plt.show()
