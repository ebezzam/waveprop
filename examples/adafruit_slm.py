"""

Amplitude modulation

# TODO give subpixel unique wavelength

"""

import numpy as np
import time
import progressbar
from waveprop.util import sample_points, plot2d, gamma_correction, rect2d
from waveprop.rs import angular_spectrum
from waveprop.color import ColorSystem
from waveprop.slm import get_centers, get_deadspace, get_active_pixel_dim
import matplotlib.pyplot as plt


slm_dim = [128 * 3, 160]
slm_pixel_dim = np.array([0.06e-3, 0.18e-3])  # RGB sub-pixel
# slm_dim = [128, 160]
# slm_pixel_dim = [0.18e-3, 0.18e-3]
slm_size = [28.03e-3, 35.04e-3]
N = 128
deadspace = True  # model deadspace (much slower)
pyffs = True  # won't be used if interpolation is not needed

# RPi HQ camera datasheet: https://www.arducam.com/sony/imx477/#imx477-datasheet
rpi_dim = [3040, 4056]
rpi_pixel_dim = [1.55e-6, 1.55e-6]
dz = 5e-3
# dz = 1e-2
sensor_crop_fraction = 0.5

gamma = 2.4
# polychromatic
plot_int = False
gain = 1e9
n_wavelength = 100
cs = ColorSystem(n_wavelength=n_wavelength)
# wavelength = np.array([450e-9, 520e-9, 638e-9])  # wavelength of each color
# cs = ColorSystem(wv=wavelength)

# rough estimate of dead space between pixels
dead_space_pix = get_deadspace(slm_size, slm_dim, slm_pixel_dim)
pixel_pitch = slm_pixel_dim + dead_space_pix

""" determining overlapping region and number of SLM pixels """
overlapping_mask_size, overlapping_mask_dim, n_active_slm_pixels = get_active_pixel_dim(
    sensor_dim=rpi_dim,
    sensor_pixel_size=rpi_pixel_dim,
    sensor_crop=sensor_crop_fraction,
    slm_size=slm_size,
    slm_dim=slm_dim,
    slm_pixel_size=slm_pixel_dim,
    deadspace=deadspace,
)
print(overlapping_mask_size)
print(overlapping_mask_dim)
print(n_active_slm_pixels)

""" generate random mask """
centers = get_centers(n_active_slm_pixels, pixel_pitch=pixel_pitch)
mask = np.random.rand(*n_active_slm_pixels)
mask_flat = mask.reshape(-1)

""" discretize aperture (some SLM pixels will overlap due to coarse sampling) """
if deadspace:
    d1 = np.array(overlapping_mask_size) / N
    x1, y1 = sample_points(N=N, delta=d1)
    u_in = np.zeros((len(y1), x1.shape[1]))
    for i, _center in enumerate(centers):
        u_in += rect2d(x1, y1, slm_pixel_dim, offset=_center) * mask_flat[i]
else:
    u_in = np.zeros(overlapping_mask_dim)
    random_mask = np.random.rand(*n_active_slm_pixels)
    u_in[: n_active_slm_pixels[0], : n_active_slm_pixels[1]] = random_mask
    shift = ((np.array(overlapping_mask_dim) - np.array(n_active_slm_pixels)) / 2).astype(int)
    u_in = np.roll(u_in, shift=shift, axis=(0, 1))
    np.testing.assert_equal(u_in.shape, overlapping_mask_dim)
    x1, y1 = sample_points(N=overlapping_mask_dim, delta=slm_pixel_dim)

# plot input
plot2d(x1.squeeze(), y1.squeeze(), u_in, title="Aperture")

""" simulate """
if deadspace:
    # shift same aperture in the frequency domain
    u_in_cent = rect2d(x1, y1, slm_pixel_dim)
u_out = np.zeros((u_in.shape[0], u_in.shape[1], len(cs.wv)), dtype=np.float32)
bar = progressbar.ProgressBar()
start_time = time.time()
for i in bar(range(cs.n_wavelength)):

    if deadspace:
        u_out_wv, x2, y2 = angular_spectrum(
            u_in=u_in_cent * gain,
            wv=cs.wv[i],
            d1=d1,
            dz=dz,
            in_shift=centers,
            weights=mask_flat,
            pyffs=pyffs,
        )
    else:
        u_out_wv, x2, y2 = angular_spectrum(
            u_in=u_in * gain, wv=cs.wv[i], d1=slm_pixel_dim, dz=dz, pyffs=pyffs
        )

    if plot_int:
        res = np.real(u_out_wv * np.conjugate(u_out_wv))
    else:
        res = np.abs(u_out_wv)
    u_out[:, :, i] = res

# convert to RGB
rgb = cs.to_rgb(u_out, clip=True, gamma=gamma)

print(f"Computation time: {time.time() - start_time}")

if deadspace:
    plot2d(x2, y2, rgb, title="BLAS {} m, superimposed Fourier".format(dz))
else:
    plot2d(x2, y2, rgb, title="BLAS {} m".format(dz))


plt.show()
