"""
TODO : polychromatic simulation
TODO : deadspace

Amplitude modulation

"""

import numpy as np
import time
import progressbar
from waveprop.util import sample_points, plot2d, gamma_correction, rect2d
from waveprop.rs import angular_spectrum
from waveprop.color import ColorSystem
import matplotlib.pyplot as plt


# TODO incorporate RGB subpixel
slm_dim = [128 * 3, 160]
slm_pixel_dim = np.array([0.06e-3, 0.18e-3])  # RGB sub-pixel
# slm_dim = [128, 160]
# slm_pixel_dim = [0.18e-3, 0.18e-3]
slm_size = [28.03e-3, 35.04e-3]

# RPi HQ camera datasheet: https://www.arducam.com/sony/imx477/#imx477-datasheet
rpi_dim = [3040, 4056]
rpi_pixel_dim = [1.55e-6, 1.55e-6]
dz = 5e-3
# dz = 1
sensor_crop_fraction = 0.8

# polychromatic
plot_int = False
gain = 1e9
n_wavelength = 100
cs = ColorSystem(n_wavelength)

# rough estimate of dead space between pixels
dead_space = np.array(slm_size) - np.array(slm_pixel_dim) * np.array(slm_dim)
dead_space_bw_pix = dead_space / (np.array(slm_dim) - 1)
dead_space_bw_pix = slm_pixel_dim

""" create gray scale mask """
# output region
x2, y2 = sample_points(N=rpi_dim, delta=rpi_pixel_dim)
sensor_size = [
    rpi_dim[0] * rpi_pixel_dim[0],
    rpi_dim[1] * rpi_pixel_dim[1],
]
print(f"\nSensor size: ", sensor_size)
print("Sensor y range: ", y2.min(), y2.max() + rpi_pixel_dim[1])
print("Sensor x range: ", x2.min(), x2.max() + rpi_pixel_dim[0])

# overlapping mask
overlapping_mask_pixels = (np.array(sensor_size) + dead_space_bw_pix) / (
    np.array(slm_pixel_dim) + dead_space_bw_pix
)
overlapping_mask_pixels = np.ceil(overlapping_mask_pixels).astype(
    int
)  # a bit more than sensor so we can simulate over full sensor
overlapping_mask_pixels[0] = overlapping_mask_pixels[0] // 3 * 3  # need all three sub-pixels
overlapping_mask_size = [
    overlapping_mask_pixels[0] * slm_pixel_dim[0],
    overlapping_mask_pixels[1] * slm_pixel_dim[1],
]
print(f"\nOverlapping mask size: ", overlapping_mask_size)
print(f"Overlapping mask dimension: ", overlapping_mask_pixels)

# crop out a region of mask that is a bit less than sensor region
active_mask_size = [
    rpi_dim[0] * rpi_pixel_dim[1] * sensor_crop_fraction,
    rpi_dim[1] * rpi_pixel_dim[1] * sensor_crop_fraction,
]
print(f"\nCropped region ({sensor_crop_fraction * 100}%): ", active_mask_size)
n_active_slm_pixels = (np.array(active_mask_size) + dead_space_bw_pix) / (
    np.array(slm_pixel_dim) + dead_space_bw_pix
)
n_active_slm_pixels = n_active_slm_pixels.astype(np.int)
n_active_slm_pixels[0] = n_active_slm_pixels[0] // 3 * 3  # need all three sub-pixels
# TODO round to odd number for now for easy viz and processing
n_active_slm_pixels = n_active_slm_pixels // 2 * 2 + 1
print("Active mask dimension: ", n_active_slm_pixels)
active_mask_size = [
    n_active_slm_pixels[0] * slm_pixel_dim[0],
    n_active_slm_pixels[1] * slm_pixel_dim[1],
]
print(f"Active mask size: ", active_mask_size)


centers_y = np.arange(n_active_slm_pixels[0])[:, np.newaxis] * (
    slm_pixel_dim[0] + dead_space_bw_pix[0]
)
centers_y -= np.mean(centers_y)
centers_x = np.arange(n_active_slm_pixels[1])[np.newaxis, :] * (
    slm_pixel_dim[1] + dead_space_bw_pix[1]
)
centers_x -= np.mean(centers_x)
centers = np.array(np.meshgrid(centers_y, centers_x)).T.reshape(-1, 2)

# generate cropped random mask
# TODO: for polychromatic each pixel will be RGB!! or slightly shifted masks
x1, y1 = sample_points(N=overlapping_mask_pixels, delta=slm_pixel_dim)
u_in_cent = rect2d(x1, y1, D=slm_pixel_dim)

# TODO incorporate random mask, pass to angular spectrum?
random_mask = np.random.rand(*n_active_slm_pixels)
# u_in = np.zeros(overlapping_mask_pixels)
# random_mask = np.random.rand(*n_active_slm_pixels)
# u_in[: n_active_slm_pixels[0], : n_active_slm_pixels[1]] = random_mask
# shift = ((np.array(overlapping_mask_pixels) - np.array(n_active_slm_pixels)) / 2).astype(int)
# u_in = np.roll(u_in, shift=shift, axis=(0, 1))
# np.testing.assert_equal(u_in.shape, overlapping_mask_pixels)
# x1, y1 = sample_points(N=overlapping_mask_pixels, delta=slm_pixel_dim)
plot2d(x1, y1, u_in_cent, title="Single SLM cell")

""" discretize aperture """
u_in = np.zeros_like(u_in_cent)
for _center in centers:
    pix_off = _center / (slm_pixel_dim + dead_space_bw_pix)

    u_in += rect2d(
        x1, y1, slm_pixel_dim, offset=pix_off.astype(int) * (slm_pixel_dim + dead_space_bw_pix)
    )
plot2d(x1, y1, u_in, title="SLM")

# raise ValueError

""" loop over wavelengths for simulation """
u_out = np.zeros((len(cs.wv),) + u_in_cent.shape, dtype=np.float32)
bar = progressbar.ProgressBar()
start_time = time.time()

for i in bar(range(cs.n_wavelength)):
    # -- propagate with angular spectrum (pyFFS)
    u_out_wv, x2, y2 = angular_spectrum(
        u_in=u_in_cent * gain, wv=cs.wv[i], d1=slm_pixel_dim, dz=dz, pyffs=True, in_shift=centers
    )
    if plot_int:
        res = np.real(u_out_wv * np.conjugate(u_out_wv))
    else:
        res = np.abs(u_out_wv)
    u_out[i] = res

# convert to RGB
rgb = cs.to_rgb(u_out, clip=True)

# gamma correction
rgb = gamma_correction(rgb, gamma=2.4)

# reshape back
rgb = (rgb.T).reshape((u_in_cent.shape[0], u_in_cent.shape[1], 3))

print(f"Computation time: {time.time() - start_time}")

# plot
plot2d(x2, y2, np.abs(rgb), title="BLAS {} m".format(dz))

plt.show()
