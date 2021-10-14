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
from waveprop.slm import get_centers, get_deadspace, get_active_pixel_dim
import matplotlib.pyplot as plt


# TODO incorporate RGB subpixel
slm_dim = [128 * 3, 160]
slm_pixel_dim = np.array([0.06e-3, 0.18e-3])  # RGB sub-pixel
# slm_dim = [128, 160]
# slm_pixel_dim = [0.18e-3, 0.18e-3]
slm_size = [28.03e-3, 35.04e-3]
N = 128

# RPi HQ camera datasheet: https://www.arducam.com/sony/imx477/#imx477-datasheet
rpi_dim = [3040, 4056]
rpi_pixel_dim = [1.55e-6, 1.55e-6]
dz = 5e-3
# dz = 1e-2
sensor_crop_fraction = 0.5

# polychromatic
plot_int = False
gain = 1e9
n_wavelength = 10
cs = ColorSystem(n_wavelength)

# rough estimate of dead space between pixels
dead_space_pix = get_deadspace(slm_size, slm_dim, slm_pixel_dim)
pixel_pitch = slm_pixel_dim + dead_space_pix

""" determining overlapping region and number of SLM pixels """
overlapping_mask_size, n_active_slm_pixels = get_active_pixel_dim(
    sensor_dim=rpi_dim,
    sensor_pixel_size=rpi_pixel_dim,
    sensor_crop=sensor_crop_fraction,
    slm_size=slm_size,
    slm_dim=slm_dim,
    slm_pixel_size=slm_pixel_dim,
)
print(overlapping_mask_size)
print(n_active_slm_pixels)

""" generate random mask """
centers = get_centers(n_active_slm_pixels, pixel_pitch=pixel_pitch)
mask = np.random.rand(*n_active_slm_pixels)
mask_flat = mask.reshape(-1)

""" discretize aperture (some SLM pixels will overlap due to coarse sampling) """
d1 = np.array(overlapping_mask_size) / N
x1, y1 = sample_points(N=N, delta=d1)
u_in = np.zeros((len(y1), x1.shape[1]))
for i, _center in enumerate(centers):
    u_in += rect2d(x1, y1, slm_pixel_dim, offset=_center) * mask_flat[i]

# plot input
plot2d(x1.squeeze(), y1.squeeze(), u_in, title="Aperture")

""" shift same aperture in the frequency domain """
u_in_cent = rect2d(x1, y1, slm_pixel_dim)
u_out = np.zeros((len(cs.wv),) + u_in.shape, dtype=np.float32)
bar = progressbar.ProgressBar()
start_time = time.time()
for i in bar(range(cs.n_wavelength)):
    u_out_wv, x2, y2 = angular_spectrum(
        u_in=u_in_cent * gain, wv=cs.wv[i], d1=d1, dz=dz, in_shift=centers, weights=mask_flat
    )

    if plot_int:
        res = np.real(u_out_wv * np.conjugate(u_out_wv))
    else:
        res = np.abs(u_out_wv)
    u_out[i] = res

# convert to RGB
rgb = cs.to_rgb(u_out, clip=True, gamma=2.4)

# # reshape back
# u_out_four = (rgb.T).reshape((u_in.shape[0], u_in.shape[1], 3))

print(f"Computation time: {time.time() - start_time}")

plot2d(x2, y2, rgb, title="BLAS {} m, superimposed Fourier".format(dz))

# """ create gray scale mask """
# # output region
# x2, y2 = sample_points(N=rpi_dim, delta=rpi_pixel_dim)
# sensor_size = [
#     rpi_dim[0] * rpi_pixel_dim[0],
#     rpi_dim[1] * rpi_pixel_dim[1],
# ]
# print(f"\nSensor size: ", sensor_size)
# print("Sensor y range: ", y2.min(), y2.max() + rpi_pixel_dim[1])
# print("Sensor x range: ", x2.min(), x2.max() + rpi_pixel_dim[0])
#
# # overlapping mask
# overlapping_mask_pixels = (np.array(sensor_size) + dead_space_bw_pix) / (
#     np.array(slm_pixel_dim) + dead_space_bw_pix
# )
# overlapping_mask_pixels = np.ceil(overlapping_mask_pixels).astype(
#     int
# )  # a bit more than sensor so we can simulate over full sensor
# overlapping_mask_pixels[0] = overlapping_mask_pixels[0] // 3 * 3  # need all three sub-pixels
# overlapping_mask_size = [
#     overlapping_mask_pixels[0] * slm_pixel_dim[0],
#     overlapping_mask_pixels[1] * slm_pixel_dim[1],
# ]
# print(f"\nOverlapping mask size: ", overlapping_mask_size)
# print(f"Overlapping mask dimension: ", overlapping_mask_pixels)
#
# # crop out a region of mask that is a bit less than sensor region
# active_mask_size = [
#     rpi_dim[0] * rpi_pixel_dim[1] * sensor_crop_fraction,
#     rpi_dim[1] * rpi_pixel_dim[1] * sensor_crop_fraction,
# ]
# print(f"\nCropped region ({sensor_crop_fraction * 100}%): ", active_mask_size)
# n_active_slm_pixels = (np.array(active_mask_size) + dead_space_bw_pix) / (
#     np.array(slm_pixel_dim) + dead_space_bw_pix
# )
# n_active_slm_pixels = n_active_slm_pixels.astype(np.int)
# n_active_slm_pixels[0] = n_active_slm_pixels[0] // 3 * 3  # need all three sub-pixels
# # TODO round to odd number for now for easy viz and processing
# n_active_slm_pixels = n_active_slm_pixels // 2 * 2 + 1
# print("Active mask dimension: ", n_active_slm_pixels)
# active_mask_size = [
#     n_active_slm_pixels[0] * slm_pixel_dim[0],
#     n_active_slm_pixels[1] * slm_pixel_dim[1],
# ]
# print(f"Active mask size: ", active_mask_size)
#
#
# centers_y = np.arange(n_active_slm_pixels[0])[:, np.newaxis] * (
#     slm_pixel_dim[0] + dead_space_bw_pix[0]
# )
# centers_y -= np.mean(centers_y)
# centers_x = np.arange(n_active_slm_pixels[1])[np.newaxis, :] * (
#     slm_pixel_dim[1] + dead_space_bw_pix[1]
# )
# centers_x -= np.mean(centers_x)
# centers = np.array(np.meshgrid(centers_y, centers_x)).T.reshape(-1, 2)
#
# # generate cropped random mask
# # TODO: for polychromatic each pixel will be RGB!! or slightly shifted masks
# x1, y1 = sample_points(N=overlapping_mask_pixels, delta=slm_pixel_dim)
# u_in_cent = rect2d(x1, y1, D=slm_pixel_dim)
#
# # TODO incorporate random mask, pass to angular spectrum?
# random_mask = np.random.rand(*n_active_slm_pixels)
# # u_in = np.zeros(overlapping_mask_pixels)
# # random_mask = np.random.rand(*n_active_slm_pixels)
# # u_in[: n_active_slm_pixels[0], : n_active_slm_pixels[1]] = random_mask
# # shift = ((np.array(overlapping_mask_pixels) - np.array(n_active_slm_pixels)) / 2).astype(int)
# # u_in = np.roll(u_in, shift=shift, axis=(0, 1))
# # np.testing.assert_equal(u_in.shape, overlapping_mask_pixels)
# # x1, y1 = sample_points(N=overlapping_mask_pixels, delta=slm_pixel_dim)
# plot2d(x1, y1, u_in_cent, title="Single SLM cell")
#
# """ discretize aperture """
# u_in = np.zeros_like(u_in_cent)
# for _center in centers:
#     pix_off = _center / (slm_pixel_dim + dead_space_bw_pix)
#
#     u_in += rect2d(
#         x1, y1, slm_pixel_dim, offset=pix_off.astype(int) * (slm_pixel_dim + dead_space_bw_pix)
#     )
# plot2d(x1, y1, u_in, title="SLM")
#
# # raise ValueError
#
# """ loop over wavelengths for simulation """
# u_out = np.zeros((len(cs.wv),) + u_in_cent.shape, dtype=np.float32)
# bar = progressbar.ProgressBar()
# start_time = time.time()
#
# for i in bar(range(cs.n_wavelength)):
#     # -- propagate with angular spectrum (pyFFS)
#     u_out_wv, x2, y2 = angular_spectrum(
#         u_in=u_in_cent * gain, wv=cs.wv[i], d1=slm_pixel_dim, dz=dz, pyffs=True, in_shift=centers
#     )
#     if plot_int:
#         res = np.real(u_out_wv * np.conjugate(u_out_wv))
#     else:
#         res = np.abs(u_out_wv)
#     u_out[i] = res
#
# # convert to RGB
# rgb = cs.to_rgb(u_out, clip=True)
#
# # gamma correction
# rgb = gamma_correction(rgb, gamma=2.4)
#
# # reshape back
# rgb = (rgb.T).reshape((u_in_cent.shape[0], u_in_cent.shape[1], 3))
#
# print(f"Computation time: {time.time() - start_time}")
#
# # plot
# plot2d(x2, y2, np.abs(rgb), title="BLAS {} m".format(dz))

plt.show()
