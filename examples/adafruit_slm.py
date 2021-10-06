"""
TODO : polychromatic simulation
TODO : deadspace

Amplitude modulation

"""

import numpy as np
from waveprop.util import sample_points, plot2d
from waveprop.fresnel import shifted_fresnel
from waveprop.rs import angular_spectrum
import matplotlib.pyplot as plt


# TODO incorporate RGB subpixel
# slm_dim = [128 * 3, 160]
# slm_pixel_dim = [0.06e-3, 0.18e-3]   # RGB sub-pixel
slm_dim = [128, 160]
slm_pixel_dim = [0.18e-3, 0.18e-3]

# RPi HQ camera datasheet: https://www.arducam.com/sony/imx477/#imx477-datasheet
rpi_dim = [3040, 4056]
rpi_pixel_dim = [1.55e-6, 1.55e-6]
dz_vals = np.linspace(2e-3, 1, num=50)
plot_pause = 0.05
wv = 635e-9  # TODO polychromatic

# # random mask
# u_in = np.random.rand(*slm_dim)
# x1, y1 = sample_points(N=slm_dim, delta=slm_pixel_dim)
# plot2d(x1.squeeze(), y1.squeeze(), u_in, title="Aperture")
# print("SLM x range: ", x1.min(), x1.max())
# print("SLM y range: ", y1.min(), y1.max())

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
overlapping_mask_pixels = np.array(sensor_size) / np.array(slm_pixel_dim)
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
sensor_crop_fraction = 0.8
active_mask_size = [
    rpi_dim[0] * rpi_pixel_dim[1] * sensor_crop_fraction,
    rpi_dim[1] * rpi_pixel_dim[1] * sensor_crop_fraction,
]
print(f"\nCropped region ({sensor_crop_fraction * 100}%): ", active_mask_size)
n_active_slm_pixels = np.array(active_mask_size) / np.array(slm_pixel_dim)
n_active_slm_pixels = n_active_slm_pixels.astype(np.int)
n_active_slm_pixels[0] = n_active_slm_pixels[0] // 3 * 3  # need all three sub-pixels
active_mask_size = [
    n_active_slm_pixels[0] * slm_pixel_dim[0],
    n_active_slm_pixels[1] * slm_pixel_dim[1],
]
print(f"Active mask size: ", active_mask_size)
print("Active mask dimension: ", n_active_slm_pixels)

# generate cropped random max
# TODO: for polychromatic each pixel will be RGB!! or slightly shifted masks
u_in = np.zeros(overlapping_mask_pixels)
random_mask = np.random.rand(*n_active_slm_pixels)
u_in[: n_active_slm_pixels[0], : n_active_slm_pixels[1]] = random_mask
shift = ((np.array(overlapping_mask_pixels) - np.array(n_active_slm_pixels)) / 2).astype(int)
u_in = np.roll(u_in, shift=shift, axis=(0, 1))
np.testing.assert_equal(u_in.shape, overlapping_mask_pixels)
x1, y1 = sample_points(N=overlapping_mask_pixels, delta=slm_pixel_dim)
plot2d(x1, y1, u_in, title="Aperture")

# -- propagate with angular spectrum (pyFFS)
_, ax = plt.subplots(ncols=3, figsize=(25, 5))
for dz in dz_vals:
    u_out_asm_pyffs, x_asm_pyffs, y_asm_pyffs = angular_spectrum(
        u_in=u_in,
        wv=wv,
        d1=slm_pixel_dim,
        dz=dz,
        pyffs=True
        # d2=d2,
        # N_out=N_out
    )
    print("\nu_out_asm_pyffs shape: ", u_out_asm_pyffs.shape)
    plot2d(
        x_asm_pyffs,
        y_asm_pyffs,
        np.abs(u_out_asm_pyffs),
        title="pyFFS BLAS {} m".format(dz),
        ax=ax[0],
        colorbar=False,
    )

    # propagate with angular spectrum
    u_out_asm, x_asm, y_asm = angular_spectrum(u_in=u_in, wv=wv, d1=slm_pixel_dim, dz=dz)
    print("u_out_asm shape: ", u_out_asm.shape)
    plot2d(x_asm, y_asm, np.abs(u_out_asm), title="BLAS {} m".format(dz), ax=ax[1], colorbar=False)
    diff = np.linalg.norm(u_out_asm_pyffs - u_out_asm)
    print(f"|| u_out_asm_pyffs - u_out_asm || = {diff}")

    # -- propagate with shifted Fresnel
    u_out_sfres, x2_sfres, y2_sfres = shifted_fresnel(u_in=u_in, wv=wv, d1=slm_pixel_dim, dz=dz)
    print("u_out_sfres shape: ", u_out_sfres.shape)
    plot2d(
        x2_sfres,
        y2_sfres,
        np.abs(u_out_sfres),
        title="Shifted Fresnel {} m".format(dz),
        ax=ax[2],
        colorbar=False,
    )

    plt.draw()
    plt.pause(plot_pause)

plt.show()
