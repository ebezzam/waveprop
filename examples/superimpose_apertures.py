"""

TODO : check clipping and gamma correction code
- give warning for clipping

"""


import progressbar
import time
import numpy as np
from waveprop.util import sample_points, plot2d, rect2d, gamma_correction
from waveprop.rs import angular_spectrum
import matplotlib.pyplot as plt
from waveprop.color import ColorSystem

N = 256  # number of grid points per size
L = 4e-3  # total size of grid
diam = [0.06e-3, 0.18e-3]  # diameter of aperture [m]
gain = 1e9
plot_int = False  # or amplitude
pyffs = False  # doesn't make a difference if same input/output resolution
n_wavelength = 10
dz = 5e-3
pixel_grid = [3, 3]


d1 = L / N  # source-plane grid spacing

""" prepare color system """
cs = ColorSystem(n_wavelength)

""" discretize aperture """
x1, y1 = sample_points(N=N, delta=d1)

centers_y = np.arange(pixel_grid[0])[:, np.newaxis] * (2 * diam[0])
centers_y -= np.mean(centers_y)
centers_x = np.arange(pixel_grid[1])[np.newaxis, :] * (2 * diam[1])
centers_x -= np.mean(centers_x)
centers = np.array(np.meshgrid(centers_y, centers_x)).T.reshape(-1, 2)


u_in = np.zeros((len(y1), x1.shape[1]))
for _center in centers:
    u_in += rect2d(x1, y1, diam, offset=_center)

# plot input
plot2d(x1.squeeze(), y1.squeeze(), u_in, title="Aperture")

""" simulate as one aperture"""
# TODO : easily grow too large! Break into partitions??
u_out = np.zeros((len(cs.wv),) + u_in.shape, dtype=np.float32)
bar = progressbar.ProgressBar()
start_time = time.time()
for i in bar(range(cs.n_wavelength)):
    # -- propagate with angular spectrum (pyFFS)
    u_out_wv, x2, y2 = angular_spectrum(u_in=u_in * gain, wv=cs.wv[i], d1=d1, dz=dz, pyffs=pyffs)
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
u_out_combined = (rgb.T).reshape((u_in.shape[0], u_in.shape[1], 3))

print(f"Computation time: {time.time() - start_time}")

plot2d(x2, y2, u_out_combined, title="BLAS {} m".format(dz))


""" simulate apertures individually """
u_out = np.zeros((len(cs.wv),) + u_in.shape, dtype=np.float32)
bar = progressbar.ProgressBar()
start_time = time.time()
for i in bar(range(cs.n_wavelength)):
    u_out_wv = np.zeros((len(y1), x1.shape[1]), dtype=np.complex64)

    # first aperture
    u_out_wv_1, x2, y2 = angular_spectrum(
        u_in=u_in_1 * gain, wv=cs.wv[i], d1=d1, dz=dz, pyffs=pyffs
    )
    u_out_wv_2, _, _ = angular_spectrum(u_in=u_in_2 * gain, wv=cs.wv[i], d1=d1, dz=dz, pyffs=pyffs)
    u_out_wv = u_out_wv_1 + u_out_wv_2

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
u_out_super = (rgb.T).reshape((u_in.shape[0], u_in.shape[1], 3))

print(f"Computation time: {time.time() - start_time}")

plot2d(x2, y2, u_out_super, title="BLAS {} m, superimposed".format(dz))

""" shift same aperture in the frequency domain """
u_in_cent = rect2d(x1, y1, diam)
u_out = np.zeros((len(cs.wv),) + u_in.shape, dtype=np.float32)
bar = progressbar.ProgressBar()
start_time = time.time()
for i in bar(range(cs.n_wavelength)):
    u_out_wv, x2, y2 = angular_spectrum(
        u_in=u_in_cent * gain,
        wv=cs.wv[i],
        d1=d1,
        dz=dz,
        pyffs=pyffs,
        in_shift=[[0, diam[1]], [0, -diam[1]]],
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
u_out_four = (rgb.T).reshape((u_in.shape[0], u_in.shape[1], 3))

print(f"Computation time: {time.time() - start_time}")

plot2d(x2, y2, u_out_four, title="BLAS {} m, superimposed Fourier".format(dz))


""" Compare """
# u_out_combined /= np.max(u_out_combined)
# u_out_super /= np.max(u_out_super)
# u_out_four /= np.max(u_out_four)
err = np.linalg.norm(u_out_combined - u_out_super)
print(f"Error, superimposed : {err}")
err = np.linalg.norm(u_out_combined - u_out_four)
print(f"Error, superimposed Fourier : {err}")

print(u_out_combined.max())
print(u_out_super.max())
print(u_out_four.max())

plt.show()
