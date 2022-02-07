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
from waveprop.slm import get_centers

N = 256  # number of grid points per size
L = 4e-3  # total size of grid
diam = np.array([0.06e-3, 0.18e-3])  # diameter of aperture [m]
gain = 1e9
plot_int = False  # or amplitude
pyffs = False  # doesn't make a difference if same input/output resolution
n_wavelength = 10
gamma = 2.4
dz = 5e-2
pixel_grid = [23, 9]
# pixel_grid = [3, 3]
dead_space = np.array(diam) * 0.2


d1 = L / N  # source-plane grid spacing

""" prepare color system """
cs = ColorSystem(n_wavelength)

""" discretize aperture """
x1, y1 = sample_points(N=N, delta=d1)
centers = get_centers(pixel_grid, pixel_pitch=diam + dead_space)
u_in = np.zeros((len(y1), x1.shape[1]))
for _center in centers:
    u_in += rect2d(x1, y1, diam, offset=_center)

# plot input
plot2d(x1.squeeze(), y1.squeeze(), u_in, title="Aperture")

""" shift same aperture in the frequency domain """
u_in_cent = rect2d(x1, y1, diam)
u_out = np.zeros((u_in.shape[0], u_in.shape[1], len(cs.wv)), dtype=np.float32)
bar = progressbar.ProgressBar()
start_time = time.time()
for i in bar(range(cs.n_wavelength)):
    u_out_wv, x2, y2 = angular_spectrum(
        u_in=u_in_cent * gain,
        wv=cs.wv[i],
        d1=d1,
        dz=dz,
        pyffs=pyffs,
        in_shift=centers,
    )

    if plot_int:
        res = np.real(u_out_wv * np.conjugate(u_out_wv))
    else:
        res = np.abs(u_out_wv)
    u_out[:, :, i] = res

# convert to RGB
u_out_four = cs.to_rgb(u_out, clip=True, gamma=gamma)

print(f"Computation time: {time.time() - start_time}")

plot2d(x2, y2, u_out_four, title="BLAS {} m, superimposed Fourier".format(dz))

plt.show()
