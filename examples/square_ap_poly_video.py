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
L = 1e-2  # total size of grid
diam = 1e-3  # diameter of aperture [m]
gain = 1e9
plot_int = False  # or amplitude
pyffs = False  # doesn't make a difference if same input/output resolution
n_wavelength = 10
dz_vals = (
    list(np.arange(start=1, stop=10, step=2, dtype=int) * 1e-3)
    + list(np.arange(start=1, stop=10, step=1) * 1e-2)
    + list(np.arange(start=1, stop=10, step=1) * 1e-1)
    + list(np.arange(start=1, stop=11, step=1, dtype=int))
)
d1 = L / N  # source-plane grid spacing

""" prepare color system """
cs = ColorSystem(n_wavelength)

""" discretize aperture """
x1, y1 = sample_points(N=N, delta=d1)
u_in = rect2d(x1, y1, diam)

""" loop over distance """
fig, ax = plt.subplots()
plot_pause = 0.01
dz_vals = np.around(dz_vals, decimals=3)
for dz in dz_vals:
    """loop over wavelengths for simulation"""
    # TODO : easily grow too large! Break into partitions??
    u_out = np.zeros((len(cs.wv),) + u_in.shape, dtype=np.float32)
    bar = progressbar.ProgressBar()
    start_time = time.time()
    for i in bar(range(cs.n_wavelength)):
        # -- propagate with angular spectrum (pyFFS)
        u_out_wv, x2, y2 = angular_spectrum(
            u_in=u_in * gain, wv=cs.wv[i], d1=d1, dz=dz, pyffs=pyffs
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
    rgb = (rgb.T).reshape((u_in.shape[0], u_in.shape[1], 3))

    print(f"Computation time: {time.time() - start_time}")

    plot2d(x2, y2, rgb, title="BLAS {} m".format(dz), ax=ax)
    plt.draw()
    plt.pause(plot_pause)

plt.show()
