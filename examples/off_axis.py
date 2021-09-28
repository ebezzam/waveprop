"""
Off-axis optical wave propagation.

"""


import numpy as np
import matplotlib.pyplot as plt
import time
from waveprop.util import rect2d, sample_points, plot2d
from waveprop.fresnel import fresnel_one_step, shifted_fresnel
from waveprop.prop import angular_spectrum, direct_integration
from waveprop.condition import fresnel_goodman, fresnel_saleh
import matplotlib

# plotting parameters
font = {"family": "Times New Roman", "weight": "normal", "size": 20}
matplotlib.rc("font", **font)
ALPHA = 0.7

# simulation parameters
N = 512  # number of grid points per size
L = 1e-2  # total size of grid
diam = 2e-3  # diameter of aperture [m]
wv = 1e-6  # wavelength
dz_vals = [0.2, 0.5]  # distance [m]
d1 = L / N  # source-plane grid spacing
out_shift = d1 * N / 2

""" discretize aperture """
x1, y1 = sample_points(N=N, delta=d1)
u_in = rect2d(x1, y1, diam)
plot2d(x1.squeeze(), y1.squeeze(), u_in, title="Aperture")

""" loop through distances """
_, ax_cross = plt.subplots(ncols=len(dz_vals), figsize=(10, 20))
for i, dz in enumerate(dz_vals):

    print("\nPROPAGATION DISTANCE : {} m".format(dz))

    """ Shifted Fresnel """
    start_time = time.time()
    u_out_sfres, x2_sfres, y2_sfres = shifted_fresnel(u_in, wv, d1, dz, d2=d1, out_shift=out_shift)
    print("Fresnel : {} s".format(time.time() - start_time))
    u_out_fres, x2_fres, y2_fres = fresnel_one_step(u_in, wv, d1, dz)  # TODO fix/remove scaling
    u_out_sfres /= np.max(np.abs(u_out_sfres))
    u_out_sfres *= np.max(np.abs(u_out_fres))
    print("-- ", end="")
    fresnel_goodman(wv, dz, x1, y1, x2_sfres, y2_sfres)
    print("-- ", end="")
    fresnel_saleh(wv, dz, x2_sfres, y2_sfres)

    """ Shifted Angular Spectrum"""
    start_time = time.time()
    u_out_asm, x_asm, y_asm = angular_spectrum(
        u_in=u_in, wv=wv, d1=d1, dz=dz, bandlimit=True, out_shift=out_shift
    )
    print("Angular spectrum : {} s".format(time.time() - start_time))

    """ Direct integration (ground truth) """
    start_time = time.time()
    u_out_di = direct_integration(u_in, wv, d1, dz, x=x_asm[0], y=[0])
    n_lines = len(y_asm)  # just computing one line!
    print("Direct integration : {} s".format((time.time() - start_time) * n_lines))

    """ Plot """
    # plot y2 = 0 cross-section
    idx_sf = y2_sfres[:, 0] == 0
    idx_asm = y_asm[:, 0] == 0

    ax_cross[i].plot(x2_sfres[0], np.abs(u_out_sfres[:, idx_sf]), label="fresnel")
    ax_cross[i].plot(x_asm[0], np.abs(u_out_asm[:, idx_asm]), label="angular spectrum")
    ax_cross[i].plot(x_asm[0], np.abs(u_out_di[0]), label="direct integration")
    ax_cross[i].set_xlabel("x[m]")
    ax_cross[i].set_title("amplitude, y = 0, {} m".format(dz))
    ax_cross[i].set_yscale("log")
    if i == len(dz_vals) - 1:
        ax_cross[i].legend()

    # plot outputs
    _, ax_2d = plt.subplots(ncols=2, figsize=(10, 20))
    plot2d(
        x2_sfres.squeeze(),
        y2_sfres.squeeze(),
        np.abs(u_out_sfres),
        ax=ax_2d[0],
        title="Fresnel {} m".format(dz),
    )
    plot2d(
        x_asm.squeeze(),
        y_asm.squeeze(),
        np.abs(u_out_asm),
        ax=ax_2d[1],
        title="Angular spectrum {} m".format(dz),
    )

plt.show()
