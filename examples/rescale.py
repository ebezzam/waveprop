"""
Rescale optical wave propagation.

"""


import numpy as np
import matplotlib.pyplot as plt
import time
from waveprop.util import rect2d, sample_points, plot2d, bounding_box
from waveprop.fresnel import shifted_fresnel
from waveprop.prop import angular_spectrum
from waveprop.condition import fresnel_goodman, fresnel_saleh
import matplotlib

# plotting parameters
font = {"family": "Times New Roman", "weight": "normal", "size": 10}
# matplotlib.rcParams["lines.linewidth"] = 4
matplotlib.rc("font", **font)
ALPHA = 0.7

# simulation parameters
N = 512  # number of grid points per size
L = 1e-2  # total size of grid
diam = 1e-3  # diameter of aperture [m]
wv = 1e-6  # wavelength
dz_vals = [0.01, 0.1]  # distance [m]
d1 = L / N  # source-plane grid spacing

output_scaling = 1 / 4
out_shift = d1 * N / 10
N_out = None  # shifted Fresnel doesn't give this flexibility

""" discretize aperture """
x1, y1 = sample_points(N=N, delta=d1)
u_in = rect2d(x1, y1, diam)
plot2d(x1.squeeze(), y1.squeeze(), u_in, title="Aperture")

""" loop through distances """
for i, dz in enumerate(dz_vals):

    print("\nPROPAGATION DISTANCE : {} m".format(dz))

    """ Shifted Fresnel """
    start_time = time.time()
    u_out_sfres, x2_sfres, y2_sfres = shifted_fresnel(
        u_in, wv, d1, dz, d2=output_scaling * d1, out_shift=out_shift
    )
    print("Fresnel : {} s".format(time.time() - start_time))
    print("-- ", end="")
    fresnel_goodman(wv, dz, x1, y1, x2_sfres, y2_sfres)
    print("-- ", end="")
    fresnel_saleh(wv, dz, x2_sfres, y2_sfres)

    """ Scaled BLAS """
    start_time = time.time()
    u_out_asm_scaled, x_asm_scaled, y_asm_scaled = angular_spectrum(
        u_in=u_in,
        wv=wv,
        d1=d1,
        dz=dz,
        bandlimit=True,
        N_out=N_out,
        d2=output_scaling * d1,
        out_shift=out_shift,
    )
    print("Angular spectrum : {} s".format(time.time() - start_time))
    u_out_asm, x_asm, y_asm = angular_spectrum(u_in=u_in, wv=wv, d1=d1, dz=dz, bandlimit=True)

    """ Plot """
    _, ax_2d = plt.subplots(ncols=3, figsize=(25, 5))
    plot2d(
        x2_sfres.squeeze(),
        y2_sfres.squeeze(),
        np.abs(u_out_sfres),
        title="Shifted Fresnel {} m".format(dz),
        ax=ax_2d[1],
    )
    plot2d(
        x_asm_scaled.squeeze(),
        y_asm_scaled.squeeze(),
        np.abs(u_out_asm_scaled),
        title="Scaled BLAS {} m".format(dz),
        ax=ax_2d[2],
    )
    plot2d(x_asm.squeeze(), y_asm.squeeze(), np.abs(u_out_asm), title="BLAS", ax=ax_2d[0])
    bounding_box(
        ax=ax_2d[0],
        start=[np.min(x_asm_scaled), np.min(y_asm_scaled)],
        stop=[np.max(x_asm_scaled), np.max(y_asm_scaled)],
        shift=output_scaling * d1 / 2,
        period=L,
        c="r",
        linestyle="--",
    )

plt.show()
