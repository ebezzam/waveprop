"""
Circular aperture pattern in Fraunhofer regime
----------------------------------------------

Similar to Listing 4.2 from "Numerical Simulation of Optical Wave Propagation with Examples in
MATLAB" (2010).

Added Fresnel propagation and direct integration for comparison.

"""


import numpy as np
import matplotlib.pyplot as plt

import matplotlib

font = {"family": "Times New Roman", "weight": "normal", "size": 20}
matplotlib.rc("font", **font)
ALPHA = 0.7

from waveprop.util import circ, sample_points, plot2d
from waveprop.prop import (
    fraunhofer,
    fraunhofer_prop_circ_ap,
    fresnel_one_step,
    fft_di,
    direct_integration,
)
from waveprop.condition import fraunhofer_schmidt, fraunhofer_goodman, fraunhofer_saleh


N = 512  # number of grid points per size
L = 7.5e-3  # total size of grid
diam = 1e-3  # diameter of aperture [m]
wv = 1e-6  # wavelength
dz = 20  # distance [m]
d1 = L / N  # source-plane grid spacing

# plot param
xlim = 0.1

print("\nPROPAGATION DISTANCE : {} m".format(dz))

""" discretize aperture """
x1, y1 = sample_points(N=N, delta=d1)
u_in = circ(x=x1, y=y1, diam=diam)

""" Fraunhofer propagation """

# Fraunhofer simulation
u_out_fraun, x2, y2 = fraunhofer(u_in, wv, d1, dz)

# Fraunhofer theoretical
u_out_fraun_th = fraunhofer_prop_circ_ap(wv, dz, diam, x2, y2)

# check condition
fraunhofer_schmidt(wv, dz, diam)
fraunhofer_goodman(wv, dz, x1, y1, x2, y2)
fraunhofer_saleh(wv, dz, x1, y1, x2, y2)

""" Fresnel approximation """
u_out_fres, x2_fres, y2_fres = fresnel_one_step(u_in, wv, d1, dz)


""" FFT direct integration"""
# x2_fft_di, y2_fft_di = sample_points(N=N * 8, delta=d1)
u_out_fft_di, x2_fft_di, y2_fft_di = fft_di(u_in, wv, d1, dz, N_out=N * 8, use_simpson=True)

# """ Direct integration (ground truth) """
# u_out_di = direct_integration(u_in, wv, d1, dz, x=x2[0], y=[0])

""" Plot """
# plot y2 = 0 cross-section
idx = y2[:, 0] == 0
idx_fft_di = y2_fft_di[:, 0] == 0
plt.figure()
plt.plot(
    x2[0], np.abs(u_out_fraun[:, idx]), marker="o", label="fraunhofer (numerical)", alpha=ALPHA
)
plt.plot(
    x2[0], np.abs(u_out_fraun_th[:, idx]), marker="x", label="fraunhofer (theoretical)", alpha=ALPHA
)
plt.plot(x2_fres[0], np.abs(u_out_fres[:, idx]), label="fresnel (numerical)")
plt.plot(x2_fft_di[0], np.abs(u_out_fft_di[:, idx_fft_di]), marker="*", label="FFT-DI", alpha=ALPHA)
# plt.plot(x2[0], np.abs(u_out_di[0]), marker="*", label="direct integration", alpha=ALPHA)
plt.xlabel("x[m]")
plt.title("amplitude, y = 0")

plt.legend()
# plt.yscale("log")
if xlim is not None:
    plt.xlim([-xlim, xlim])

# plot input
ax = plot2d(x1.squeeze(), y1.squeeze(), u_in)
ax.set_title("Aperture")

# plot outputs
ax = plot2d(x2.squeeze(), y2.squeeze(), np.abs(u_out_fraun))
ax.set_title("Fraunhofer diffraction pattern")
ax.set_xlim([np.min(x2_fft_di), np.max(x2_fft_di)])
ax.set_ylim([np.min(y2_fft_di), np.max(y2_fft_di)])

ax = plot2d(x2_fft_di.squeeze(), y2_fft_di.squeeze(), np.abs(u_out_fft_di))
ax.set_title("FFT-DI diffraction pattern")


plt.show()
