"""
Circular aperture pattern in Fraunhofer regime
----------------------------------------------

Similar to Listing 4.2 from "Numerical Simulation of Optical Wave Propagation with Examples in
MATLAB" (2010).

Added Fresnel propagation and direct integration for comparison.

"""


import numpy as np
import matplotlib.pyplot as plt

from util import circ, sample_points
from prop import (
    fraunhofer,
    fraunhofer_prop_circ_ap,
    fresnel_one_step,
    direct_integration,
)
from condition import fraunhofer_schmidt, fraunhofer_goodman, fraunhofer_saleh


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

""" Direct integration (ground truth) """
u_out_di = direct_integration(u_in, wv, d1, dz, x=x2[0], y=[0])

""" Plot """
# plot y2 = 0 cross-section
idx = y2[:, 0] == 0
plt.figure()
plt.plot(x2[0], np.abs(u_out_fraun[:, idx]), label="fraunhofer (numerical)")
plt.plot(x2[0], np.abs(u_out_fraun_th[:, idx]), label="fraunhofer (theoretical)")
plt.plot(x2_fres[0], np.abs(u_out_fres[:, idx]), label="fresnel (numerical)")
plt.plot(x2[0], np.abs(u_out_di[0]), label="direct integration")

plt.legend()
plt.xlim([-xlim, xlim])

# plot input
X1, Y1 = np.meshgrid(x1, y1)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
cp = ax.contourf(X1, Y1, u_in)
fig = plt.gcf()
fig.colorbar(cp, ax=ax, orientation="vertical")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")

# plot outputs
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
X2, Y2 = np.meshgrid(x2, y2)
cp = ax.contourf(X2, Y2, np.abs(u_out_fraun))
fig = plt.gcf()
fig.colorbar(cp, ax=ax, orientation="vertical")
ax.set_title("Fraunhofer diffraction pattern")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")


plt.show()
