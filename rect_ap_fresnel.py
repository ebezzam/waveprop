"""
Rectangular aperture pattern in Fresnel regime
----------------------------------------------

Similar to Listings 6.1-6.5 from "Numerical Simulation of Optical Wave Propagation with Examples
in MATLAB" (2010).

"""


import numpy as np
import matplotlib.pyplot as plt

from util import rect, sample_points
from prop import (
    fraunhofer_prop_rect_ap,
    fresnel_one_step,
    fresnel_prop_square_ap,
    fresnel_two_step,
    fresnel_conv,
    angular_spectrum,
    direct_integration,
)
from condition import (
    fraunhofer_schmidt,
    fraunhofer_goodman,
    fraunhofer_saleh,
    fresnel_saleh,
    fresnel_goodman,
)


N = 1024  # number of grid points per size
L = 1e-2  # total size of grid
diam = 2e-3  # diameter of aperture [m]
wv = 1e-6  # wavelength
dz = 1  # distance [m]
d1 = L / N  # source-plane grid spacing

# plot param
xlim = 0.001

print("\nPROPAGATION DISTANCE : {} m".format(dz))

""" discretize aperture """
x1, y1 = sample_points(N=N, delta=d1)
u_in = rect(x1 / diam) * rect(y1 / diam)

""" Fresnel propagation """

# Fresnel simulation
u_out_fres_one_step, x2, y2 = fresnel_one_step(u_in, wv, d1, dz)
d2 = x2[0][1] - x2[0][0]
u_out_fres_two_step, _, _ = fresnel_two_step(u_in, wv, d1=d1, d2=d2, dz=dz)
u_out_fres_conv, _, _ = fresnel_conv(u_in, wv, d1=d1, d2=d2, dz=dz)

# Fresnel theoretical
u_out_th_fres = fresnel_prop_square_ap(x=x2, y=y2, width=diam, wv=wv, dz=dz)


""" Fraunhofer propagation """

print("\nFraunhofer propagation")
print("-" * 30)

# Fraunhofer simulation
u_out_fraun = fraunhofer_prop_rect_ap(wv, dz, x2, y2, diam, diam)

# check condition
fraunhofer_schmidt(wv, dz, diam)
fraunhofer_goodman(wv, dz, x1, y1, x2, y2)
fraunhofer_saleh(wv, dz, x1, y1, x2, y2)

# Fresnel condition
print("\nFresnel propagation")
print("-" * 30)
fresnel_goodman(wv, dz, x1, y1, x2, y2)
fresnel_saleh(wv, dz, x2, y2)


""" Angular spectrum """
u_out_asm, x_asm, y_asm = angular_spectrum(u_in=u_in, wv=wv, delta=d1, dz=dz, bandlimit=False)
u_out_asm_bl, _, _ = angular_spectrum(u_in=u_in, wv=wv, delta=d1, dz=dz, bandlimit=True)


""" Direct integration (ground truth) """
# TODO : something wrong in DI computation, very flat...
u_out_di = direct_integration(u_in, wv, d1, dz, x=x_asm[0], y=[0])


""" Plot """

# plot y2 = 0 cross-section
idx = y2[:, 0] == 0
plt.figure()
# plt.plot(x2[0], np.abs(u_out_fraun[:, idx]), label="fraunhofer (theoretical)")
plt.plot(x2[0], np.abs(u_out_fres_one_step[idx][0]), label="fresnel (one step)")
plt.plot(x2[0], np.abs(u_out_fres_two_step[idx][0]), label="fresnel (two step)")
plt.plot(x2[0], np.abs(u_out_fres_conv[idx][0]), label="fresnel (conv)")
plt.plot(x2[0], np.abs(u_out_th_fres[idx][0]), label="fresnel (theoretical)")
plt.plot(x_asm[0], np.abs(u_out_asm[idx][0]), label="angular spectrum (numerical)")
plt.plot(x_asm[0], np.abs(u_out_asm_bl[idx][0]), label="angular spectrum (numerical, BL)")
plt.plot(x_asm[0], np.abs(u_out_di[0]), label="direct integration")
plt.legend()

xlim = min(xlim, np.max(x_asm))
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


# plot output
xlim = np.max(x_asm)
ylim = np.max(y_asm)
X2, Y2 = np.meshgrid(x2, y2)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
cp = ax.contourf(X2, Y2, np.abs(u_out_fraun))
fig = plt.gcf()
fig.colorbar(cp, ax=ax, orientation="vertical")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_title("Fraunhofer (theoretical)")
ax.set_xlim([-xlim, xlim])
ax.set_ylim([-ylim, ylim])

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
cp = ax.contourf(X2, Y2, np.abs(u_out_th_fres))
fig = plt.gcf()
fig.colorbar(cp, ax=ax, orientation="vertical")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_title("Fresnel (theoretical)")
ax.set_xlim([-xlim, xlim])
ax.set_ylim([-ylim, ylim])

X_ASM, Y_ASM = np.meshgrid(x_asm, y_asm)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
cp = ax.contourf(X_ASM, Y_ASM, np.abs(u_out_asm_bl))
fig = plt.gcf()
fig.colorbar(cp, ax=ax, orientation="vertical")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_title("Angular spectrum (numerical, bandlimited)")
ax.set_xlim([-xlim, xlim])
ax.set_ylim([-ylim, ylim])

X_ASM, Y_ASM = np.meshgrid(x_asm, y_asm)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
cp = ax.contourf(X_ASM, Y_ASM, np.abs(u_out_asm))
fig = plt.gcf()
fig.colorbar(cp, ax=ax, orientation="vertical")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_title("Angular spectrum (numerical, non-bandlimited)")
ax.set_xlim([-xlim, xlim])
ax.set_ylim([-ylim, ylim])


plt.show()
