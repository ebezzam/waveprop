"""
Rectangular aperture pattern in Fresnel regime
----------------------------------------------

Similar to Listings 6.1-6.5 from "Numerical Simulation of Optical Wave Propagation with Examples
in MATLAB" (2010).

"""


import numpy as np
import matplotlib.pyplot as plt

from waveprop.util import rect2d, sample_points, plot2d
from waveprop.prop import (
    angular_spectrum,
    direct_integration,
)
from waveprop.fresnel import (
    fresnel_one_step,
    fresnel_two_step,
    fresnel_conv,
    fresnel_prop_square_ap,
)
from waveprop.fraunhofer import fraunhofer_prop_rect_ap
from waveprop.condition import (
    fraunhofer_schmidt,
    fraunhofer_goodman,
    fraunhofer_saleh,
    fresnel_saleh,
    fresnel_goodman,
)


N = 512  # number of grid points per size
L = 1e-2  # total size of grid
diam = 2e-3  # diameter of aperture [m]
wv = 1e-6  # wavelength
dz = 1  # distance [m]
d1 = L / N  # source-plane grid spacing

# plot param
xlim = 0.01
log_scale = False

print("\nPROPAGATION DISTANCE : {} m".format(dz))

""" discretize aperture """
x1, y1 = sample_points(N=N, delta=d1)
u_in = rect2d(x1, y1, diam)

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
u_out_di = direct_integration(u_in, wv, d1, dz, x=x_asm[0], y=[0])


""" Plot """
# plot y2 = 0 cross-section
idx = y2[:, 0] == 0
plt.figure()
plt.plot(x2[0], np.abs(u_out_fraun[:, idx]), label="fraunhofer (theoretical)")
plt.plot(x2[0], np.abs(u_out_fres_one_step[idx][0]), label="fresnel (one step)")
plt.plot(x2[0], np.abs(u_out_fres_two_step[idx][0]), label="fresnel (two step)")
plt.plot(x2[0], np.abs(u_out_fres_conv[idx][0]), label="fresnel (conv)")
plt.plot(x2[0], np.abs(u_out_th_fres[idx][0]), label="fresnel (theoretical)")
plt.plot(x_asm[0], np.abs(u_out_di[0]), label="direct integration")

plt.xlabel("x[m]")
plt.title("log amplitude, y2 = 0")
plt.legend()
if log_scale:
    plt.yscale("log")
if xlim is not None:
    xlim = min(xlim, np.max(x_asm))
else:
    xlim = np.max(x_asm)
plt.xlim([-xlim, xlim])

# plot y2 = 0 cross-section
idx = y2[:, 0] == 0
plt.figure()
plt.plot(x2[0], np.abs(u_out_th_fres[idx][0]), label="fresnel (theoretical)")
plt.plot(x_asm[0], np.abs(u_out_asm[idx][0]), label="angular spectrum (numerical)")
plt.plot(x_asm[0], np.abs(u_out_asm_bl[idx][0]), label="angular spectrum (numerical, BL)")
plt.plot(x_asm[0], np.abs(u_out_di[0]), label="direct integration")

plt.xlabel("x[m]")
plt.title("log amplitude, y2 = 0")
plt.legend()
if log_scale:
    plt.yscale("log")
if xlim is not None:
    xlim = min(xlim, np.max(x_asm))
else:
    xlim = np.max(x_asm)
plt.xlim([-xlim, xlim])

# plot input
plot2d(x1.squeeze(), y1.squeeze(), u_in, title="Aperture")

# plot output
xlim = np.max(x_asm)
ylim = np.max(y_asm)
ax = plot2d(x2.squeeze(), y2.squeeze(), np.abs(u_out_fraun), title="Fraunhofer (theoretical)")
ax.set_xlim([-xlim, xlim])
ax.set_ylim([-ylim, ylim])
ax = plot2d(x2.squeeze(), y2.squeeze(), np.abs(u_out_th_fres), title="Fresnel (theoretical)")
ax.set_xlim([-xlim, xlim])
ax.set_ylim([-ylim, ylim])
plot2d(x_asm.squeeze(), y_asm.squeeze(), np.abs(u_out_asm), title="Angular spectrum")
plot2d(
    x_asm.squeeze(), y_asm.squeeze(), np.abs(u_out_asm_bl), title="Band-limited angular spectrum"
)

plt.show()
