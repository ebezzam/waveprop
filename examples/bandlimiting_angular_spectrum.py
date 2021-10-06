"""
Show benefits of bandlimiting angular spectrum.

"""


import numpy as np
import matplotlib.pyplot as plt
from waveprop.util import rect2d, sample_points, plot2d
from waveprop.rs import (
    angular_spectrum,
    direct_integration,
)
import matplotlib

# plotting params
font = {"family": "Times New Roman", "weight": "normal", "size": 14}
matplotlib.rc("font", **font)
xlim = None
log_scale = True

# simulation params
N = 512  # number of grid points per size
L = 1e-2  # total size of grid
diam = 3e-4  # diameter of aperture [m]
wv = 1e-6  # wavelength
dz = 1  # distance [m]
d1 = L / N  # source-plane grid spacing


print("\nPROPAGATION DISTANCE : {} m".format(dz))

""" discretize aperture """
x1, y1 = sample_points(N=N, delta=d1)
u_in = rect2d(x1, y1, diam)

""" Angular spectrum """
u_out_asm, x_asm, y_asm = angular_spectrum(u_in=u_in, wv=wv, d1=d1, dz=dz, bandlimit=False)
u_out_asm_bl, _, _ = angular_spectrum(u_in=u_in, wv=wv, d1=d1, dz=dz, bandlimit=True)

""" Direct integration (ground truth) """
u_out_di = direct_integration(u_in, wv, d1, dz, x=x_asm[0], y=[0])

# plot y2 = 0 cross-section
idx = y_asm[:, 0] == 0
plt.figure()
plt.plot(x_asm[0], np.abs(u_out_asm[idx][0]), label="AS", alpha=0.7)
plt.plot(x_asm[0], np.abs(u_out_asm_bl[idx][0]), label="BLAS", alpha=0.7)
plt.plot(x_asm[0], np.abs(u_out_di[0]), label="direct integration", alpha=0.7)

plt.xlabel("x[m]")
plt.legend()
if log_scale:
    plt.yscale("log")
    plt.title("log amplitude, y = 0")
else:
    plt.title("amplitude, y = 0")
if xlim is not None:
    xlim = min(xlim, np.max(x_asm))
else:
    xlim = np.max(x_asm)
plt.xlim([-xlim, xlim])

# plot input
ax = plot2d(x1.squeeze(), y1.squeeze(), u_in, title="Aperture")

# plot output
plot2d(x_asm.squeeze(), y_asm.squeeze(), np.abs(u_out_asm_bl), title="BLAS")
plot2d(x_asm.squeeze(), y_asm.squeeze(), np.abs(u_out_asm), title="AS")


plt.show()
