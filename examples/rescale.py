"""
Rescale optical wave propagation.

"""


import numpy as np
import matplotlib.pyplot as plt
import time
from waveprop.util import rect2d, sample_points, plot2d
from waveprop.fresnel import fresnel_one_step, shifted_fresnel
from waveprop.fraunhofer import fraunhofer, fraunhofer_prop_circ_ap
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
diam = 1e-3  # diameter of aperture [m]
wv = 1e-6  # wavelength
dz = 1  # distance [m]
d1 = L / N  # source-plane grid spacing

output_scaling = 1 / 3
N_out = None

""" discretize aperture """
x1, y1 = sample_points(N=N, delta=d1)
u_in = rect2d(x1, y1, diam)
plot2d(x1.squeeze(), y1.squeeze(), u_in, title="Aperture")

""" Shifted Fresnel """
u_out_sfres, x2_sfres, y2_sfres = shifted_fresnel(u_in, wv, d1, dz, d2=output_scaling * d1)
print("-- ", end="")
fresnel_goodman(wv, dz, x1, y1, x2_sfres, y2_sfres)
print("-- ", end="")
fresnel_saleh(wv, dz, x2_sfres, y2_sfres)


""" Scaled BLAS """
u_out_asm_scaled, x_asm_scaled, y_asm_scaled = angular_spectrum(
    u_in=u_in, wv=wv, d1=d1, dz=dz, bandlimit=True, N_out=N_out, d2=output_scaling * d1
)
u_out_asm, x_asm, y_asm = angular_spectrum(u_in=u_in, wv=wv, d1=d1, dz=dz, bandlimit=True)

# plot output
plot2d(x2_sfres.squeeze(), y2_sfres.squeeze(), np.abs(u_out_sfres), title="Shifted Fresnel")
plot2d(
    x_asm_scaled.squeeze(), y_asm_scaled.squeeze(), np.abs(u_out_asm_scaled), title="Scaled BLAS"
)
plot2d(x_asm.squeeze(), y_asm.squeeze(), np.abs(u_out_asm), title="BLAS")


plt.show()
