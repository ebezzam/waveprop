"""
Rectangular tiling as described in "Shifted Fresnel diffraction for computational holography".

For higher resolution

TODO: output dimension of tiles is different from input dimension
TODO: larger output
TODO: sweep over distance

"""

import numpy as np
import matplotlib.pyplot as plt
from waveprop.util import rect2d, sample_points, plot2d, rect_tiling
from waveprop.fresnel import shifted_fresnel
import time
from waveprop.rs import angular_spectrum


# simulation parameters
N_in = 128  # number of grid points per size
L_in = 1e-2  # total size of grid, input and output
n_tiles = 3
N_out = n_tiles * N_in

diam = [1e-3, 3e-3]  # diameter of aperture [m]
d1 = L_in / N_in  # source-plane grid spacing
d2 = L_in / N_out
wv = 1e-6  # wavelength
dz = 1


""" discretize aperture """
x1, y1 = sample_points(N=N_in, delta=d1)
u_in = rect2d(x1, y1, diam)
plot2d(x1.squeeze(), y1.squeeze(), u_in, title="Aperture")


def prop_func(out_shift, pyffs=True):
    # return shifted_fresnel(u_in, wv, d1, dz, d2=d2, out_shift=out_shift)[0]
    # return angular_spectrum(
    #     u_in=u_in, wv=wv, d1=d1, dz=dz, d2=d2, bandlimit=True, out_shift=out_shift
    # )[0]
    return angular_spectrum(u_in=u_in, wv=wv, d1=d1, d2=d2, dz=dz, out_shift=out_shift, pyffs=True)[
        0
    ]


start_time = time.time()
u_out, x2, y2 = rect_tiling(N_in=N_in, N_out=N_out, L=L_in, n_tiles=n_tiles, prop_func=prop_func)
print(f"Proc time : {time.time() - start_time}")

# plot
plot2d(
    x2.squeeze(),
    y2.squeeze(),
    np.abs(u_out),
    title=f"Combined tiles",
)


plt.show()
