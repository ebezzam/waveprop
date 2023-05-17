"""

Amplitude modulation with coherent light source (single wavelength).

# TODO script to compare different deadspace approaches, and non-deadspace

# (old) TODO give subpixel unique wavelength, see `examples/incoherent_source_adafruit_slm.py`



TODO:
- first do without deadspace
- then do with deadspace (FFT and shift)

"""

import hydra
import os
import numpy as np
import time
import progressbar
from waveprop.util import sample_points, plot2d, rect2d, plot_field
from waveprop.rs import angular_spectrum
from waveprop.color import ColorSystem
from waveprop.slm import get_centers, get_active_pixel_dim
import matplotlib.pyplot as plt
import matplotlib
from waveprop.devices import SLMOptions, slm_dict, SLMParam, SensorOptions, sensor_dict


@hydra.main(version_base=None, config_path="../configs", config_name="slm_simulation")
def slm(config):

    matplotlib.rc("font", **config.plot.font)

    # device configurations
    slm_config = slm_dict[config.slm]
    wv = 532e-9
    dz = 4e-3

    # random values for phase SLM
    shape = slm_config[SLMParam.SHAPE]
    slm_vals = np.random.rand(*shape) * 2 * np.pi
    phase_mask = np.exp(1j * slm_vals)

    # plot phase mask
    # TODO use SLM dimensions for plotting
    fig, _ = plot_field(phase_mask, title="SLM function")
    fig.savefig("1_slm_input.png", dpi=config.plot.dpi)
    
    # print range
    print(f"SLM values range: {slm_vals.min()} to {slm_vals.max()}")

    # propagate without deadspace
    print("Propagating without deadspace...")
    d1 = slm_config[SLMParam.CELL_SIZE]
    u_out, x2, y2 = angular_spectrum(
        u_in=phase_mask, wv=wv, d1=d1, dz=dz, 
    )
    # print(u_out.shape)
    # print(u_out.dtype)

    # plot output mask
    # TODO use x2, y2
    fig, _ = plot_field(u_out, title="SLM propagated {}".format(dz))
    fig.savefig(f"2_slm_propagated_{dz}.png", dpi=config.plot.dpi)


    print(f"\nSaved figures to {os.getcwd()}")


if __name__ == "__main__":
    slm()
