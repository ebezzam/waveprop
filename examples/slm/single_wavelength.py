"""

Propagating SLM pattern with coherent light source of single wavelength).

"""

import hydra
import os
import numpy as np
import time
import matplotlib
from waveprop.slm import SLM


@hydra.main(version_base=None, config_path="../configs", config_name="slm_prop")
def slm(config):

    np.random.seed(config.seed)
    matplotlib.rc("font", **config.plot.font)

    # simulation parameters
    wv = config.wv
    dz = config.dz
    deadspace = config.deadspace
    deadspace_fft = config.deadspace_fft
    percent = config.percent

    # only for deadspace
    oversampling = config.oversampling
    if not deadspace:
        deadspace_fft = False

    # load SLM
    slm = SLM.from_string(
        config.slm,
        percent=percent,
        model_deadspace=deadspace,
        oversampling=oversampling,
        deadspace_fft=deadspace_fft,
    )
    slm.print()

    # random mask values
    print("\nSection resolution [px] : ", slm.shape)
    if slm.phase:
        slm_vals = np.random.rand(*slm.shape) * 2 * np.pi
    else:
        slm_vals = np.random.rand(*slm.shape)

    # plot mask
    fig, _ = slm.plot_mask(vals=slm_vals)
    fig.savefig("1_slm_input.png", dpi=config.plot.dpi)

    # propagate
    start_time = time.time()
    fig, _ = slm.plot_propagation(vals=slm_vals, wv=wv, dz=dz)
    fig.savefig(f"2_slm_propagated_{dz}.png", dpi=config.plot.dpi)
    print("Processing time [s] : ", time.time() - start_time)

    print(f"\nSaved figures to {os.getcwd()}")


if __name__ == "__main__":
    slm()
