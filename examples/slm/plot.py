"""
Useful to visualize deadspace of SLM.

TODO : change to using SLM class
"""

import matplotlib.pyplot as plt
import matplotlib
import os
import numpy as np
import hydra
from waveprop.devices import SLMOptions, slm_dict, SLMParam
from waveprop.slm import get_centers, get_color_filter, SLM
from waveprop.util import sample_points, plot2d


@hydra.main(version_base=None, config_path="../configs", config_name="slm_plot")
def plot(config):

    matplotlib.rc("font", **config.plot.font)

    # load SLM
    percent = config.percent
    oversampling = config.oversampling
    slm = SLM.from_string(
        config.slm, percent=percent, model_deadspace=True, oversampling=oversampling
    )
    slm.print()

    # all-ones
    if slm.phase:
        slm_vals = np.ones(slm.shape) * 0
    else:
        slm_vals = np.ones(slm.shape)

    # plot mask
    fig, _ = slm.plot_mask(vals=slm_vals)
    fig.savefig("slm_{}.png".format(config.slm), dpi=config.plot.dpi)

    print(f"\nSaved figures to {os.getcwd()}")

    raise ValueError

    slm = config.slm
    oversampling = config.oversampling
    if config.percent is None:
        plot_percent = 1
    else:
        assert 0 < config.percent <= 100, "Percent must be between 0 and 100"
        plot_percent = config.percent / 100
    side_percent = np.sqrt(plot_percent)  # percent on each side of the SLM to plot

    # load device configuration
    assert slm in SLMOptions.values(), "Choose an SLM from {}".format(SLMOptions.values())
    slm_config = slm_dict[slm]

    print("SLM Resolution [px] : {}".format(slm_config[SLMParam.SHAPE]))
    print("Deadspace [m] : {}".format(slm_config[SLMParam.DEADSPACE]))
    print("Cell size [m] : {}".format(slm_config[SLMParam.CELL_SIZE]))
    print("Pitch [m] : {}".format(slm_config[SLMParam.PITCH]))
    print("Fill factor [%] : {}".format(slm_config[SLMParam.FILL_FACTOR]))

    # loop over all slm pixels
    n_active_pixels = (slm_config[SLMParam.SHAPE] * side_percent).astype(int)
    centers = get_centers(
        slm_dim=n_active_pixels,
        pixel_pitch=slm_config[SLMParam.PITCH],
    )
    n_pixels = centers.shape[0]
    print("Number of pixels plotted : {}".format(n_pixels))
    print("Total number of pixels : {}".format(np.prod(slm_config[SLMParam.SHAPE])))
    print(
        "Percentage of pixels [%] : {}".format(100 * n_pixels / np.prod(slm_config[SLMParam.SHAPE]))
    )

    sim_size = (slm_config[SLMParam.SHAPE] * oversampling).astype(int)
    print("Simulation size [px] : {}".format(sim_size))

    # discretization
    d1 = np.array(slm_config[SLMParam.SIZE] * side_percent) / sim_size
    _height_pixel, _width_pixel = (slm_config[SLMParam.CELL_SIZE] / d1).astype(int)
    print("Dimensions of cell in pixels : ", _height_pixel, _width_pixel)

    # pitch in pixels
    print("Pitch in pixels : ", (slm_config[SLMParam.PITCH] / d1).astype(int))
    print("Discretization : {}".format(d1))

    # create mask with all 1s
    if SLMParam.COLOR_FILTER in slm_config.keys():
        n_color_filter = np.prod(slm_config["color_filter"].shape[:2])
        cf = get_color_filter(
            slm_dim=n_active_pixels,
            color_filter=slm_config[SLMParam.COLOR_FILTER],
            shift=0,
            flat=True,
        )
    else:
        n_color_filter = 1
        cf = None
    mask = np.zeros(np.insert(sim_size, 0, n_color_filter))

    for i, _center in enumerate(centers):

        _center_pixel = (_center / d1 + sim_size / 2).astype(int)
        _center_top_left_pixel = (
            _center_pixel[0] - np.floor(_height_pixel / 2).astype(int),
            _center_pixel[1] + 1 - np.floor(_width_pixel / 2).astype(int),
        )

        if cf is not None:
            _rect = np.tile(cf[i][:, np.newaxis, np.newaxis], (1, _height_pixel, _width_pixel))
        else:
            _rect = np.ones((1, _height_pixel, _width_pixel))

        mask[
            :,
            _center_top_left_pixel[0] : _center_top_left_pixel[0] + _height_pixel,
            _center_top_left_pixel[1] : _center_top_left_pixel[1] + _width_pixel,
        ] = _rect

    # plot SLM
    print("Simulated mask shape : ", mask.shape)
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    x1, y1 = sample_points(N=sim_size, delta=d1)
    plot2d(x1, y1, mask, ax=ax, colorbar=False)
    ax.set_title(f"SLM {slm} ({plot_percent*100}% of pixels)")

    # save figure
    fig.savefig("slm_{}.png".format(slm), dpi=config.plot.dpi)

    print(f"\nSaved figures to {os.getcwd()}")


if __name__ == "__main__":
    plot()
