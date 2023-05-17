import matplotlib.pyplot as plt
import numpy as np
from waveprop.devices import SLMOptions, slm_dict, SLMParam, SensorOptions, sensor_dict
from waveprop.slm import get_centers


slm = "holoeye_lc2012"
oversampling = 2
plot_percent = 1 / 100

if plot_percent is None:
    plot_percent = 1
side_percent = np.sqrt(plot_percent)  # percent on each side of the SLM to plot

# device configurations
slm_config = slm_dict[slm]


print("Resolution [px] : {}".format(slm_config[SLMParam.SHAPE]))
print("Deadspace [m] : {}".format(slm_config[SLMParam.DEADSPACE]))
print("Cell size [m] : {}".format(slm_config[SLMParam.CELL_SIZE]))
print("Pitch [m] : {}".format(slm_config[SLMParam.PITCH]))
print("Fill factor [%] : {}".format(slm_config[SLMParam.FILL_FACTOR]))

# loop over all slm pixels
centers = get_centers(
    slm_dim=(slm_config[SLMParam.SHAPE] * side_percent).astype(int),
    pixel_pitch=slm_config[SLMParam.PITCH],
)
n_pixels = centers.shape[0]
print("Number of pixels plotted : {}".format(n_pixels))
print("Total number of pixels : {}".format(np.prod(slm_config[SLMParam.SHAPE])))
print("Percentage of pixels [%] : {}".format(100 * n_pixels / np.prod(slm_config[SLMParam.SHAPE])))

# raise ValueError

sim_size = slm_config[SLMParam.SHAPE] * oversampling
print("Simulation size [px] : {}".format(sim_size))

# discretization
d1 = np.array(slm_config[SLMParam.SIZE] * side_percent) / sim_size
_height_pixel, _width_pixel = (slm_config[SLMParam.CELL_SIZE] / d1).astype(int)
print("Dimensions of cell in pixels : ", _height_pixel, _width_pixel)

# pitch in pixels
print("Pitch in pixels : ", (slm_config[SLMParam.PITCH] / d1).astype(int))

print("Discretization : {}".format(d1))
# raise ValueError

mask = np.zeros(sim_size)
for i, _center in enumerate(centers):

    _center_pixel = (_center / d1 + sim_size / 2).astype(int)
    _center_top_left_pixel = (
        _center_pixel[0] - np.floor(_height_pixel / 2).astype(int),
        _center_pixel[1] + 1 - np.floor(_width_pixel / 2).astype(int),
    )

    mask[
        _center_top_left_pixel[0] : _center_top_left_pixel[0] + _height_pixel,
        _center_top_left_pixel[1] : _center_top_left_pixel[1] + _width_pixel,
    ] = 1


# plot SLM
# TODO : fix ticks
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

extent = [
    -0.5 * slm_config[SLMParam.PITCH][1],
    (sim_size[1] - 0.5) * slm_config[SLMParam.PITCH][1],
    (sim_size[0] - 0.5) * slm_config[SLMParam.PITCH][0],
    -0.5 * slm_config[SLMParam.PITCH][0],
]  # [left, right, bottom, top]

ax.imshow(mask, cmap="gray", extent=extent)
ax.set_xlabel("x [um]")
ax.set_ylabel("y [um]")

# https://github.com/nbaehler/mask-designer/blob/b348204ff1bc200f570512d30b5528536e15788f/mask_designer/virtual_slm.py#L115
n_x_ticks = 100

# ticks every 10 pitch lengths

# x_ticks = np.arange(0, sim_size[1], 10 * _width_pixel)
# ax.set_xticks(x_ticks)
# x_tick_labels = np.arange(0, sim_size[1], 10 * _width_pixel) / _width_pixel
# ax.set_xticklabels(x_tick_labels)

# x_ticks = np.arange(-0.5, sim_size[1], n_x_ticks) * slm_config[SLMParam.PITCH][1]
# ax.set_xticks(x_ticks)
# x_tick_labels = (np.arange(-0.5, sim_size[1], n_x_ticks) + 0.5).astype(int)
# ax.set_xticklabels(x_tick_labels)

# import pudb; pudb.set_trace()

# # set x and y ticks according to discretization (20 ticks)
# xticks = np.arange(0, sim_size[1], sim_size[1] / 20) * d1[1] * 1e6
# yticks = np.arange(0, sim_size[0], sim_size[0] / 20) * d1[0] * 1e6

# ax.set_xticks(xticks)
# ax.set_yticks(yticks)

ax.set_title(f"SLM {slm} ({plot_percent*100}% of pixels)")

# save figure
fig.savefig("slm_{}.png".format(slm), dpi=300)
