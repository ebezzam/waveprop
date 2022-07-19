from waveprop.slm import get_active_pixel_dim, get_slm_mask, get_slm_mask_separable
from waveprop.devices import SLMOptions, slm_dict, SensorOptions, SensorParam, sensor_dict
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
from waveprop.util import sample_points, plot2d


n_trials = 50
deadspace = True  # False is failing
crop_fact = 0.8
down = 6
pytorch = True  # check False
device = "cuda"
shift = 1


if pytorch:
    dtype = torch.float32
else:
    dtype = np.float32


# SLM parameters (Adafruit screen)
slm_config = slm_dict[SLMOptions.ADAFRUIT.value]

# RPi HQ camera datasheet: https://www.arducam.com/sony/imx477/#imx477-datasheet
sensor_config = sensor_dict[SensorOptions.RPI_HQ.value]
target_dim = sensor_config[SensorParam.SHAPE] // down


overlapping_mask_size, overlapping_mask_dim, n_active_slm_pixels = get_active_pixel_dim(
    sensor_config=sensor_config,
    sensor_crop=crop_fact,
    slm_config=slm_config,
)

if deadspace:
    slm_vals = [
        np.random.rand(n_active_slm_pixels[0], 1).astype(np.float32),
        np.random.rand(1, n_active_slm_pixels[1]).astype(np.float32),
    ]
    slm_vals_full = np.random.rand(*n_active_slm_pixels).astype(np.float32)
else:
    slm_vals = [
        np.random.rand(overlapping_mask_dim[0], 1).astype(np.float32),
        np.random.rand(1, overlapping_mask_dim[1]).astype(np.float32),
    ]
    slm_vals_full = np.random.rand(*overlapping_mask_dim).astype(np.float32)

d1 = np.array(overlapping_mask_size) / target_dim
x1, y1 = sample_points(N=target_dim, delta=d1)


""" NON-SEPARABLE """
mask = get_slm_mask(
    slm_config=slm_config,
    sensor_config=sensor_config,
    crop_fact=crop_fact,
    target_dim=target_dim,
    slm_vals=slm_vals_full,
    deadspace=deadspace,
    pytorch=pytorch,
    device=device,
    dtype=dtype,
    shift=shift,
)
if torch.is_tensor(mask):
    plot2d(x1.squeeze(), y1.squeeze(), mask.detach().cpu(), title="Full slm")
else:
    plot2d(x1.squeeze(), y1.squeeze(), mask, title="Full slm")

_slm_vals = slm_vals[0] @ slm_vals[1]
start_time = time.perf_counter()
for _ in range(n_trials):
    mask = get_slm_mask(
        slm_config=slm_config,
        sensor_config=sensor_config,
        crop_fact=crop_fact,
        target_dim=target_dim,
        slm_vals=_slm_vals,
        deadspace=deadspace,
        pytorch=pytorch,
        device=device,
        dtype=dtype,
        shift=shift,
    )
proc_time = time.perf_counter() - start_time
print(f"Non separable : {proc_time / n_trials * 1000} ms")

if torch.is_tensor(mask):
    plot2d(x1.squeeze(), y1.squeeze(), mask.detach().cpu(), title="Non Separable")
else:
    plot2d(x1.squeeze(), y1.squeeze(), mask, title="Non Separable")

""" SEPARABLE """
start_time = time.perf_counter()
for _ in range(n_trials):
    mask_sep = get_slm_mask_separable(
        slm_config=slm_config,
        sensor_config=sensor_config,
        crop_fact=crop_fact,
        target_dim=target_dim,
        slm_vals=slm_vals,
        deadspace=deadspace,
        pytorch=pytorch,
        device=device,
        dtype=dtype,
        shift=shift,
    )
proc_time = time.perf_counter() - start_time
print(f"Separable : {proc_time / n_trials * 1000} ms")

mask_sep = mask_sep[0] @ mask_sep[1]
if pytorch:
    assert torch.equal(mask, mask_sep)
else:
    try:
        np.testing.assert_array_equal(mask, mask_sep)
    except:
        np.testing.assert_almost_equal(mask, mask_sep)

if torch.is_tensor(mask_sep):
    plot2d(x1.squeeze(), y1.squeeze(), mask_sep.detach().cpu(), title="Separable")
else:
    plot2d(x1.squeeze(), y1.squeeze(), mask_sep, title="Separable")


""" TEST FILE PATH """

mask = get_slm_mask(
    slm_config=slm_config,
    sensor_config=sensor_config,
    crop_fact=crop_fact,
    target_dim=target_dim,
    slm_vals="data/adafruit_pattern_20200802.npy",
    deadspace=deadspace,
    pytorch=pytorch,
    device=device,
    dtype=dtype,
)

if torch.is_tensor(mask):
    plot2d(x1.squeeze(), y1.squeeze(), mask.detach().cpu(), title="From file")
else:
    plot2d(x1.squeeze(), y1.squeeze(), mask, title="From file")

plt.show()
