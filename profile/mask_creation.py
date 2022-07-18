from waveprop.slm import get_active_pixel_dim, get_slm_mask, get_slm_mask_separable
from waveprop.devices import SLMOptions, slm_dict, SensorOptions, SensorParam, sensor_dict
import numpy as np
import torch
import time


n_trials = 100
deadspace = True  # False is failing
crop_fact = 0.8
down = 6
pytorch = True
device = "cuda"


if pytorch:
    dtype = torch.float32
    ctype = torch.complex64
else:
    dtype = np.float32
    ctype = np.complex64


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
else:
    slm_vals = [
        np.random.rand(overlapping_mask_dim[0], 1).astype(np.float32),
        np.random.rand(1, overlapping_mask_dim[1]).astype(np.float32),
    ]


""" NON-SEPARABLE """
start_time = time.perf_counter()
for _ in range(n_trials):
    mask = get_slm_mask(
        slm_config=slm_config,
        sensor_config=sensor_config,
        crop_fact=crop_fact,
        target_dim=target_dim,
        slm_vals=slm_vals[0] @ slm_vals[1],
        deadspace=deadspace,
        pytorch=pytorch,
        device=device,
        dtype=dtype,
    )
print(mask.shape)
proc_time = time.perf_counter() - start_time
print(f"Non separable : {proc_time / n_trials * 1000} ms")

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
    )
proc_time = time.perf_counter() - start_time
print(f"Separable : {proc_time / n_trials * 1000} ms")


np.testing.assert_array_equal(mask, mask_sep[0] @ mask_sep[1])
