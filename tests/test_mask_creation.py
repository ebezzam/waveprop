import torch
import numpy as np
from waveprop.devices import SLMOptions, slm_dict, SensorOptions, SensorParam, sensor_dict
from waveprop.slm import get_active_pixel_dim, get_slm_mask, get_slm_mask_separable
import os


def test_separable():

    down = 6
    crop_fact = 0.8

    # SLM parameters (Adafruit screen)
    slm_config = slm_dict[SLMOptions.ADAFRUIT.value]

    # RPi HQ camera datasheet: https://www.arducam.com/sony/imx477/#imx477-datasheet
    sensor_config = sensor_dict[SensorOptions.RPI_HQ.value]
    target_dim = sensor_config[SensorParam.SHAPE] // down

    # determine number of overlapping pixels
    _, overlapping_mask_dim, n_active_slm_pixels = get_active_pixel_dim(
        sensor_config=sensor_config,
        sensor_crop=crop_fact,
        slm_config=slm_config,
    )

    for deadspace in [False, True]:

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

        for pytorch in [False, True]:

            device_vals = ["cpu"]
            if pytorch:
                dtype = torch.float32
                if torch.cuda.is_available():
                    device_vals = ["cuda", "cpu"]
            else:
                dtype = np.float32

            for device in device_vals:

                for shift in [0, 1, 2]:

                    print(
                        "deadspace",
                        deadspace,
                        ", pytorch",
                        pytorch,
                        ", device",
                        device,
                        ", shift",
                        shift,
                    )

                    # not separable
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
                        shift=shift,
                    )

                    # separable
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
                    mask_sep = mask_sep[0] @ mask_sep[1]
                    if pytorch:
                        assert torch.equal(mask, mask_sep)
                    else:
                        try:
                            np.testing.assert_array_equal(mask, mask_sep)
                        except:
                            np.testing.assert_almost_equal(mask, mask_sep)


def test_file_input(fp=None):

    if fp is None:
        dirname = os.path.dirname(__file__)
        fp = os.path.join(dirname, "../data/adafruit_pattern_20200802.npy")

    down = 6
    crop_fact = 0.8

    # SLM parameters (Adafruit screen)
    slm_config = slm_dict[SLMOptions.ADAFRUIT.value]

    # RPi HQ camera datasheet: https://www.arducam.com/sony/imx477/#imx477-datasheet
    sensor_config = sensor_dict[SensorOptions.RPI_HQ.value]
    target_dim = sensor_config[SensorParam.SHAPE] // down

    for deadspace in [False, True]:

        for pytorch in [False, True]:

            device_vals = ["cpu"]
            if pytorch:

                dtype = torch.float32
                if torch.cuda.is_available():
                    device_vals = ["cuda", "cpu"]
            else:
                dtype = np.float32

            for device in device_vals:

                for shift in [0, 1, 2]:

                    for pattern_shift in [None, [2, 3]]:

                        print(
                            "deadspace",
                            deadspace,
                            ", pytorch",
                            pytorch,
                            ", device",
                            device,
                            ", shift",
                            shift,
                            ", pattern_shift",
                            pattern_shift,
                        )

                        get_slm_mask(
                            slm_config=slm_config,
                            sensor_config=sensor_config,
                            crop_fact=crop_fact,
                            target_dim=target_dim,
                            slm_vals=fp,
                            deadspace=deadspace,
                            pytorch=pytorch,
                            device=device,
                            dtype=dtype,
                            shift=shift,
                            pattern_shift=pattern_shift,
                        )


if __name__ == "__main__":

    print("\n----- Test separable -----")
    test_separable()

    print("\n----- Test file input -----")
    test_file_input()
