"""

Compare two approaches for deadspace modeling:
- Creating mask in spatial domain with imperfect pixels due to sampling.
- In frequency domain, for arbitrary shifts.

Use fixed mask pattern for reproducibility.

Just profiling the PSF computation!

"""

from waveprop.util import rect2d, zero_pad, ft2, sample_points, plot2d
from waveprop.slm import get_slm_mask, get_active_pixel_dim
from waveprop.devices import SLMOptions, slm_dict, SensorOptions, SensorParam, sensor_dict, SLMParam
import numpy as np
from waveprop.color import ColorSystem
import torch
import time
from waveprop.rs import angular_spectrum
import matplotlib.pyplot as plt


slm_pattern_fp = "data/adafruit_pattern_20200802.npy"
crop_fact = 0.7
device = "cuda"
mask2sensor = 4e-3

# SLM  (Adafruit screen)
slm_config = slm_dict[SLMOptions.ADAFRUIT.value]

# RPi HQ camera datasheet: https://www.arducam.com/sony/imx477/#imx477-datasheet
sensor_config = sensor_dict[SensorOptions.RPI_HQ.value]

# polychromatric
cs = ColorSystem.rgb()

# determining overlapping region and number of SLM pixels
overlapping_mask_size, overlapping_mask_dim, n_active_slm_pixels = get_active_pixel_dim(
    sensor_config=sensor_config,
    sensor_crop=crop_fact,
    slm_config=slm_config,
)

for pytorch in [True, False]:
    print("\nPYTORCH : ", pytorch)

    if pytorch:
        dtype = torch.float32
        ctype = torch.complex64
    else:
        dtype = np.float32
        ctype = np.complex64

    for down in [16, 8, 4]:
        print("\nDOWNSAMPLE : ", down)
        target_dim = sensor_config[SensorParam.SHAPE] // down
        print("shape :", target_dim)
        d1 = np.array(overlapping_mask_size) / target_dim

        try:
            """Creating mask in spatial domain"""
            mask = get_slm_mask(
                slm_config=slm_config,
                sensor_config=sensor_config,
                crop_fact=crop_fact,
                target_dim=target_dim,
                slm_vals=slm_pattern_fp,
                deadspace=True,
                pytorch=pytorch,
                device=device,
            )

            if pytorch:
                psfs = torch.zeros(mask.shape, dtype=ctype).to(device)
            else:
                psfs = np.zeros(mask.shape, dtype=ctype)
            # -- mask to sensor simulation
            start_time = time.time()
            for i in range(cs.n_wavelength):
                psf_wv, x2, y2 = angular_spectrum(
                    u_in=mask[i], wv=cs.wv[i], d1=d1, dz=mask2sensor, dtype=dtype, device=device
                )
                psfs[i] = psf_wv
            proc_time_spatial = time.time() - start_time
            print(f"Computation time (spatial): {proc_time_spatial} s")

            if pytorch:
                spatial = torch.square(torch.abs(psfs))
                spatial = spatial.cpu().detach().numpy()
            else:
                spatial = np.abs(psfs) ** 2
            spatial /= spatial.max()
        except:
            print("Not enough CUDA memory..")
            continue

        try:
            """Creating mask in spatial frequency domain"""
            mask_flat, centers = get_slm_mask(
                slm_config=slm_config,
                sensor_config=sensor_config,
                crop_fact=crop_fact,
                target_dim=target_dim,
                slm_vals=slm_pattern_fp,
                deadspace=True,
                pytorch=pytorch,
                device=device,
                return_slm_vals=True,
            )

            # -- mask to sensor simulation
            # precompute aperture FT
            x1, y1 = sample_points(N=target_dim, delta=d1)
            u_in_cent = rect2d(x1, y1, slm_config[SLMParam.CELL_SIZE])
            aperture_pad = zero_pad(u_in_cent)
            aperture_ft = ft2(aperture_pad, delta=d1).astype(np.complex64)
            if pytorch:
                aperture_ft = torch.tensor(aperture_ft, dtype=ctype).to(device)
                # input field is all ones
                u_in = torch.ones(target_dim.tolist(), dtype=ctype).to(device)
            else:
                u_in = np.ones(target_dim, dtype=ctype)
            start_time = time.time()
            for i in range(cs.n_wavelength):
                psf_wv, x2, y2 = angular_spectrum(
                    u_in=u_in,
                    wv=cs.wv[i],
                    d1=d1,
                    dz=mask2sensor,
                    aperture_ft=aperture_ft,
                    in_shift=centers,
                    weights=mask_flat[i],
                    dtype=dtype,
                    device=device,
                )
                psfs[i] = psf_wv
            proc_time_freq = time.time() - start_time
            print(f"Computation time (frequency): {proc_time_freq} s")
            print(f"{proc_time_freq/proc_time_spatial}x computation")

            """ comparison / plot """
            fig, ax = plt.subplots(
                ncols=3,
                nrows=1,
                figsize=(10, 5),
            )
            fig.suptitle(f"pytorch={pytorch}, downsample={down}")
            plot2d(x2.squeeze(), y2.squeeze(), spatial, title="In spatial domain", ax=ax[0])

            if pytorch:
                frequency = torch.square(torch.abs(psfs))
                frequency = frequency.cpu().detach().numpy()
            else:
                frequency = np.abs(psfs) ** 2
            frequency /= frequency.max()
            plot2d(x2.squeeze(), y2.squeeze(), frequency, title="In frequency domain", ax=ax[1])

            # error
            err = np.linalg.norm(spatial - frequency) / spatial.size
            print("error :", err)
            plot2d(
                x2.squeeze(),
                y2.squeeze(),
                np.abs(spatial - frequency),
                title=f"error = {err}",
                ax=ax[2],
            )

            del mask_flat
            del frequency
        except:
            print("Not enough CUDA memory..")

        torch.cuda.empty_cache()
        del mask
        del psfs
        del spatial
        del psf_wv

plt.show()
