"""

Compare three approaches:
- Sitzmann paper (spherical propagation, multiplication with mask, Fresnel)
- BLAS (spherical propagation, multiplication with mask, BLAS)
- Fully Fresnel (Fresnel, multiplication with mask, Fresnel)

"""

# TODO : loop pytorch, downsampling, and deadspace

from waveprop.util import plot2d
from waveprop.slm import get_slm_mask, get_active_pixel_dim
from waveprop.devices import SLMOptions, slm, SensorOptions, SensorParam, sensor
import numpy as np
from waveprop.color import ColorSystem
import torch
from waveprop.spherical import spherical_prop
import time
from waveprop.rs import angular_spectrum
from waveprop.fresnel import fresnel_conv, fresnel_one_step
import matplotlib.pyplot as plt


slm_pattern_fp = "data/slm_pattern_20200802.npy"
crop_fact = 0.7
device = "cuda"
scene2mask = 40e-2
mask2sensor = 4e-3
deadspace = True
n_trials = 10

# SLM  (Adafruit screen)
slm_config = slm[SLMOptions.ADAFRUIT.value]

# RPi HQ camera datasheet: https://www.arducam.com/sony/imx477/#imx477-datasheet
sensor_config = sensor[SensorOptions.RPI_HQ.value]

# polychromatric
cs = ColorSystem.rgb()

# determining overlapping region and number of SLM pixels
overlapping_mask_size, overlapping_mask_dim, n_active_slm_pixels = get_active_pixel_dim(
    sensor_config=sensor_config,
    sensor_crop=crop_fact,
    slm_config=slm_config,
)

# for pytorch in [True, False]:
for pytorch in [True, False]:
    print("\nPYTORCH : ", pytorch)
    if pytorch:
        dtype = torch.float32
        ctype = torch.complex64
    else:
        dtype = np.float32
        ctype = np.complex64

    for down in [16, 8]:
        print("\nDOWNSAMPLE : ", down)
        target_dim = sensor_config[SensorParam.SHAPE] // down
        print("shape :", target_dim)
        d1 = np.array(overlapping_mask_size) / target_dim

        # create mask in spatial domain
        mask = get_slm_mask(
            slm_config=slm_config,
            sensor_config=sensor_config,
            crop_fact=crop_fact,
            target_dim=target_dim,
            slm_pattern=slm_pattern_fp,
            deadspace=deadspace,
            pytorch=pytorch,
            device=device,
        )

        # spherical wavefront from scene to mask (same for Sitzmann and BLAS)
        spherical_wavefront = spherical_prop(
            in_shape=target_dim,
            d1=d1,
            wv=cs.wv,
            dz=scene2mask,
            return_psf=True,
            dtype=dtype,
            device=device,
            is_torch=pytorch,
        )

        # after mask (multiplication)
        u_in = mask * spherical_wavefront

        """ Sitzmann (Fresnel) """
        if pytorch:
            psfs = torch.zeros(u_in.shape, dtype=ctype).to(device)
        else:
            psfs = np.zeros(u_in.shape, dtype=ctype)
        start_time = time.time()
        for _ in range(n_trials):
            for i in range(cs.n_wavelength):
                # assert d1[0] == d1[1], d1   # TODO: figure out?
                psf_wv, x2, y2 = fresnel_conv(
                    u_in=u_in[i], wv=cs.wv[i], d1=d1[0], dz=mask2sensor, dtype=dtype, device=device
                )
                psfs[i] = psf_wv
        proc_time_sitzmann = (time.time() - start_time) / n_trials
        print(f"Computation time (Sitzmann): {proc_time_sitzmann} s")

        if pytorch:
            sitzmann = torch.square(torch.abs(psfs))
            sitzmann = sitzmann.cpu().detach().numpy()
        else:
            sitzmann = np.abs(psfs) ** 2
        sitzmann /= sitzmann.max()

        """ BLAS """
        if pytorch:
            psfs = torch.zeros(u_in.shape, dtype=ctype).to(device)
        else:
            psfs = np.zeros(u_in.shape, dtype=ctype)
        start_time = time.time()
        for _ in range(n_trials):
            for i in range(cs.n_wavelength):
                psf_wv, x2, y2 = angular_spectrum(
                    u_in=u_in[i], wv=cs.wv[i], d1=d1, dz=mask2sensor, dtype=dtype, device=device
                )
                psfs[i] = psf_wv
        proc_time_blas = (time.time() - start_time) / n_trials
        print(f"Computation time (BLAS): {proc_time_blas} s")

        if pytorch:
            blas = torch.square(torch.abs(psfs))
            blas = blas.cpu().detach().numpy()
        else:
            blas = np.abs(psfs) ** 2
        blas /= blas.max()

        """ full fresnel """
        from waveprop.util import sample_points

        x1, y1 = sample_points(N=target_dim, delta=d1)
        k = (2 * np.pi / cs.wv)[:, np.newaxis, np.newaxis]
        curvature = (x1 ** 2 + y1 ** 2)[np.newaxis, :]
        # eq. 4-7 of Goodman
        psf = (
            np.exp(1j * k * curvature / 2 / scene2mask)
            * np.exp(1j * k * scene2mask)
            / (1j * scene2mask * cs.wv)[:, np.newaxis, np.newaxis]
        )
        if pytorch:
            psf = torch.tensor(psf, dtype=ctype, device=device)

        # after mask (multiplication)
        u_in = mask * psf

        # Fresnel prop to sensor
        if pytorch:
            psfs = torch.zeros(u_in.shape, dtype=ctype).to(device)
        else:
            psfs = np.zeros(u_in.shape, dtype=ctype)
        start_time = time.time()
        for _ in range(n_trials):
            for i in range(cs.n_wavelength):
                # TODO : could use single step Fresnel?? given we know relation between input and output sampling
                psf_wv, x2, y2 = fresnel_conv(
                    u_in=u_in[i], wv=cs.wv[i], d1=d1[0], dz=mask2sensor, dtype=dtype, device=device
                )
                psfs[i] = psf_wv
        proc_time_blas = (time.time() - start_time) / n_trials
        print(f"Computation time (fresnel): {proc_time_blas} s")

        # intensity PSF
        if pytorch:
            fresnel = torch.square(torch.abs(psfs))
            fresnel = fresnel.cpu().detach().numpy()
        else:
            fresnel = np.abs(psfs) ** 2
        fresnel /= fresnel.max()

        # """ single step Fresnel """
        # if not pytorch:
        #     from waveprop.util import sample_points
        #
        #     d2 = d1
        #     Ny, Nx = target_dim
        #     if pytorch:
        #         psfs = torch.zeros(u_in.shape, dtype=ctype).to(device)
        #     else:
        #         psfs = np.zeros(u_in.shape, dtype=ctype)
        #     for i in range(cs.n_wavelength):
        #         d1 = np.array(
        #             [1 / Ny / d2[0] * cs.wv[i] * mask2sensor, 1 / Nx / d2[1] * cs.wv[i] * mask2sensor]
        #         )
        #
        #         # TODO : input field is too small, need to expand! much too large...
        #         print(d1 * target_dim, overlapping_mask_size)
        #         input_target_dim = overlapping_mask_size / d1
        #         # TODO : much too large...
        #         print("Input shape needed for desired output size", input_target_dim)
        #
        #         # fresnel between scene and mask
        #         x1, y1 = sample_points(N=target_dim, delta=d1)
        #         k = 2 * np.pi / cs.wv[i]
        #         curvature = x1 ** 2 + y1 ** 2
        #         # eq. 4-7 of Goodman
        #         psf = (
        #             np.exp(1j * k * curvature / 2 / scene2mask)
        #             * np.exp(1j * k * scene2mask)
        #             / (1j * scene2mask * cs.wv[i])
        #         )
        #         if pytorch:
        #             psf = torch.tensor(psf, dtype=ctype, device=device)
        #
        #         # mask multiplication
        #         _u_in = mask[i] * psf
        #
        #         # fresnel between mask and sensor
        #         psf_wv, x2, y2 = fresnel_one_step(u_in=_u_in, wv=cs.wv[i], d1=d1, dz=mask2sensor)
        #         psfs[i] = psf_wv
        #
        #     # intensity PSF
        #     if pytorch:
        #         fresnel = torch.square(torch.abs(psfs))
        #         fresnel = fresnel.cpu().detach().numpy()
        #     else:
        #         fresnel = np.abs(psfs) ** 2
        #     fresnel /= fresnel.max()
        #
        # else:
        #     fresnel = np.zeros_like(sitzmann)

        """ comparison / plot """
        fig, ax = plt.subplots(
            ncols=3,
            nrows=2,
            figsize=(12, 8),
        )
        fig.suptitle(f"pytorch={pytorch}, downsample={down}")
        plot2d(x2.squeeze(), y2.squeeze(), sitzmann, title="sitzmann", ax=ax[0][0])
        plot2d(x2.squeeze(), y2.squeeze(), blas, title="blas", ax=ax[0][1])

        # error
        err = np.linalg.norm(sitzmann - blas) / sitzmann.size
        print("error (sitzmann) :", err)
        plot2d(
            x2.squeeze(),
            y2.squeeze(),
            np.abs(sitzmann - blas),
            title=f"error = {err}",
            ax=ax[0][2],
        )

        plot2d(x2.squeeze(), y2.squeeze(), fresnel, title="fresnel", ax=ax[1][0])
        plot2d(x2.squeeze(), y2.squeeze(), blas, title="blas", ax=ax[1][1])

        # error
        err = np.linalg.norm(fresnel - blas) / fresnel.size
        print("error (fresnel) :", err)
        plot2d(
            x2.squeeze(),
            y2.squeeze(),
            np.abs(fresnel - blas),
            title=f"error = {err}",
            ax=ax[1][2],
        )

        del sitzmann
        del blas
        del fresnel
        del u_in
        del mask
        del spherical_wavefront

plt.show()
