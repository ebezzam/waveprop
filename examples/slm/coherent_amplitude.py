"""

Amplitude modulation with coherent light source (single wavelength).

# TODO script to compare different deadspace approaches, and non-deadspace

# (old) TODO give subpixel unique wavelength, see `examples/incoherent_source_adafruit_slm.py`



"""

import hydra
import os
import numpy as np
import time
import progressbar
from waveprop.util import sample_points, plot2d, rect2d
from waveprop.rs import angular_spectrum
from waveprop.color import ColorSystem
from waveprop.slm import get_centers, get_active_pixel_dim
import matplotlib.pyplot as plt
from waveprop.devices import SLMOptions, slm_dict, SLMParam, SensorOptions, sensor_dict


@hydra.main(version_base=None, config_path="../configs", config_name="slm_simulation")
def slm(config):

    # device configurations
    slm_config = slm_dict[config.slm]
    slm_pixel_dim = slm_config[SLMParam.CELL_SIZE]
    sensor_config = sensor_dict[config.sensor]

    N = config.sim.N  # number of grid points per side
    deadspace = config.sim.deadspace  # whether to account for deadspace between SLM pixels
    deadspace_fft = False
    pyffs = config.sim.pyffs  # won't be used if interpolation is not needed
    dz = config.sim.dz  # propagation distance
    sensor_crop_fraction = config.sim.sensor_crop_fraction  # fraction of sensor to use
    n_wavelength = config.sim.n_wavelength  # number of wavelengths to simulate

    gamma = config.plot.gamma
    gain = config.sim.gain
    plot_int = False

    # polychromatic simulation
    cs = ColorSystem(n_wavelength=n_wavelength)

    """ determining overlapping region and number of SLM pixels """
    overlapping_mask_size, overlapping_mask_dim, n_active_slm_pixels = get_active_pixel_dim(
        sensor_config=sensor_config,
        sensor_crop=sensor_crop_fraction,
        slm_config=slm_config,
    )

    """ generate random mask """
    from waveprop.slm import get_slm_mask

    if deadspace:
        d1 = np.array(overlapping_mask_size) / N
        x1, y1 = sample_points(N=N, delta=d1)
        slm_vals = np.random.rand(*n_active_slm_pixels)
    else:
        # extra ones will get blacked out... but not intuitive
        slm_vals = np.random.rand(*overlapping_mask_dim)
        x1, y1 = sample_points(N=overlapping_mask_dim, delta=slm_pixel_dim)
    target_dim = np.array([N, N])
    mask = get_slm_mask(
        slm_vals=slm_vals,
        slm_config=slm_config,
        sensor_config=sensor_config,
        crop_fact=sensor_crop_fraction,
        requires_grad=False,
        target_dim=target_dim,
        deadspace=deadspace,
    )
    u_in = np.sum(mask, axis=0)

    print(u_in.shape)

    # mask = np.random.rand(*n_active_slm_pixels)
    # centers = get_centers(n_active_slm_pixels, pixel_pitch=slm_config[SLMParam.PITCH])
    # mask = np.random.rand(*n_active_slm_pixels)
    # mask_flat = mask.reshape(-1)

    # """ discretize aperture (some SLM pixels will overlap due to coarse sampling) """
    # if deadspace:
    #     d1 = np.array(overlapping_mask_size) / N
    #     x1, y1 = sample_points(N=N, delta=d1)
    #     u_in = np.zeros((len(y1), x1.shape[1]))
    #     for i, _center in enumerate(centers):
    #         u_in += rect2d(x1, y1, slm_pixel_dim, offset=_center) * mask_flat[i]
    # else:
    #     u_in = np.zeros(overlapping_mask_dim)
    #     random_mask = np.random.rand(*n_active_slm_pixels)
    #     u_in[: n_active_slm_pixels[0], : n_active_slm_pixels[1]] = random_mask
    #     shift = ((np.array(overlapping_mask_dim) - np.array(n_active_slm_pixels)) / 2).astype(int)
    #     u_in = np.roll(u_in, shift=shift, axis=(0, 1))
    #     np.testing.assert_equal(u_in.shape, overlapping_mask_dim)
    #     x1, y1 = sample_points(N=overlapping_mask_dim, delta=slm_pixel_dim)
    # print(u_in.shape)

    # raise ValueError

    # plot input
    plot2d(x1.squeeze(), y1.squeeze(), u_in, title="Aperture")

    plt.savefig("1_input.png", dpi=config.plot.dpi)

    """ simulate """
    u_out = np.zeros((u_in.shape[0], u_in.shape[1], len(cs.wv)), dtype=np.float32)
    bar = progressbar.ProgressBar()
    start_time = time.time()
    for i in bar(range(cs.n_wavelength)):

        if deadspace_fft:
            # shift same aperture in the frequency domain
            u_in_cent = rect2d(x1, y1, slm_pixel_dim)
            centers = get_centers(n_active_slm_pixels, pixel_pitch=slm_config[SLMParam.PITCH])
            mask_flat = slm_vals.reshape(-1)
            u_out_wv, x2, y2 = angular_spectrum(
                u_in=u_in_cent * gain,
                wv=cs.wv[i],
                d1=d1,
                dz=dz,
                in_shift=centers,
                weights=mask_flat,
                pyffs=pyffs,
            )
        elif deadspace:
            u_out_wv, x2, y2 = angular_spectrum(
                u_in=u_in * gain, wv=cs.wv[i], d1=d1, dz=dz, pyffs=pyffs
            )
        else:
            u_out_wv, x2, y2 = angular_spectrum(
                u_in=u_in * gain, wv=cs.wv[i], d1=slm_pixel_dim, dz=dz, pyffs=pyffs
            )

        if plot_int:
            res = np.real(u_out_wv * np.conjugate(u_out_wv))
        else:
            res = np.abs(u_out_wv)
        u_out[:, :, i] = res

    # convert to RGB
    rgb = cs.to_rgb(u_out, clip=True, gamma=gamma)

    print(f"Computation time: {time.time() - start_time}")

    print("\n--out info")
    print("SHAPE : ", rgb.shape)
    print("DTYPE : ", rgb.dtype)
    print("MINIMUM : ", rgb.min())
    print("MAXIMUM : ", rgb.max())

    if deadspace:
        plot2d(x2, y2, rgb, title="BLAS {} m, superimposed Fourier".format(dz))
    else:
        plot2d(x2, y2, rgb, title="BLAS {} m".format(dz))

    plt.savefig("2_output.png", dpi=config.plot.dpi)

    print(f"\nSaved figures to {os.getcwd()}")


if __name__ == "__main__":
    slm()
