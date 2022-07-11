import numpy as np
import torch
from waveprop.util import crop, rect2d, sample_points
import torch.nn.functional as F
import cv2
from waveprop.devices import SLMParam, SensorParam


def get_slm_mask(
    slm_config,
    sensor_config,
    crop_fact,
    target_dim,
    slm_vals=None,
    slm_pattern=None,
    deadspace=True,
    pattern_shift=None,
    pytorch=False,
    device="cuda",
    dtype=None,
    first_color=0,
    return_slm_vals=False,
    requires_grad=True,
):
    """

    Parameters
    ----------
    slm_dim
    slm_size
    slm_pixel_dim
    rpi_dim
    rpi_pixel_dim
    crop_fact
    target_dim
    slm_pattern
    deadspace
    pattern_shift
    pytorch
    device
    dtype
    first_color

    Returns
    -------

    """
    if dtype is None:
        if pytorch:
            dtype = torch.float32
        else:
            dtype = np.float32

    overlapping_mask_size, overlapping_mask_dim, n_active_slm_pixels = get_active_pixel_dim(
        sensor_config=sensor_config,
        sensor_crop=crop_fact,
        slm_config=slm_config,
    )

    d1 = np.array(overlapping_mask_size) / target_dim
    x, y = sample_points(N=target_dim, delta=d1)

    # load mask pattern
    if slm_vals is not None:
        """use provided values"""
        if deadspace:
            assert np.array_equal(slm_vals.shape, n_active_slm_pixels)
        else:
            assert np.array_equal(slm_vals.shape, overlapping_mask_dim)
        if torch.is_tensor(slm_vals):
            pytorch = True
            device = slm_vals.device
        else:
            pytorch = False

    elif slm_pattern is not None:
        """load from file"""
        slm_pattern_values = np.load(slm_pattern)
        # stack RGB pixels along columns
        slm_pattern_values = slm_pattern_values.reshape((-1, 160), order="F")
        # crop section
        top_left = (
            int((slm_pattern_values.shape[0] - n_active_slm_pixels[0]) / 2),
            int((slm_pattern_values.shape[1] - n_active_slm_pixels[1]) / 2),
        )
        if pattern_shift:
            top_left = np.array(top_left) + np.array(pattern_shift)
        first_color = slm_config[SLMParam.COLOR_ORDER][
            top_left[0] % len(slm_config[SLMParam.COLOR_ORDER])
        ]
        if deadspace:
            slm_vals = crop(slm_pattern_values, shape=n_active_slm_pixels, topleft=top_left).astype(
                np.float32
            )
        else:
            slm_vals = crop(
                slm_pattern_values, shape=overlapping_mask_dim, topleft=top_left
            ).astype(np.float32)

    else:
        """randomly generate"""
        if deadspace:
            slm_vals = np.random.rand(*n_active_slm_pixels).astype(np.float32)
        else:
            slm_vals = np.random.rand(*overlapping_mask_dim).astype(np.float32)

    # create mask
    if deadspace:
        if torch.is_tensor(slm_vals):
            slm_vals_flat = slm_vals.flatten()
            mask = torch.zeros((3, len(y), x.shape[1]), dtype=dtype, device=device)
        else:
            slm_vals_flat = slm_vals.reshape(-1)
            mask = np.zeros((3, len(y), x.shape[1]), dtype=np.float32)
            if pytorch:
                slm_vals_flat = torch.tensor(
                    slm_vals_flat, dtype=dtype, device=device, requires_grad=requires_grad
                )
                mask = torch.tensor(mask, dtype=dtype, device=device)

        centers, cf = get_centers(
            n_active_slm_pixels,
            pixel_pitch=slm_config[SLMParam.PITCH],
            return_color_filter=True,
            first_color=first_color,
        )
        if return_slm_vals:
            if pytorch:
                cf = torch.tensor(cf).to(slm_vals_flat)
            return cf * slm_vals_flat, centers

        for i, _center in enumerate(centers):

            _center_pixel = (_center / d1 + target_dim/2).astype(int)
            _height_pixel, _width_pixel = (slm_config[SLMParam.CELL_SIZE] / d1).astype(int)

            rect =  np.tile(cf[:, i][:, np.newaxis, np.newaxis], (1, _height_pixel, _width_pixel))

            if pytorch:
                rect = torch.tensor(rect).to(slm_vals_flat)

            mask[:, _center_pixel[0] - np.floor(_height_pixel/2).astype(int) : _center_pixel[0] + np.ceil(_height_pixel/2).astype(int),
                 _center_pixel[1]+1 - np.floor(_width_pixel/2).astype(int) : _center_pixel[1]+1 + np.ceil(_width_pixel/2).astype(int)] = slm_vals_flat[i] * rect

    else:
        mask = np.zeros((3,) + tuple(overlapping_mask_dim), dtype=np.float32)
        for i in range(n_active_slm_pixels[0]):
            mask[
                (i + first_color) % 3, n_active_slm_pixels[0] - 1 - i, : n_active_slm_pixels[1]
            ] = 1
        shift = ((np.array(overlapping_mask_dim) - np.array(n_active_slm_pixels)) / 2).astype(int)
        mask = np.roll(mask, shift=shift, axis=(1, 2))

        if pytorch:
            mask = torch.tensor(mask.astype(np.float32), dtype=dtype, device=device)
            slm_vals = torch.tensor(
                slm_vals, dtype=dtype, device=device, requires_grad=requires_grad
            )
            mask *= slm_vals
            mask = F.interpolate(
                mask.unsqueeze(0).unsqueeze(0),
                size=(3,) + tuple(target_dim.tolist()),
                mode="nearest",
            )
            mask = mask.squeeze()
        else:
            mask *= slm_vals
            mask = cv2.resize(
                np.transpose(mask, (1, 2, 0)),
                dsize=(target_dim[1], target_dim[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            mask = np.transpose(mask, (2, 0, 1))

    return mask


def get_active_pixel_dim(
    sensor_config,
    sensor_crop,
    slm_config,
):
    """
    Assumption is that SLM is larger than sensor.

    TODO add constraint on multiple of three for RGB pixel?

    Parameters
    ----------
    sensor_config : config from waveprop.devices.sensor
    sensor_crop : fraction of sensor that is used
    slm_config : config from waveprop.devices.slm

    Returns
    -------

    """
    assert sensor_crop > 0
    assert sensor_crop <= 1

    # get overlapping pixels (SLM larger than sensor)
    overlapping_mask_dim = (
        sensor_config[SensorParam.SIZE] + slm_config[SLMParam.DEADSPACE]
    ) / slm_config[SLMParam.PITCH]
    overlapping_mask_dim = overlapping_mask_dim.astype(np.int)
    overlapping_mask_size = overlapping_mask_dim * slm_config[SLMParam.PITCH]

    # crop out a region
    # cropped_mask_size = sensor_dim * sensor_pixel_size * sensor_crop
    cropped_mask_size = overlapping_mask_size * sensor_crop

    # determine number of active SLM cells
    n_active_slm_pixels = (cropped_mask_size + slm_config[SLMParam.DEADSPACE]) / slm_config[
        SLMParam.PITCH
    ]
    n_active_slm_pixels = n_active_slm_pixels.astype(np.int)

    return overlapping_mask_size, overlapping_mask_dim, n_active_slm_pixels


def get_deadspace(slm_size, slm_dim, pixel_size):
    """
    Get amount of deadspace along each dimensions for an individual pixel.

    Parameters
    ----------
    slm_size : array_like
        Dimensions of SLM in meters.
    slm_dim : array_like
        Dimensions of SLM in number of pixels (Ny, Nx).
    pixel_size : array_like
        Dimenions of each pixel in meters.

    Returns
    -------
    dead_space_pix : :py:class:`~numpy.ndarray`
        Dead space along each dimension for each pixel in meters.

    """
    assert len(slm_size) == 2
    slm_size = np.array(slm_size)
    assert len(slm_dim) == 2
    slm_dim = np.array(slm_dim)
    assert len(pixel_size) == 2
    pixel_size = np.array(pixel_size)

    dead_space = np.array(slm_size) - pixel_size * np.array(slm_dim)
    return dead_space / (np.array(slm_dim) - 1)


def get_centers(slm_dim, pixel_pitch, return_color_filter=False, first_color=0):
    """
    Return

    Parameters
    ----------
    slm_dim : array_like
        Dimensions of SLM in number of pixels (Ny, Nx).
    pixel_pitch : array_like
        Spacing between each pixel along each dimension.
    return_color_filter : bool
        Whether to return color filter for each center.
    first_color : int
        Which color is first row. R:0, G:1, or B:2.

    Returns
    -------
    centers : :py:class:`~numpy.ndarray`
        (Ny*Nx, 2) array of SLM pixel centers.
    """
    assert len(slm_dim) == 2
    assert len(pixel_pitch) == 2

    centers_y = np.arange(slm_dim[0])[::-1, np.newaxis] * pixel_pitch[0]
    centers_y -= np.mean(centers_y)
    centers_x = np.arange(slm_dim[1])[np.newaxis, ::-1] * pixel_pitch[1]
    centers_x -= np.mean(centers_x)
    centers = np.array(np.meshgrid(centers_y, centers_x)).T.reshape(-1, 2)
    if return_color_filter:
        cf = np.zeros((3,) + tuple(slm_dim), dtype=np.float32)
        for i in range(slm_dim[0]):
            cf[(i + first_color) % 3, i] = 1
        return centers, cf.reshape(3, -1)
    else:
        return centers
