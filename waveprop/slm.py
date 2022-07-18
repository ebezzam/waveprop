from turtle import pu
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
    shift=0,
    # return_slm_vals=False,
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

    if SLMParam.COLOR_FILTER in slm_config.keys():
        n_color_filter = np.prod(slm_config["color_filter"].shape[:2])
    else:
        n_color_filter = 1

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
        # TODO check
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
            mask = torch.zeros((n_color_filter, len(y), x.shape[1]), dtype=dtype, device=device)
        else:
            slm_vals_flat = slm_vals.reshape(-1)
            mask = np.zeros((n_color_filter, len(y), x.shape[1]), dtype=np.float32)
            if pytorch:
                slm_vals_flat = torch.tensor(
                    slm_vals_flat, dtype=dtype, device=device, requires_grad=requires_grad
                )
                mask = torch.tensor(mask, dtype=dtype, device=device)

        centers = get_centers(n_active_slm_pixels, pixel_pitch=slm_config[SLMParam.PITCH])
        if SLMParam.COLOR_FILTER in slm_config.keys():
            cf = get_color_filter(
                slm_dim=n_active_slm_pixels,
                color_filter=slm_config[SLMParam.COLOR_FILTER],
                shift=shift,
                flat=True,
            )
        else:
            # monochrome
            cf = None

        _height_pixel, _width_pixel = (slm_config[SLMParam.CELL_SIZE] / d1).astype(int)

        for i, _center in enumerate(centers):

            _center_pixel = (_center / d1 + target_dim / 2).astype(int)
            _center_top_left_pixel = (
                _center_pixel[0] - np.floor(_height_pixel / 2).astype(int),
                _center_pixel[1] + 1 - np.floor(_width_pixel / 2).astype(int),
            )

            if cf is not None:
                _rect = np.tile(cf[i][:, np.newaxis, np.newaxis], (1, _height_pixel, _width_pixel))
            else:
                _rect = np.ones((1, _height_pixel, _width_pixel))

            if pytorch:
                _rect = torch.tensor(_rect).to(slm_vals_flat)

            mask[
                :,
                _center_top_left_pixel[0] : _center_top_left_pixel[0] + _height_pixel,
                _center_top_left_pixel[1] : _center_top_left_pixel[1] + _width_pixel,
            ] = (
                slm_vals_flat[i] * _rect
            )

    else:

        if SLMParam.COLOR_FILTER in slm_config.keys():
            cf = get_color_filter(
                slm_dim=overlapping_mask_dim,
                color_filter=slm_config[SLMParam.COLOR_FILTER],
                shift=shift,
                flat=False,
            )
        else:
            cf = np.ones((n_color_filter, len(y), x.shape[1]), dtype=np.float32)
        cf[n_active_slm_pixels[0] :, :, :] = 0
        cf[:, n_active_slm_pixels[1] :, :] = 0
        shift_center = (
            (np.array(overlapping_mask_dim) - np.array(n_active_slm_pixels)) / 2
        ).astype(int)
        cf = np.roll(cf, shift=shift_center, axis=(0, 1))
        cf = np.flipud(cf)  # so that indexing starts in top left
        mask = np.transpose(cf, (2, 0, 1))

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


def get_centers(slm_dim, pixel_pitch, return_mesh=True):
    """
    Return

    Parameters
    ----------
    slm_dim : array_like
        Dimensions of SLM in number of pixels (Ny, Nx).
    pixel_pitch : array_like
        Spacing between each pixel along each dimension.

    Returns
    -------
    centers : :py:class:`~numpy.ndarray`
        (Ny*Nx, 2) array of SLM pixel centers if `return_mesh.
        (Nx, 2), (Ny, 2) array of SLM pixel centers otherwise.
    """
    assert len(slm_dim) == 2
    assert len(pixel_pitch) == 2

    centers_y = np.arange(slm_dim[0])[::-1, np.newaxis] * pixel_pitch[0]
    centers_y -= np.mean(centers_y)
    centers_x = np.arange(slm_dim[1])[np.newaxis, ::-1] * pixel_pitch[1]
    centers_x -= np.mean(centers_x)
    if return_mesh:
        return np.array(np.meshgrid(centers_y, centers_x)).T.reshape(-1, 2)
    else:
        return centers_y, centers_x
    # if return_color_filter:
    #     cf = np.zeros((3,) + tuple(slm_dim), dtype=np.float32)
    #     for i in range(slm_dim[0]):
    #         cf[(i + first_color) % 3, i] = 1
    #     return centers, cf.reshape(3, -1)
    # else:
    #     return centers


def get_color_filter(slm_dim, color_filter, shift=0, flat=True, separable=False):
    """
    Replicate color filter for full SLM mask.

    Parameters
    ----------
    slm_dim : array_like
        Dimensions of SLM in number of pixels (Ny, Nx).
    color_filter : array_like
        Ny x Nx array of length-3 tuples.
    shift : int
        By how much to shift color filter (due to cropping).
        TODO support 2D.
    flat : bool
        return as (Ny*Nx, 3)
    separable : bool
        return as separable filter. One of dimensions of color filter must be !
    """

    rep_y = int(np.ceil(slm_dim[0] / color_filter.shape[0]))
    rep_x = int(np.ceil(slm_dim[1] / color_filter.shape[1]))

    if not separable:
        cf = np.tile(np.roll(color_filter, shift=shift), reps=(rep_y, rep_x, 1))
        if flat:
            return cf[: slm_dim[0], : slm_dim[1]].reshape(-1, 3).astype(np.float32)
        else:
            return cf[: slm_dim[0], : slm_dim[1]].astype(np.float32)

    else:

        if color_filter.shape[0] == 1:
            cf_col = np.ones((slm_dim[0], 1, 3), dtype=np.float32)
            cf_row = np.tile(np.roll(color_filter, shift=shift), reps=(1, rep_x, 1))
        elif color_filter.shape[1] == 1:
            cf_col = np.tile(np.roll(color_filter, shift=shift), reps=(rep_y, 1, 1))
            cf_row = np.ones((1, slm_dim[1], 3), dtype=np.float32)
        else:
            raise ValueError("Color filter must have a dimension equal to 1 for separability.")

        # import pudb; pudb.set_trace()

        return cf_col, cf_row

    # import pudb; pudb.set_trace()

    # cf = np.zeros((3,) + tuple(slm_dim), dtype=np.float32)
    # for i in range(slm_dim[0]):
    #     cf[(i + first_color) % 3, i] = 1
    # return cf.reshape(3, -1)


def get_centers_separable(slm_dim, pixel_pitch, return_color_filter=False, first_color=0):
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
    centers_separable : tuple
        ((Nx, 2), (Ny, 2)) array of SLM pixel centers.
    """
    assert len(slm_dim) == 2
    assert len(pixel_pitch) == 2

    centers_y = np.arange(slm_dim[0])[::-1, np.newaxis] * pixel_pitch[0]
    centers_y -= np.mean(centers_y)
    centers_x = np.arange(slm_dim[1])[np.newaxis, ::-1] * pixel_pitch[1]
    centers_x -= np.mean(centers_x)
    centers = (centers_y, centers_x)

    if return_color_filter:
        cf_col = np.zeros((3, slm_dim[0], 1), dtype=np.float32)
        cf_row = np.ones((3, 1, slm_dim[1]), dtype=np.float32)
        for i in range(slm_dim[0]):
            cf_col[(i + first_color) % 3, i] = 1
        return centers, [cf_col, cf_row]
    else:
        return centers


def get_slm_mask_separable(
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
    shift=0,
    # first_color=0,
    return_slm_vals=False,
    requires_grad=True,
):
    """
    Parameters
    ----------`
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

    if SLMParam.COLOR_FILTER in slm_config.keys():
        n_color_filter = np.prod(slm_config["color_filter"].shape[:2])
    else:
        n_color_filter = 1

    # load mask pattern
    if slm_vals is not None:
        """use provided values"""
        if deadspace:
            assert np.array_equal(slm_vals[0].shape, (n_active_slm_pixels[0], 1))
            assert np.array_equal(slm_vals[1].shape, (1, n_active_slm_pixels[1]))
        else:
            assert np.array_equal(slm_vals[0].shape, (overlapping_mask_dim[0], 1))
            assert np.array_equal(slm_vals[1].shape, (1, overlapping_mask_dim[1]))

        if torch.is_tensor(slm_vals[0]) and torch.is_tensor(slm_vals[1]):
            pytorch = True
            device = slm_vals[0].device
        else:
            pytorch = False

    elif slm_pattern is not None:
        raise NotImplementedError

    else:
        """randomly generate"""
        if deadspace:
            _slm_vals_col = np.random.rand(*(n_active_slm_pixels[0], 1)).astype(np.float32)
            _slm_vals_row = np.random.rand(*(1, n_active_slm_pixels[1])).astype(np.float32)
            slm_vals = [_slm_vals_col, _slm_vals_row]

        else:
            _slm_vals_col = np.random.rand(*(overlapping_mask_dim[0], 1)).astype(np.float32)
            _slm_vals_row = np.random.rand(*(1, overlapping_mask_dim[1])).astype(np.float32)
            slm_vals = [_slm_vals_col, _slm_vals_row]

    # create mask
    if deadspace:
        if torch.is_tensor(slm_vals[0]):
            _mask_col = torch.zeros((n_color_filter, y.shape[0], 1), dtype=dtype, device=device)
            _mask_row = torch.zeros((n_color_filter, 1, x.shape[1]), dtype=dtype, device=device)

        else:
            _mask_col = np.zeros((n_color_filter, y.shape[0], 1), dtype=np.float32)
            _mask_row = np.zeros((n_color_filter, 1, x.shape[1]), dtype=np.float32)

            if pytorch:
                slm_vals[0] = torch.tensor(
                    slm_vals[0], dtype=dtype, device=device, requires_grad=requires_grad
                )
                slm_vals[1] = torch.tensor(
                    slm_vals[1], dtype=dtype, device=device, requires_grad=requires_grad
                )
                _mask_col = torch.tensor(_mask_col, dtype=dtype, device=device)
                _mask_row = torch.tensor(_mask_row, dtype=dtype, device=device)

        # centers, cf = get_centers_separable(
        #     n_active_slm_pixels,
        #     pixel_pitch=slm_config[SLMParam.PITCH],
        #     return_color_filter=True,
        #     first_color=first_color,
        # )
        centers = get_centers(
            n_active_slm_pixels, pixel_pitch=slm_config[SLMParam.PITCH], return_mesh=False
        )
        if SLMParam.COLOR_FILTER in slm_config.keys():
            cf = get_color_filter(
                slm_dim=n_active_slm_pixels,
                color_filter=slm_config[SLMParam.COLOR_FILTER],
                shift=shift,
                flat=False,
                separable=True,
            )
        else:
            cf = None

        if return_slm_vals:
            print("test")
            if pytorch:
                cf[0] = torch.tensor(cf[0]).to(slm_vals[0])
                cf[1] = torch.tensor(cf[1]).to(slm_vals[1])
            return (cf[0] * slm_vals[0], cf[1] * slm_vals[1]), centers

        _height_pixel, _width_pixel = (slm_config[SLMParam.CELL_SIZE] / d1).astype(int)

        # Make mask_col
        for i, _center in enumerate(centers[0][:, 0]):

            _center_top_pixel = (_center / d1[0] + target_dim[0] / 2).astype(int) - np.floor(
                _height_pixel / 2
            ).astype(int)

            # Let the possibility to have non-RGB filter
            _rect = np.tile(cf[0][i, 0][:, np.newaxis, np.newaxis], (1, _height_pixel, 1))

            if pytorch:
                _rect = torch.tensor(_rect).to(slm_vals[0])

            _mask_col[:, _center_top_pixel : _center_top_pixel + _height_pixel] = (
                slm_vals[0][i, 0] * _rect
            )

        # Make mask_row
        for i, _center in enumerate(centers[1][0, :]):

            _center_left_pixel = (
                (_center / d1[1] + target_dim[1] / 2).astype(int)
                - np.floor(_width_pixel / 2).astype(int)
                + 1
            )
            # Let the possibility to have non-RGB filter
            _rect = np.tile(cf[1][0, i][:, np.newaxis, np.newaxis], (1, 1, _width_pixel))

            if pytorch:
                _rect = torch.tensor(_rect).to(slm_vals[1])

            _mask_row[:, :, _center_left_pixel : _center_left_pixel + _width_pixel] = (
                slm_vals[1][0, i] * _rect
            )

    else:

        _mask_col = np.zeros((3, overlapping_mask_dim[0], 1), dtype=np.float32)
        _mask_row = np.zeros((3, 1, overlapping_mask_dim[1]), dtype=np.float32)

        for i in range(n_active_slm_pixels[0]):
            _mask_col[(i + first_color) % 3, n_active_slm_pixels[0] - 1 - i] = 1
        for i in range(n_active_slm_pixels[1]):
            _mask_row[:, :, : n_active_slm_pixels[1]] = 1

        shift = ((np.array(overlapping_mask_dim) - np.array(n_active_slm_pixels)) / 2).astype(int)
        _mask_col = np.roll(_mask_col, shift=shift[0], axis=1)
        _mask_row = np.roll(_mask_row, shift=shift[1], axis=2)

        if pytorch:
            _mask_col = torch.tensor(_mask_col.astype(np.float32), dtype=dtype, device=device)
            _mask_row = torch.tensor(_mask_row.astype(np.float32), dtype=dtype, device=device)

            slm_vals[0] = torch.tensor(
                slm_vals[0], dtype=dtype, device=device, requires_grad=requires_grad
            )
            slm_vals[1] = torch.tensor(
                slm_vals[1], dtype=dtype, device=device, requires_grad=requires_grad
            )

            _mask_col *= slm_vals[0]
            _mask_row *= slm_vals[1]

            _mask_col = F.interpolate(
                _mask_col.unsqueeze(0),
                size=(target_dim[0], 1),
                mode="nearest",
            ).squeeze(0)
            _mask_row = F.interpolate(
                _mask_row.unsqueeze(0),
                size=(1, target_dim[1]),
                mode="nearest",
            ).squeeze(0)

        else:
            _mask_col *= slm_vals[0]
            _mask_row *= slm_vals[1]

            _mask_col = cv2.resize(
                np.transpose(_mask_col, (1, 2, 0)),
                dsize=(1, target_dim[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            _mask_row = cv2.resize(
                np.transpose(_mask_row, (1, 2, 0)),
                dsize=(target_dim[1], 1),
                interpolation=cv2.INTER_NEAREST,
            )

            _mask_col = np.transpose(_mask_col, (2, 0, 1))
            _mask_row = np.transpose(_mask_row, (2, 0, 1))

    return [_mask_col, _mask_row]
