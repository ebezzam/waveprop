import numpy as np
import torch
from waveprop.util import crop, rect2d, sample_points
import torch.nn.functional as F
import cv2


COLOR_ORDER = [0, 1, 2]  # R:0, G:1, B:2 indices, verified with measurements


def get_slm_mask(
    slm_dim,
    slm_size,
    slm_pixel_dim,
    rpi_dim,
    rpi_pixel_dim,
    crop_fact,
    N,
    slm_vals=None,
    slm_pattern=None,
    deadspace=True,
    pattern_shift=None,
    pytorch=False,
    device="cuda",
    dtype=np.float32,
    first_color=0,
):
    """
    TODO : directly pass slm_vals when optimizing

    Parameters
    ----------
    slm_dim
    slm_size
    slm_pixel_dim
    rpi_dim
    rpi_pixel_dim
    crop_fact
    N
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
    dead_space_pix = get_deadspace(slm_size, slm_dim, slm_pixel_dim)
    pixel_pitch = slm_pixel_dim + dead_space_pix
    overlapping_mask_size, overlapping_mask_dim, n_active_slm_pixels = get_active_pixel_dim(
        sensor_dim=rpi_dim,
        sensor_pixel_size=rpi_pixel_dim,
        sensor_crop=crop_fact,
        slm_size=slm_size,
        slm_dim=slm_dim,
        slm_pixel_size=slm_pixel_dim,
    )
    d1 = np.array(overlapping_mask_size) / N
    x, y = sample_points(N=N, delta=d1)

    # create mask
    if deadspace:

        mask = np.zeros((3, len(y), x.shape[1]), dtype=np.float32)
        if slm_vals is not None:
            raise NotImplementedError

        elif slm_pattern is not None:
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
            first_color = COLOR_ORDER[top_left[0] % len(COLOR_ORDER)]
            slm_vals = crop(slm_pattern_values, shape=n_active_slm_pixels, topleft=top_left).astype(
                np.float32
            )
        else:
            slm_vals = np.random.rand(*n_active_slm_pixels).astype(np.float32)

        slm_vals_flat = slm_vals.reshape(-1)
        if pytorch:
            slm_vals_flat = torch.tensor(
                slm_vals_flat, dtype=dtype, device=device, requires_grad=True
            )
            mask = torch.tensor(mask, dtype=dtype, device=device)

        centers, cf = get_centers(
            n_active_slm_pixels,
            pixel_pitch=pixel_pitch,
            return_color_filter=True,
            first_color=first_color,
        )
        for i, _center in enumerate(centers):
            ap = rect2d(x, y, slm_pixel_dim, offset=_center).astype(np.float32)
            ap = np.tile(ap, (3, 1, 1)) * cf[:, i][:, np.newaxis, np.newaxis]
            if pytorch:
                # TODO : is pytorch autograd compatible with in-place?
                # https://pytorch.org/docs/stable/notes/autograd.html#in-place-operations-with-autograd
                # https://discuss.pytorch.org/t/what-is-in-place-operation/16244/15
                index_tensor = torch.tensor([i], dtype=torch.int, device=device)
                mask += torch.tensor(ap, dtype=dtype, device=device) * torch.index_select(
                    slm_vals_flat, 0, index_tensor
                )
            else:
                mask += ap * slm_vals_flat[i]

    else:

        if slm_vals is not None:
            raise NotImplementedError

        elif slm_pattern is not None:
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
            first_color = COLOR_ORDER[top_left[0] % len(COLOR_ORDER)]
            slm_vals = crop(
                slm_pattern_values, shape=overlapping_mask_dim, topleft=top_left
            ).astype(np.float32)

        else:
            slm_vals = np.random.rand(*overlapping_mask_dim).astype(np.float32)

        mask = np.zeros((3,) + tuple(overlapping_mask_dim), dtype=np.float32)
        for i in range(n_active_slm_pixels[0]):
            mask[
                (i + first_color) % 3, n_active_slm_pixels[0] - 1 - i, : n_active_slm_pixels[1]
            ] = 1
        shift = ((np.array(overlapping_mask_dim) - np.array(n_active_slm_pixels)) / 2).astype(int)
        mask = np.roll(mask, shift=shift, axis=(1, 2))

        if pytorch:
            mask = torch.tensor(mask.astype(np.float32), dtype=dtype, device=device)
            slm_vals = torch.tensor(slm_vals, dtype=dtype, device=device, requires_grad=True)
            mask *= slm_vals
            mask = F.interpolate(
                mask.unsqueeze(0).unsqueeze(0), size=(3,) + tuple(N.tolist()), mode="nearest"
            )
            mask = mask.squeeze()
        else:
            mask *= slm_vals
            mask = cv2.resize(
                np.transpose(mask, (1, 2, 0)), dsize=(N[1], N[0]), interpolation=cv2.INTER_NEAREST
            )
            mask = np.transpose(mask, (2, 0, 1))

    return mask


def get_active_pixel_dim(
    sensor_dim, sensor_pixel_size, sensor_crop, slm_size, slm_dim, slm_pixel_size
):
    """
    TODO add constraint on multiple of three for RGB pixel

    Parameters
    ----------
    sensor_dim : dimension of camera in pixels
    sensor_pixel_size : dimension of individual pixel on camera in meters
    sensor_crop : fraction of sensor that is used
    slm_size : dimension of SLM in meters
    slm_dim : dimension of SLM in pixels
    slm_pixel_size : dimension of individual SLM pixel in meters.

    Returns
    -------

    """
    assert sensor_crop > 0
    assert sensor_crop <= 1

    sensor_dim = np.array(sensor_dim)
    sensor_pixel_dim = np.array(sensor_pixel_size)

    # TODO could be different to due deadspace in sensor?
    sensor_size = sensor_dim * sensor_pixel_dim

    # determine SLM pitch
    slm_pixel_dead_space = get_deadspace(slm_size, slm_dim, slm_pixel_size)
    slm_pixel_pitch = slm_pixel_size + slm_pixel_dead_space

    # get overlapping pixels
    overlapping_mask_dim = (sensor_size + slm_pixel_dead_space) / slm_pixel_pitch
    overlapping_mask_dim = overlapping_mask_dim.astype(np.int)
    overlapping_mask_size = overlapping_mask_dim * slm_pixel_pitch

    # crop out a region
    # cropped_mask_size = sensor_dim * sensor_pixel_size * sensor_crop
    cropped_mask_size = overlapping_mask_size * sensor_crop
    n_active_slm_pixels = (cropped_mask_size + slm_pixel_dead_space) / slm_pixel_pitch
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
