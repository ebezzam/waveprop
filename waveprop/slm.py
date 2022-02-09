import numpy as np


def get_active_pixel_dim(
    sensor_dim, sensor_pixel_size, sensor_crop, slm_size, slm_dim, slm_pixel_size, deadspace
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


def get_centers(slm_dim, pixel_pitch, return_color_filter=False):
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
        (Ny*Nx, 2) array of SLM pixel centers.
    """
    assert len(slm_dim) == 2
    assert len(pixel_pitch) == 2

    centers_y = np.arange(slm_dim[0])[:, np.newaxis] * pixel_pitch[0]
    centers_y -= np.mean(centers_y)
    centers_x = np.arange(slm_dim[1])[np.newaxis, :] * pixel_pitch[1]
    centers_x -= np.mean(centers_x)
    centers = np.array(np.meshgrid(centers_y, centers_x)).T.reshape(-1, 2)
    if return_color_filter:
        cf = np.zeros((3,) + tuple(slm_dim), dtype=np.float32)
        for i in range(slm_dim[0]):
            cf[i % 3, i] = 1
        return centers, cf.reshape(3, -1)
    else:
        return centers
