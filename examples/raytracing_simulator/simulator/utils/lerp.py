import numpy as np

def safe_get(array, x,y ,w,h):
    """
    Used to get the value from an array at the specified indexes if it exists, or 0 else.

    Parameters
    ----------
    array : Data to get
    x,y : coordinates to get
    w,h : shape of data to get

    Returns
    -------
    array[x,y] if it exists, else 0
    """
    return np.zeros(array[0, 0].shape) if x < 0 or x >= w or y < 0 or y >= h else array[x,y]


def lerp(x, y, w, h):
    """
    Used by lerp_add and lerp_read to compute right indexes and coefficients

    Parameters
    ----------
    x, y : relative coordinates of the point to interpolate in [0;1]^2
    w, h : the width and height of the image to interpolate (in pixels)

    Returns
    -------
    a tuple containing indexes of the smallest pixel to interpolate from as well as the interpolation coefficients

    """
    x = x * w - 0.5
    y = y * h - 0.5

    x_int = int(x)
    y_int = int(y)

    x_delta = x - float(x_int)
    y_delta = y - float(y_int)

    if x_delta < 0:  # can happen when -1<x<0
        x_delta += 1
        x_int -= 1

    if y_delta < 0:  # can happen when -1<x<0
        y_delta += 1
        y_int -= 1

    return x_int, y_int, x_delta, y_delta, w, h

def lerp_add(array, x, y, w, h, value):
    """
    Adds color to an arbitrary coordinate of the image by interpolating linearly from nearby pixels and returns it

    Parameters
    ----------
    array : the array to interpolate
    x, y : relative coordinates of the point to interpolate in [0;1]^2
    w, h : the kernel_size of the array in pixels
    value : the value to add

    Returns
    -------
    the array with the added value at the correct pixels.
    """
    x1, y1, x_delta, y_delta, w, h = lerp(x, y, w, h)
    x2 = x1+1
    y2 = y1+1

    if 0 <= x1 < w:
        if 0 <= y1 < h:
            array[x1, y1] += value * (1-x_delta) * (1-y_delta)
        if 0 <= y2 < h:
            array[x1, y2] += value * (1-x_delta) * y_delta
    if 0 <= x2 < w:
        if 0 <= y1 < h:
            array[x2, y1] += value * x_delta * (1 - y_delta)
        if 0 <= y2 < h:
            array[x2, y2] += value * x_delta * y_delta

    return array


def lerp_read(array, x, y, w, h):
    """
    Return the color at the given coordinate from an image, interpolating linearly from nearby pixels.

    Parameters
    ----------
    array : the array to interpolate
    x, y : relative coordinates of the point to interpolate in [0;1]^2
    w, h : the kernel_size of the array in pixels

    Returns
    -------
    linear interpolation of the image at the given coordinate

    """

    x1, y1, x_delta, y_delta, w, h = lerp(x,y,w,h)

    x2 = x1+1
    y2 = y1+1

    v1 = (1-x_delta) * safe_get(array, x1, y1, w, h) + x_delta * safe_get(array, x2, y1, w, h)
    v2 = (1-x_delta) * safe_get(array, x1, y2, w, h) + x_delta * safe_get(array, x2, y2, w, h)

    return (1-y_delta) * v1 + y_delta * v2