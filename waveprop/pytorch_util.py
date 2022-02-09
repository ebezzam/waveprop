import torch
import numpy as np
from torchvision.transforms.functional import crop


ctypes = [torch.complex64, torch.complex128]


def compute_numpy_error(tensor, numpy_array, verbose=True):
    err = np.mean(np.abs((numpy_array - tensor.cpu().numpy()) ** 2))
    if verbose:
        print("- tensor -")
        print("shape", tensor.shape)
        print("type", tensor.dtype)
        print("- numpy array -")
        print("shape", numpy_array.shape)
        print("type", numpy_array.dtype)
        print("MSE wrt numpy", err)
    return err


def fftconvolve(in1, in2, mode=None, axes=None):
    """
    https://github.com/scipy/scipy/blob/v1.7.1/scipy/signal/signaltools.py#L554-L668

    TODO : add support for mode (padding) and axes

    """

    s1 = in1.shape
    s2 = in2.shape
    if axes is None:
        shape = [s1[i] + s2[i] - 1 for i in range(len(s1))]
    else:
        shape = [s1[i] + s2[i] - 1 for i in axes]
    if mode is not None:
        if mode != "same":
            raise ValueError(f"{mode} mode not supported ")

    is_complex = False
    if in1.dtype in ctypes or in2.dtype in ctypes:
        is_complex = True
        sp1 = torch.fft.fftn(in1, shape, dim=axes)
        sp2 = torch.fft.fftn(in2, shape, dim=axes)
        ret = torch.fft.ifftn(sp1 * sp2, shape, dim=axes)
    else:
        sp1 = torch.fft.rfftn(in1, shape, dim=axes)
        sp2 = torch.fft.rfftn(in2, shape, dim=axes)
        ret = torch.fft.irfftn(sp1 * sp2, shape, dim=axes)

    # same shape, mode="same"
    # TODO : assuming 2D here
    if axes is None:
        y_pad_edge = int((shape[0] - s1[0]) / 2)
        x_pad_edge = int((shape[1] - s1[1]) / 2)
        if is_complex:
            _real = crop(ret.real, top=y_pad_edge, left=x_pad_edge, height=s1[0], width=s1[1])
            _imag = crop(ret.imag, top=y_pad_edge, left=x_pad_edge, height=s1[0], width=s1[1])
            return torch.complex(_real, _imag)
        else:
            return crop(ret, top=y_pad_edge, left=x_pad_edge, height=s1[0], width=s1[1])
    else:
        y_pad_edge = int((shape[0] - s1[axes[0]]) / 2)
        x_pad_edge = int((shape[1] - s1[axes[1]]) / 2)
        if is_complex:
            _real = crop(
                ret.real, top=y_pad_edge, left=x_pad_edge, height=s1[axes[0]], width=s1[axes[1]]
            )
            _imag = crop(
                ret.imag, top=y_pad_edge, left=x_pad_edge, height=s1[axes[0]], width=s1[axes[1]]
            )
            return torch.complex(_real, _imag)
        else:
            return crop(ret, top=y_pad_edge, left=x_pad_edge, height=s1[axes[0]], width=s1[axes[1]])
