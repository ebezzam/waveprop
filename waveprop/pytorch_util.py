import torch
import numpy as np
from torchvision.transforms.functional import crop


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

    TODO : add support for mode (padding) and axes. And real.

    """

    s1 = in1.shape
    s2 = in2.shape
    shape = [s1[i] + s2[i] - 1 for i in range(len(s1))]

    sp1 = torch.fft.fft2(in1, shape)
    sp2 = torch.fft.fft2(in2, shape)
    ret = torch.fft.ifft2(sp1 * sp2, shape)

    # same shape
    y_pad_edge = int((shape[0] - s1[0]) / 2)
    x_pad_edge = int((shape[1] - s1[1]) / 2)
    _real = crop(ret.real, top=y_pad_edge, left=x_pad_edge, height=s1[0], width=s1[1])
    _imag = crop(ret.imag, top=y_pad_edge, left=x_pad_edge, height=s1[0], width=s1[1])
    return torch.complex(_real, _imag)
