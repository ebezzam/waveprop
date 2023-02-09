import numpy as np
import progressbar


def gerchberg_saxton(target_amp, source_amp=None, n_iter=100):
    """
    Iterative Fourier transform algorithm to determine phase pattern given
    a Fourier relationship between source and target plane.

    Parameters
    ----------
    target_amp : array_like
        Target amplitude.
    source_amp : array_like
        Source amplitude, e.g. beam shape.
    n_iter : int
        Number of iterations to run algorithm.
    """

    Nx = target_amp.shape[1]
    Ny = target_amp.shape[0]

    # initialize source if necessary
    if source_amp is None:
        source_amp = np.ones((Ny, Nx))

    # - assumes FT as prop model, namely focus plane of lens
    # - inverse to go back to source plane
    target_amp = np.abs(np.fft.ifftshift(target_amp))
    source_amp = np.abs(np.fft.ifftshift(source_amp))
    source_field = np.fft.ifft2(target_amp)

    bar = progressbar.ProgressBar()
    for _ in bar(range(n_iter)):

        # amplitude constraint at source
        source_field = source_amp * np.exp(1j * np.angle(source_field))

        # could replace with other propagater
        target_field = np.fft.fft2(source_field)

        # amplitude contraint at target
        target_field = target_amp * np.exp(1j * np.angle(target_field))

        # update source phase
        source_field = np.fft.ifft2(target_field)

    source_phase = np.fft.fftshift(np.angle(source_field))

    return source_phase
