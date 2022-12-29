import numpy as np
from scipy import ndimage


def add_shot_noise(image, snr_db, tol=1e-6):
    """
    Add shot noise to image.

    Parameters
    ----------
    image : np.ndarray
        Image.
    snr_db : float
        Signal-to-noise ratio in dB.
    tol : float, optional
        Tolerance for noise variance, by default 1e-6.

    Returns
    -------
    np.ndarray
        Image with added shot noise.

    """

    if image.min() < 0:
        image -= image.min()
    noise = np.random.poisson(image)

    sig_var = ndimage.variance(image)
    noise_var = np.maximum(ndimage.variance(noise), tol)
    fact = np.sqrt(sig_var / noise_var / (10 ** (snr_db / 10)))

    return image + fact * noise