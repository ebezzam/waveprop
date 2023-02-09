import numpy as np
from scipy import ndimage
import torch


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

    if torch.is_tensor(image):
        with torch.no_grad():
            image_np = image.cpu().numpy()
    else:
        image_np = image

    if image_np.min() < 0:
        image_np -= image_np.min()
    noise = np.random.poisson(image_np)

    sig_var = ndimage.variance(image_np)
    noise_var = np.maximum(ndimage.variance(noise), tol)
    fact = np.sqrt(sig_var / noise_var / (10 ** (snr_db / 10)))

    noise = fact * noise
    if torch.is_tensor(image):
        noise = torch.from_numpy(noise).to(image.device)

    return image + fact * noise
