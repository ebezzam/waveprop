import numpy as np
from .util import ft2, ift2, sample_points


def thin_scatterer(size, pxnum, corr_width, strength):
    """
    [corr_width] controls the 'correlation width' of the scattering layer.
    The lower, the bigger the detail of the layer. IN LENGTH UNITS

    [strength] controls the amount of balistic photons that go through the
    layer (controls if the phase jumps introduced by the thin layer are
    bigger or smaller than one wavelength)

    Adapted from Fernando Soldevilla: https://scholar.google.com/citations?user=kCv2GFEAAAAJ&hl=es
    
    TODO : PyTorch support
    """
    delta = size / pxnum  # mesh spacing (spatial domain)
    delta_f = 1 / (pxnum * delta)  # mesh spacing (frequency domain)
    corr_width_px = int(corr_width / delta)  # calculate corr_width in pixels

    # Build gaussian in spatial domain, with the desired width
    lowpass = buildGauss(
        px=pxnum,
        sigma=(corr_width_px, corr_width_px),
        center=(int(pxnum / 2), int(pxnum / 2)),
        phi=0,
    )
    # Convert to frequency domain (to do the filtering)
    lowpass_ft = ft2(lowpass, delta)
    seed = np.random.rand(pxnum, pxnum)

    # Filter in frequency domain so detail size corresponds to corr_width
    phase = np.real(ift2(ft2(seed, delta) * lowpass_ft, delta_f))
    # shift between -0.5 and 0.5
    phase = (phase - np.min(phase)) / (np.max(phase) - np.min(phase)) - 0.5

    # build final phase mask (no wrap here)
    phase = phase * 2 * np.pi * strength

    # build 'field' (wrapped phase)
    field = np.exp(1j * phase)
    x, y = sample_points(N=[pxnum, pxnum], delta=delta)

    return field, x, y


def buildGauss(px, sigma, center, phi):
    """
    buildGauss generates a Gaussian function in 2D. Formula from
    https://en.wikipedia.org/wiki/Gaussian_function

    Parameters
    ----------
    px : image size of the output (in pixels)
    sigma : 2-element vector, sigma_x
    and sigma_y for the 2D Gaussian
    center : 2-element vector, center position
    of the Gaussian in the image
    phi : Rotation angle for the Gaussian

    Returns
    -------
    gaus : 2D image with the Gaussian

    """
    # Generate mesh
    x = np.linspace(1, px, px)
    X, Y = np.meshgrid(x, x)

    # Generate gaussian parameters
    a = np.cos(phi) ** 2 / (2 * sigma[0] ** 2) + np.sin(phi) ** 2 / (2 * sigma[1] ** 2)
    b = -np.sin(2 * phi) / (4 * sigma[0] ** 2) + np.sin(2 * phi) / (4 * sigma[1] ** 2)
    c = np.sin(phi) ** 2 / (2 * sigma[0] ** 2) + np.cos(phi) ** 2 / (2 * sigma[1] ** 2)

    # Generate Gaussian
    gaus = np.exp(
        -(
            a * (X - center[0]) ** 2
            + 2 * b * (X - center[0]) * (Y - center[1])
            + c * (Y - center[1]) ** 2
        )
    )

    return gaus
