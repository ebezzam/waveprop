import numpy as np


TOL = 1e-2


def fresnel_number(wv, dz, a):
    """
    Eq. 4.1-14 of Saleh

    Parameters
    ----------
    wv: wavelength [m]
    dz : distance to observation plane [m]
    a : radius of observation plane [m]
    """
    return a**2 / wv / dz


def fresnel_saleh(wv, dz, x, y, tol=TOL, verbose=True):
    """
    Eq. 4.1-13

    Parameters
    ----------
    wv: wavelength [m]
    dz : distance to observation plane [m]
    x : observation x-coordinates [m]
    y : observation y-coordinates [m]
    tol : threshold for satisfying condition << 1

    """
    # output max radius
    a = np.sqrt(np.max(x**2 + y**2))

    cond = (tol * a**4 / 4 / wv) ** (1 / 3)
    if verbose:
        if dz > cond:
            print("Saleh Fresnel condition [>{} m] met!".format(cond))
        else:
            print("Saleh Fresnel condition [>{} m] NOT met!".format(cond))
    return dz > cond

    # N_F = fresnel_number(wv, dz, a)
    # theta_m = a / dz
    # return N_F * theta_m ** 2 / 4 < tol


def fraunhofer_valid_output_region(wv, dz, tol=TOL):
    """
    From Saleh, by looking at Fresnel number, Eq 4.2-2
    """
    return np.sqrt(wv * dz * tol)


def fresnel_valid_output_region(wv, dz, tol=TOL):
    """
    From Saleh, by looking at Fresnel number, Eq 4.1-13 isolate radius `a`
    """
    return (wv * dz**3 * 4 * tol) ** (1 / 4)


def distance_from_output_region(wv, r_out, tol=TOL):
    return r_out**2 / wv / tol


def fraunhofer_saleh(wv, dz, x1, y1, x2, y2, tol=TOL, verbose=True):
    """
    Eq. 4.2-2

    Parameters
    ----------
    wv: wavelength [m]
    dz : distance to observation plane [m]
    x1 : aperture x-coordinates [m]
    y1 : aperture y-coordinates [m]
    x2 : observation x-coordinates [m]
    y2 : observation y-coordinates [m]
    tol : threshold for satisying condition << 1

    """

    # output and input max radius
    a = np.sqrt(np.max(x2**2 + y2**2))
    b = np.sqrt(np.max(x1**2 + y1**2))

    cond_in = tol * b**2 / wv
    cond_out = tol * a**2 / wv
    cond = max(cond_in, cond_out)
    if verbose:
        if dz > cond:
            print("Saleh Fraunhofer condition [>{} m] met!".format(cond))
        else:
            print("Saleh Fraunhofer condition [>{} m] NOT met!".format(cond))
    return dz > cond


def fraunhofer_schmidt(wv, dz, diam, verbose=True):
    """
    Eq. 4.2 of Numerical Simulation of Optical Wave Propagation with Examples in MATLAB.

    https://www.spiedigitallibrary.org/ebooks/PM/Numerical-Simulation-of-Optical-Wave-Propagation-with-Examples-in-MATLAB/eISBN-9780819483270/10.1117/3.866274

    Less stringent "antenna designer's formula" as noted in Goodman, E.q. 4-27.

    Parameters
    ----------
    wv: wavelength [m]
    dz : distance to observation plane [m]
    diam : diameter of source aperture[m]

    """
    cond = 2 * diam**2 / wv
    if verbose:
        if dz > cond:
            print("Weak Fraunhofer condition [>{} m] met!".format(cond))
        else:
            print("Weak Fraunhofer condition [>{} m] NOT met!".format(cond))
    return dz > cond


def fresnel_goodman(wv, dz, x1, y1, x2, y2, tol=TOL, verbose=True, get_val=False):
    """
    Eq. 4-18 (Second Edition).

    Parameters
    ----------
    wv: wavelength [m]
    dz : distance to observation plane [m]
    x1 : aperture x-coordinates [m]
    y1 : aperture y-coordinates [m]
    x2 : observation x-coordinates [m]
    y2 : observation y-coordinates [m]
    """
    cond = (tol * np.max((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 2 * np.pi / 4 / wv) ** (1 / 3)
    if verbose:
        if dz > cond:
            print("Goodman Fresnel condition [>{} m] met!".format(cond))
        else:
            print("Goodman Fresnel condition [>{} m] NOT met!".format(cond))
    if not get_val:
        return dz > cond
    else:
        return cond


def fraunhofer_goodman(wv, dz, x1, y1, x2, y2, tol=TOL, verbose=True):
    """
    Eq. 4-24 (Second Edition).

    Parameters
    ----------
    wv: wavelength [m]
    dz : distance to observation plane [m]
    x1 : aperture x-coordinates [m]
    y1 : aperture y-coordinates [m]
    x2 : observation x-coordinates [m]
    y2 : observation y-coordinates [m]
    """

    # (typically) weaker Fresnel
    cond = fresnel_goodman(wv, dz, x1, y1, x2, y2, tol, verbose=False, get_val=True)

    # (typically) stricter Fraunhofer condition
    k = 2 * np.pi / wv
    max_rad = np.max(x1**2 + y1**2)
    cond = max(k * max_rad / 2 * tol, cond)
    if verbose:
        if dz > cond:
            print("Goodman Fraunhofer condition [>{} m] met!".format(cond))
        else:
            print("Goodman Fraunhofer condition [>{} m] NOT met!".format(cond))
    return dz > cond
