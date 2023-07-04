import numpy as np
import time
from simulator.utils.normals import normalize
from simulator.utils.lerp import lerp_add
from simulator.psf_generation.psf_generator import PSFGenerator


class DiffuserPSFGenerator(PSFGenerator):

    def __init__(self, camera_settings, normals, n_air, n_dif, refract_once):
        super(DiffuserPSFGenerator, self).__init__(camera_settings=camera_settings)
        self.normals = normals
        self.nx, self.ny = self.normals.shape[:2]
        self.n_air = n_air
        self.n_dif = n_dif
        self.refract_once = refract_once
        self.normalisation_factor = 1

        # Calculate the physical coordinates of each pixel of the diffuser normal map, where we will trace rays.
        x = np.arange(self.nx) / (self.nx - 1) - 0.5
        y = np.arange(self.ny) / (self.ny - 1) - 0.5
        self.ray_positions_2d = x[:, np.newaxis, np.newaxis] * [[[1, 0]]] + y[np.newaxis, :, np.newaxis] * [[[0, 1]]]

        return

    def generate_layer(self, depth):

        verbose = (not self.multiprocess) or (depth == self.depths[0])
        if verbose and self.multiprocess:
            print("Notice : as we will use one process per depth, we will only print the progress of the first one.")


        start_time = 0
        if verbose:
            start_time = time.time()
            print("Compute incident directions in local coordinates of the first interface")

        # calculate the position of the intersection of the light rays and the diffuser
        ray_positions = np.concatenate((self.ray_positions_2d, np.ones((self.nx, self.ny, 1)) * depth), axis=2)

        # keeping track of the distance will be useful to determine how much light arrives in the end
        distance_covered = np.sqrt((ray_positions*ray_positions).sum(axis=2))

        # calculate the normalization of these rays. The z-component will also give us the cosine with z-axis.
        ray_positions_norm = normalize(ray_positions)

        # represent the rays in the referential of normal map, so that the normal to the interface is the new z-axis
        incident_directions = np.moveaxis(to_local(ray_positions_norm, self.normals), 0, 2)

        if verbose:
            print(f" - Done in : {time.time() - start_time} s")
            start_time = time.time()
            print("Compute first refraction")

        # get refracted through first interface (air -> diffuser)
        in_incident_cos_theta, in_refracted_cos_theta, in_refracted_dir = self.compute_refraction(
            incident_directions, self.n_air / self.n_dif)

        if verbose:
            print(f" - Done in : {time.time() - start_time} s")
            start_time = time.time()
            print("Compute incident directions in local coordinates of the second interface")

        # represent the rays in global, so that the normal to the other interface's face is the new z-axis
        refracted_directions = np.moveaxis(to_global(in_refracted_dir, self.normals), 0, 2)

        if verbose:
            print(f" - Done in : {time.time() - start_time} s")
            start_time = time.time()
            print("Compute second refraction")

        # get refracted through second interface (diffuser -> air)
        out_incident_cos_theta, out_refracted_cos_theta, out_refracted_dir = self.compute_refraction(
            refracted_directions, 1 if self.refract_once else self.n_dif / self.n_air)

        if verbose:
            print(f" - Done in : {time.time() - start_time} s")
            start_time = time.time()
            print("Calculate the position where the rays hit the sensor")

        # calculate the trajectory of the rays between the diffuser and the sensor
        out_cos_theta = out_refracted_dir[:, :, 2]
        trajectory = self.cam.focal_distance * vect_safe_divide(
            out_refracted_dir,np.repeat(out_cos_theta[:, :, np.newaxis], 3, axis=2))
        distance_covered = distance_covered + np.sqrt((trajectory*trajectory).sum(axis=2))
        out_positions = ray_positions + trajectory

        if verbose:
            print(f" - Done in : {time.time() - start_time} s")
            start_time = time.time()
            print("Compute amount of light getting through")

        # computing amount of light getting through
        source_cos_theta = ray_positions_norm[:, :, 2]

        v_fresnel = np.vectorize(self.fresnel)
        f_in = v_fresnel(in_incident_cos_theta, in_refracted_cos_theta)
        f_out = v_fresnel(out_refracted_cos_theta, out_incident_cos_theta)

        visibility = np.vectorize(lambda z : 1 if z > 0 else 0)(out_positions[:,:,2])

        light = visibility * source_cos_theta * f_in * f_out * out_cos_theta / (distance_covered * distance_covered)
        if verbose:
            print(f" - Done in : {time.time() - start_time} s")
            start_time = time.time()
            print("Assign rays to output")

        # assigning rays to output
        # this part should be vectorized as well, but it seems tricky to do
        result = np.zeros(self.shape)
        for y in range(self.normals.shape[1]):
            for x in range(self.normals.shape[0]):
                i, j, _ = out_positions[x, y]
                if light[x, y] > 0:
                    lerp_add(result, 0.5 + i / self.cam.sensor_width, 0.5 + j / self.cam.sensor_height,
                                self.shape[0], self.shape[1], light[x, y])
        if verbose:
            print(f" - Done in : {time.time() - start_time} s")
            if self.multiprocess:
                print("Still waiting for other processes to finish...")
        return result.astype(np.float32)

    def fresnel(self, cos_in, cos_out):
        """
        Fresnel Equation is used to calculate the amount of light reaching the diffuser that is refracted instead of
        being reflected. Neglects the tiny part of the light that reflects twice inside the diffuser for performance.

        Parameters
        ----------
        cos_in : cosine of the angle between the ingoing light ray and the diffuser's normal
        cos_out : cosine of the angle between the outgoing light ray and the diffuser's normal

        Returns the factor of the light being refracted
        """
        f1 = (self.n_dif * cos_in - self.n_air * cos_out) / (self.n_dif * cos_in + self.n_air * cos_out)
        f2 = (self.n_air * cos_in - self.n_dif * cos_out) / (self.n_air * cos_in + self.n_dif * cos_out)
        return 1 - 0.5 * (f1 * f1 + f2 * f2)

    def compute_refraction(self, local_incident_directions, factor):
        incident_cos_theta = np.minimum(local_incident_directions[:, :, 2], 1)
        incident_sin_theta = np.sqrt(1 - incident_cos_theta * incident_cos_theta)
        refracted_sin_theta = np.minimum(incident_sin_theta * factor, 1)
        refracted_cos_theta = np.sqrt(1 - refracted_sin_theta * refracted_sin_theta)
        refracted_directions = np.moveaxis(refract(local_incident_directions, refracted_cos_theta),0,2)
        return incident_cos_theta, refracted_cos_theta, refracted_directions


def refract(v, new_z):
    """
    Compute the refraction of a unit vector through a medium given the z-component of the output unit vector
    The medium is assumed to have the z axis (in local coordinates) as surface normal.
    The z-component is to be computed beforehand using snell's law
    Parameters
    ----------
    v : the input vector
    new_z : the z-component of the output vector

    Returns the refraction of v which has new_z as z-component
    """

    a = np.sqrt(vect_safe_divide(1 - new_z * new_z, (v[:, :, 0] * v[:, :, 0] + v[:, :, 1] * v[:, :, 1])))
    return np.array([v[:,:,0] * a, v[:,:,1] * a, new_z])


def safe_divide(a, b):
    return 0 if b == 0 else a / b

vect_safe_divide = np.vectorize(safe_divide)

def to_local(vec, norm):
    """
    Expresses a given vector in a rotated coordinates system such that the given normal vector becomes the new z-axis.

    Parameters
    ----------
    vec : the vector to express locally
    norm : the normal which defines the local coordinates

    Returns the local expression of the vec
    """
    n = np.sqrt(norm[:,:,0] * norm[:,:,0] + norm[:,:,1] * norm[:,:,1])

    x = -norm[:,:,0]
    y = norm[:,:,1]

    x = vect_safe_divide(x, n)
    y = vect_safe_divide(y, n)

    xy = x * y
    c = norm[:,:,2]
    ic = 1 - c
    s = np.sqrt(1 - c * c)
    sx = s * x
    sy = s * y
    icxy = (ic * xy)

    return np.array([
        vec[:,:,0] * (c + x * x * ic) + vec[:,:,1] * icxy + vec[:,:,2] * sy,
        vec[:,:,0] * icxy + vec[:,:,1] * (c + y * y * ic) + vec[:,:,2] * -sx,
        vec[:,:,0] * -sy + vec[:,:,1] * sx + vec[:,:,2] * c
    ])


def to_global(vec, norm):
    """
    Inverse function of to_local. Give the global expression of a local vector.
    Parameters
    ----------
    vec : the local vector to express globally
    norm : the normal that was used to make the vector local in the first place

    Returns the global expression of vec
    """
    return to_local(vec, norm * [[[-1, -1, 1]]])
