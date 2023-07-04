import numpy as np
from scipy import signal


def normalize(v):
    """
    Normalise a 3D vector map (an array with shape[w,h,3])
    """
    return v / np.sqrt((v*v).sum(axis=2)[:,:,np.newaxis])

def normals_to_image(normals, opengl=False):
    """
    Converts a normal map from vector space [-1:1]^3 to a RGB space [0;255]^3
    Default mapping is DirectX (R,G,B) <-> (x,y,z)
    Can be specified to OpenGL if needed (R,G,B) <-> (x,1-y,z)
    """
    normals = normalize(normals) # ensure that the normals are indeed normalized
    if opengl:
        normals *= [[[1, -1, 1]]]
    image = ((normals+1) * 255/2).astype(np.float32)
    return image

def image_to_normals(img, opengl=False):
    """
    Converts a normal map from RGB space [0;255]^3 to vector space [-1;1]^3
    """
    normals = img.astype(np.float32) * (2/255) - 1
    normals = normalize(normals)  # ensure that the normals are indeed normalized
    if opengl:
        normals *= [[[1, -1, 1]]]
    return normals

def sobel_kernel_x(r):
    kernel = np.array([[-i/(i*i+j*j) if i != 0 else 0 for i in r] for j in r])
    normalized = kernel * (2 / (np.sum(np.abs(kernel))))
    return normalized


def heights_to_normals(height_map, width, height, thickness, kernel_size=3, clamp=True, crop=True):
    """
    Compute a normal map form the gradient of a height map using the Sobel operator

    Parameters
    ----------
    height_map : the data of the height map. Is automatically normalized.
    width, height, thickness : the dimensions of the diffuser in metric units
    kernel_size : the kernel for sobel operator. Default is 3. Can be increased to compute the gradient of the
    height map over a larger number of pixels, yielding smoother but less precise normals
    clamp : whether to clamp the value of the approximated gradient to the greatest value that it could physically take
    crop : whether to crop the border of the resulting image to avoid border artifacts

    Returns a normal map corresponding to the given height map
    """
    assert 1 < len(height_map.shape) < 4
    assert 0 < thickness
    assert 0 < width
    assert 0 < height
    assert 2 < kernel_size

    # if needed, we convert the image from rgb to grayscale
    if len(height_map.shape) == 3:
        height_map = np.sum(height_map, axis=2) / 3

    # we normalize the height map so that it ranges from 0 to 1.
    height_map /= np.max(height_map)

    half = (kernel_size + 1) // 2
    sobel_x = sobel_kernel_x(range(-half, half+1))
    sobel_y = np.swapaxes(sobel_x, 0, 1)

    normal_x = signal.fftconvolve(height_map, sobel_x)
    normal_y = signal.fftconvolve(height_map, sobel_y)

    # the gradient computed yet accounts for the fact that the distance between the center of two adjacent
    # pixels is equal to the thickness of the diffuser. It is however not generally the case, and we have to
    # correct it for it to account for the scaling, using a factor depending on the diffuser size.
    fx = width / (thickness * height_map.shape[0])
    fy = height / (thickness * height_map.shape[1])
    x2 = normal_x * normal_x
    y2 = normal_y * normal_y
    normal_x *= np.sqrt((1 / (fx*fx*(1-x2)+x2)))
    normal_y *= np.sqrt((1 / (fy*fy*(1-y2)+y2)))

    # clamp the gradient so that it can not exceed the slope that would be between two adjacent pixels
    if clamp:
        normal_x = np.minimum(np.ones(normal_x.shape) / np.cos(np.arctan(fx)), normal_x)
        normal_y = np.minimum(np.ones(normal_y.shape) / np.cos(np.arctan(fy)), normal_y)

    normal_z2 = (1 - normal_x*normal_x - normal_y*normal_y)

    # float precision can cause slightly under 0 negative values,
    normal_z2 = np.maximum(np.zeros(normal_z2.shape), normal_z2)
    normal_z = np.sqrt(normal_z2)

    result = np.stack((normal_x,normal_y,normal_z),axis=2)
    c = kernel_size+1
    return result[c:-c, c:-c, :] if crop else result

if __name__ == "__main__":
    import cv2


    normals = heights_to_normals(cv2.imread("/home/julien-sahli/git/height2.png"), 36, 36, 0.1)
    img = normals_to_image(normals)
    cv2.imwrite("/home/julien-sahli/git/this.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


    quit()
