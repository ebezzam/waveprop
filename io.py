from PIL import Image
from pathlib import Path
from waveprop.color import rgb2gray
import numpy as np


def load_image(fp, size=None, invert=False, grayscale=False):

    img = Image.open(Path(fp)).convert("RGB")
    if size != None:
        img = img.resize(size)

    # normalize
    img = np.asarray(img) / 255.0
    if invert:
        img = 1 - img

    if grayscale:
        img = rgb2gray(img)

    return np.flip(img, axis=0)
