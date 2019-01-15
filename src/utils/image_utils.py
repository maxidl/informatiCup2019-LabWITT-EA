"""
This module provides functions for image loading and saving.
"""

import os
import warnings
import imageio
import numpy as np
from skimage import io, img_as_float32, img_as_ubyte, transform


def load_image(fname: str, size: int = None) -> np.ndarray:
    """
    Loads an image file and transforms it into float representation [0.0; 1.0]. Optionally
    resizes the image.

    Args:
        fname (str): filename or path to the image.
        size (int):  optionally, resize the image to size x size.

    Returns:
        np.array: the loaded image, dtype is float with values between 0.0 and 1.0.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        image = io.imread(fname)
        image = img_as_float32(image)
        if size is not None:
            image = transform.resize(image, (size, size), mode='constant', anti_aliasing=True)
        if len(image.shape) == 4:
            image = image[:, :, :3]
    return image


def get_filename(path):
    """
    Returns a valid filename for the specified path. Adds (x) to the filename, if a file
    with this name already exists. Avoids overwriting.

    Args:
        path (str): the destination filename

    Returns:
        fpath (str): a filepath
    """
    dpath = os.path.join(os.path.dirname(__file__), "..", "..",)
    path = os.path.join(dpath, path)
    exists = os.path.isfile(path)
    counter = 1
    fpath = path
    while exists:
        fpath = path.rsplit('.', 1)
        fpath = fpath[0] + "({}).".format(counter) + fpath[1]
        counter = counter + 1
        exists = os.path.isfile(fpath)
    return fpath


def save_image(fname, image) -> None:
    """
    Saves an image to the specified filename. Possible image shape types are (X,Y), (X,Y,1), (X,
    Y,3) or (X,Y,3,1)

    Args:
        fname (str): the destination filename
        image (np.array): the image to be saved

    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        image = image.clip(0, 1)
        image = img_as_ubyte(image)
        if image.ndim == 2:
            image = np.stack((image,) * 3, axis=-1)
        if image.ndim == 3 and image.shape[2] == 1:
            image = image.squeeze()
            image = np.stack((image,) * 3, axis=-1)

        if isinstance(fname, str):
            fname = get_filename(fname)

        io.imsave(fname, image)


def save_as_gif(fname, images) -> None:
    """
    Saves a gif to the specified filename. Possible image shape types are (X,Y), (X,Y,1), (X,
    Y,3) or (X,Y,3,1)

    Args:
        fname (str): the destination filename
        images (List[np.array]): the image to be saved

    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        images = [img_as_ubyte(np.clip(img, 0, 1)) for img in images]
        fname = get_filename(fname)
        imageio.mimsave(fname, images, duration=0.25)
