# --------------------------------------------------------------------------------------------------
#  Copyright (c) 2018 Microsoft Corporation
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
#  associated documentation files (the "Software"), to deal in the Software without restriction,
#  including without limitation the rights to use, copy, modify, merge, publish, distribute,
#  sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or
#  substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
#  NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
#  NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#  DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# --------------------------------------------------------------------------------------------------

"""
Image processing utilities. This module will use either cv2 or PIL as its backend. If neither of
these is available, no image processing will be provided and importing this module will cause an
exception.
"""

import logging
import numpy as np

OPENCV_AVAILABLE = False
PILLOW_AVAILABLE = False

try:
    import cv2

    OPENCV_AVAILABLE = True
    logging.info('OpenCV found, setting as default backend.')
except ImportError:
    pass

try:
    import PIL

    PILLOW_AVAILABLE = True

    if not OPENCV_AVAILABLE:
        logging.info('Pillow found, setting as default backend.')
except ImportError:
    pass


if not (OPENCV_AVAILABLE or PILLOW_AVAILABLE):
    raise ValueError('No image library backend found.'' Install either '
                     'OpenCV or Pillow to support image processing.')


def resize(img, shape):
    """Resize the specified image.

    Args:
        img -- Image to reshape
        shape -- New image shape

    Returns:
        the resized image
    """
    if OPENCV_AVAILABLE:
        return cv2.resize(img, shape, interpolation=cv2.INTER_AREA)

    if PILLOW_AVAILABLE:
        return np.array(PIL.Image.fromarray(img).resize(shape))

    raise NotImplementedError

def rgb2gray(rgb, mode='luminosity'):
    """Convert a numpy array of RGB values to grayscale.

    Args:
        rgb -- a HWC numpy array

    Keyword Args
        mode -- one of ['lightness', 'average', 'luminosity']
    """

    if OPENCV_AVAILABLE:
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    if mode == 'lightness':
        max_channels = np.amax(rgb, axis=-1)
        min_channels = np.amin(rgb, axis=-1)
        return ((max_channels + min_channels) / 2).astype(rgb.dtype)

    if mode == 'average':
        return np.mean(rgb, axis=-1).astype(rgb.dtype)

    if mode == 'luminosity':
        return np.dot(rgb, [0.21, 0.72, 0.07]).astype(rgb.dtype)

    raise NotImplementedError


def readgray(path, shape=(84, 84)):
    """Read image in greyscale.

    Args:
        path -- string, path to the image
        shape -- tuple of 2 integers, shape of the output image

    Returns:
        ndarray of shape (1, shape[0], shape[1])
    """
    if OPENCV_AVAILABLE:
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), shape)
    elif PILLOW_AVAILABLE:
        img = rgb2gray(np.array(PIL.Image.open(path).resize(shape)))
    else:
        raise NotImplementedError

    if img.dtype != np.float32:
        img = img.astype(np.float32)
    # normalize
    if img.max() > 1:
        img *= 1. / 255.
    return img