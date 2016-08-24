from aetros.utils.image import upscale, resize_image
import numpy as np
import pytest
from PIL import Image


def test_upscale():
    test_image = np.zeros((100, 100, 3))
    scaled_image = upscale(test_image, 1.1)

    assert scaled_image.shape[0] == 110
    assert scaled_image.shape[1] == 110
    assert scaled_image.shape[2] == 3

    with pytest.raises(ValueError):
        upscale(test_image, 0.9)

    with pytest.raises(ValueError):
        upscale(42, 1.2)


def test_resize_image():
    test_image = Image.fromarray(np.uint8(np.zeros((100, 100, 3))))
    resized = resize_image(test_image, 42, 31)
    assert resized.shape[0] == 42
    assert resized.shape[1] == 31
