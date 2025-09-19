#
# test_image_tools.py: unit tests for image_tools module
#
# Copyright DeGirum Corporation 2024
# All rights reserved
#

import numpy as np
import PIL.Image
from degirum_tools.image_tools import crop_image


def test_crop_image():
    """Comprehensive test for crop_image function"""

    # Step 1: Create test images
    # Create sample OpenCV image (BGR format) - 100x80 image
    opencv_img = np.zeros((80, 100, 3), dtype=np.uint8)
    opencv_img[20:60, 25:75] = [255, 0, 0]  # Blue rectangle
    opencv_img[10:30, 10:40] = [0, 255, 0]  # Green rectangle

    # Create sample PIL image (RGB format) - 100x80 image
    pil_img = PIL.Image.new("RGB", (100, 80), color=(128, 128, 128))
    pixels = np.array(pil_img)
    pixels[20:60, 25:75] = [255, 0, 0]  # Red rectangle
    pixels[10:30, 10:40] = [0, 255, 0]  # Green rectangle
    pil_img = PIL.Image.fromarray(pixels)

    # Step 2: Test basic valid cropping
    bbox = [25, 20, 75, 60]

    # Test OpenCV cropping
    result_cv = crop_image(opencv_img, bbox)
    assert result_cv is not None
    assert isinstance(result_cv, np.ndarray)
    assert result_cv.shape == (40, 50, 3)  # height=60-20, width=75-25

    # Test PIL cropping
    result_pil = crop_image(pil_img, bbox)
    assert result_pil is not None
    assert isinstance(result_pil, PIL.Image.Image)
    assert result_pil.size == (50, 40)  # width=75-25, height=60-20

    # Step 3: Test coordinate format flexibility
    # Test reversed coordinates (TLBL format)
    bbox_reversed = [75, 60, 25, 20]
    result_reversed = crop_image(opencv_img, bbox_reversed)
    assert isinstance(result_reversed, np.ndarray)
    assert result_reversed.shape == (40, 50, 3)  # Should normalize to same size

    # Test float coordinates (should round)
    bbox_float = [25.7, 20.3, 74.9, 59.6]
    result_float = crop_image(opencv_img, bbox_float)
    assert isinstance(result_float, np.ndarray)
    assert result_float.shape == (40, 49, 3)  # Should round to [26, 20, 75, 60]

    # Step 4: Test boundary conditions
    # Test negative coordinates (should clamp to 0)
    bbox_negative = [-10, -5, 50, 40]
    result_negative = crop_image(opencv_img, bbox_negative)
    assert isinstance(result_negative, np.ndarray)
    assert result_negative.shape == (40, 50, 3)  # Should clamp to [0, 0, 50, 40]

    # Test coordinates beyond image bounds
    bbox_beyond = [50, 40, 150, 120]  # Extends beyond 100x80 image
    result_beyond = crop_image(opencv_img, bbox_beyond)
    assert isinstance(result_beyond, np.ndarray)
    assert result_beyond.shape == (40, 50, 3)  # Should clamp to [50, 40, 100, 80]

    # Step 5: Test zero-area cases
    # Test zero width bbox
    bbox_zero_width = [50, 20, 50, 60]  # x0 == x1
    result_zero_width = crop_image(opencv_img, bbox_zero_width)
    assert result_zero_width is None

    # Test zero height bbox
    bbox_zero_height = [25, 40, 75, 40]  # y0 == y1
    result_zero_height = crop_image(opencv_img, bbox_zero_height)
    assert result_zero_height is None

    # Test completely outside image bounds
    bbox_outside = [150, 100, 200, 150]  # Completely outside 100x80 image
    result_outside = crop_image(opencv_img, bbox_outside)
    assert result_outside is None  # After clamping becomes zero area

    # Step 6: Test edge cases
    # Test small valid area near edge
    bbox_small = [90, 70, 95, 75]
    result_small = crop_image(opencv_img, bbox_small)
    assert isinstance(result_small, np.ndarray)
    assert result_small.shape == (5, 5, 3)

    # Test single pixel crop
    bbox_pixel = [50, 40, 51, 41]
    result_pixel = crop_image(opencv_img, bbox_pixel)
    assert isinstance(result_pixel, np.ndarray)
    assert result_pixel.shape == (1, 1, 3)

    # Step 7: Test PIL image mode preservation
    for mode in ["RGB", "RGBA", "L", "P"]:
        if mode == "P":
            test_img = PIL.Image.new(mode, (50, 40)).convert("RGB").convert("P")
        else:
            test_img = PIL.Image.new(mode, (50, 40))

        bbox_mode = [10, 10, 30, 25]
        result_mode = crop_image(test_img, bbox_mode)
        assert result_mode is not None
        assert isinstance(result_mode, PIL.Image.Image)
        assert result_mode.mode == test_img.mode

    # Step 8: Test OpenCV dtype preservation
    for dtype in [np.uint8, np.uint16, np.float32]:
        test_img_cv = np.random.rand(50, 40, 3).astype(dtype)
        bbox_dtype = [10, 10, 30, 25]
        result_dtype = crop_image(test_img_cv, bbox_dtype)
        assert result_dtype is not None
        assert isinstance(result_dtype, np.ndarray)
        assert result_dtype.dtype == dtype

    # Step 9: Test grayscale images
    grayscale_img = np.random.rand(50, 40).astype(np.uint8)  # 2D grayscale
    bbox_gray = [10, 10, 30, 25]
    result_gray = crop_image(grayscale_img, bbox_gray)
    assert result_gray is not None
    assert isinstance(result_gray, np.ndarray)
    assert result_gray.shape == (15, 20)  # No channel dimension

    # Step 10: Test different input formats
    bbox_test = [25, 20, 75, 60]

    # Test list input
    result_list = crop_image(opencv_img, bbox_test)
    assert result_list is not None
    assert isinstance(result_list, np.ndarray)

    # Test tuple input
    bbox_tuple = (25, 20, 75, 60)
    result_tuple = crop_image(opencv_img, bbox_tuple)
    assert result_tuple is not None
    assert isinstance(result_tuple, np.ndarray)

    # Test numpy array input
    bbox_array = np.array([25, 20, 75, 60])
    result_array = crop_image(opencv_img, bbox_array)
    assert result_array is not None
    assert isinstance(result_array, np.ndarray)
