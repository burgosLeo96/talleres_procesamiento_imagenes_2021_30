from BaseFilter import *

import numpy as np


def interpolation_fft(image, i_value):
    # Assert input data
    assert int(i_value) > 1, 'The value used for I parameter is invalid'

    i_value = int(i_value) - 1

    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    num_rows, num_columns = gray_image.shape

    zeros_image = np.zeros((i_value * num_rows, i_value * num_columns), dtype=gray_image.dtype)
    zeros_image[::i_value, ::i_value] = gray_image
    window_size = 2 * i_value + 1

    interpolated_image = cv2.GaussianBlur(zeros_image, (window_size, window_size), 1.0)
    interpolated_image *= zeros_image ** 2

    # First of all, make operations over Fourier transform
    gray_image, image_fft_view, image_filtered, fft_filtered_view = compute_fourier_transform(interpolated_image, i_value)

    return image_filtered
