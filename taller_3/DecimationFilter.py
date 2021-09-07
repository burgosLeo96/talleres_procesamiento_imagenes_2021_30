from BaseFilter import *


def decimation_fft(image, d_value):
    # Assert input data
    assert int(d_value) > 1, 'The value used for D parameter is invalid'

    d_value = int(d_value)

    # First of all, make operations over Fourier transform
    gray_image, image_fft_view, image_filtered, fft_filtered_view = compute_fourier_transform(image, d_value)

    # With the results, decimate image
    decimated_image = image_filtered[::d_value, ::d_value]

    return decimated_image
