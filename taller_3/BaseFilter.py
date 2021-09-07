import numpy as np
import cv2


def compute_fourier_transform(image, filter_value):
    # First of all, convert image to grayscale
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    # Compute Fourier transform
    image_fft = np.fft.fft2(gray_image)
    image_gray_fft_shift = np.fft.fftshift(image_fft)

    # Process fourier transform to visualize it
    image_gray_fft_mag = np.absolute(image_gray_fft_shift)
    image_fft_view = np.log(image_gray_fft_mag + 1)
    image_fft_view = image_fft_view / np.max(image_fft_view)

    # Get index for each row and column
    num_rows, num_cols = (gray_image.shape[0], gray_image.shape[1])
    enum_rows = np.linspace(0, num_rows - 1, num_rows)
    enum_cols = np.linspace(0, num_cols - 1, num_cols)
    col_iter, row_iter = np.meshgrid(enum_cols, enum_rows)

    # Get half-point index
    half_size = num_rows / 2

    # Compute frequency_cutoff and filter the indexes
    low_frequency_mask = np.zeros_like(gray_image)
    frequency_cutoff = int(half_size * (1 / filter_value))
    filtered_indexes = np.sqrt((col_iter - half_size) ** 2 + (row_iter - half_size) ** 2) < frequency_cutoff
    low_frequency_mask[filtered_indexes] = 1

    # Apply the filter over the Fourier transform
    fft_filtered_shift = image_gray_fft_shift * low_frequency_mask

    # Process filterer fourier transform to visualize it
    fft_filtered_mag = np.absolute(fft_filtered_shift)
    fft_filtered_view = np.log(fft_filtered_mag + 1)
    fft_filtered_view = fft_filtered_view / np.max(fft_filtered_view)

    # Compute inverse fourier transform to retrieve processed image
    image_filtered = np.fft.ifft2(np.fft.fftshift(fft_filtered_shift))
    image_filtered = np.absolute(image_filtered)
    image_filtered /= np.max(image_filtered)

    return gray_image, image_fft_view, image_filtered, fft_filtered_view
