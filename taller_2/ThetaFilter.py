import cv2
import numpy as np


def process_limits(lower_bound, upper_bound, angles_matrix):
    if upper_bound < lower_bound:
        # Split limits in two: lower_bound to 360 and 0 to upper_bound
        lower_index_0 = angles_matrix > lower_bound
        upper_index_0 = angles_matrix <= 360

        lower_index_1 = angles_matrix >= 0
        upper_index_1 = angles_matrix < upper_bound

        idx_bitwise_0 = np.bitwise_and(lower_index_0, upper_index_0)
        idx_bitwise_1 = np.bitwise_and(lower_index_1, upper_index_1)

        return np.bitwise_or(idx_bitwise_0, idx_bitwise_1)

    else:
        lower_index = angles_matrix > lower_bound
        upper_index = angles_matrix < upper_bound

        return np.bitwise_and(lower_index, upper_index)


class ThetaFilter:

    def __init__(self, image):
        self.image = image
        self.theta = None
        self.delta = None

    def set_theta(self, theta, delta):
        self.theta = theta
        self.delta = delta

    def filtering(self, show_fourier_images=False):
        image_fft = np.fft.fft2(self.image)
        image_gray_fft_shift = np.fft.fftshift(image_fft)

        image_gray_fft_mag = np.absolute(image_gray_fft_shift)
        image_fft_view = np.log(image_gray_fft_mag + 1)
        image_fft_view = image_fft_view / np.max(image_fft_view)

        num_rows, num_cols = (self.image.shape[0], self.image.shape[1])
        enum_rows = np.linspace(0, num_rows - 1, num_rows)
        enum_cols = np.linspace(0, num_cols - 1, num_cols)
        col_iter, row_iter = np.meshgrid(enum_cols, enum_rows)
        half_size = num_rows / 2  # here we assume num_rows = num_columns

        # get the matrix of angles
        angles_matrix = np.arctan2((col_iter - half_size), (row_iter - half_size)) * 180 / np.pi
        angles_matrix = np.where(angles_matrix < 0, angles_matrix + 360, angles_matrix)

        angle_pass_mask = np.zeros_like(self.image)

        # First of all, find lower and upper bounds
        lower_bound_0 = (self.theta - self.delta) % 360
        upper_bound_0 = (self.theta + self.delta) % 360

        # If any of the limits are negative, add 360
        if lower_bound_0 < 0:
            lower_bound_0 = (lower_bound_0 + 360) % 360

        if upper_bound_0 < 0:
            upper_bound_0 = (upper_bound_0 + 360) % 360

        angle_pass_mask[process_limits(lower_bound_0, upper_bound_0, angles_matrix)] = 1

        # After getting lower and upper bound, get the other ones that belong to the other quadrant
        lower_bound_1 = (lower_bound_0 + 180) % 360
        upper_bound_1 = (upper_bound_0 + 180) % 360

        angle_pass_mask[process_limits(lower_bound_1, upper_bound_1, angles_matrix)] = 1
        angle_pass_mask[int(half_size), int(half_size)] = 1

        fft_filtered_shift = image_gray_fft_shift * angle_pass_mask
        fft_filtered_mag = np.absolute(fft_filtered_shift)
        fft_filtered_view = np.log(fft_filtered_mag + 1)
        fft_filtered_view = fft_filtered_view / np.max(fft_filtered_view)

        image_filtered = np.fft.ifft2(np.fft.fftshift(fft_filtered_shift))
        image_filtered = np.absolute(image_filtered)
        image_filtered /= np.max(image_filtered)

        # Original image
        cv2.imshow('Original image', self.image)

        if show_fourier_images:
            # Original fourier transform
            cv2.imshow('Fourier transform', image_fft_view)

            # Filtered fourier transform
            cv2.imshow('Filtered Fourier transform', fft_filtered_view)

        # Filtered image
        cv2.imshow("Filtered Image {} angle".format(self.theta), image_filtered)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return image_filtered

    def filters_bank(self):
        self.delta = 20
        filter_bank = [0, 45, 90, 135]

        processed_images = []

        for i in filter_bank:
            self.theta = i
            processed_images.append(self.filtering())

        numpy_cube = np.array(processed_images)
        average_image = numpy_cube.mean(axis=0)

        cv2.imshow("Average Image", average_image)
        cv2.waitKey(0)
        
