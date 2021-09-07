from DecimationFilter import *

low_pass_filter = np.array([[1/16, 1/8, 1/16], [1/8, 1/4, 1/8], [1/16, 1/8, 1/16]])
horizontal_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
vertical_filter = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
diagonal_filter = np.array([[2, -1, -2], [-1, 4, -1], [-2, -1, 2]])

filters_bank = [low_pass_filter, horizontal_filter, vertical_filter, diagonal_filter]


def decomposition_bank_filter(image, n_value):

    if n_value == 0:
        return [[image]]

    result = []

    input_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    for i in range(0, n_value):
        # Create temp list to store images of the current decomposition order
        iteration_result = []

        # Iterate over filters_bank
        for kernel in filters_bank:
            # Apply each kernel to the input image
            convolution_image = cv2.filter2D(input_image, -1, kernel)

            # Multiply input image with convolution image
            temp_result = input_image * convolution_image

            # Decimate the temp_result image
            result_image = decimation_fft(temp_result, 2)

            iteration_result.append(result_image)

        # After iterating over the filters_bank, append temp list to result list
        result.append(iteration_result)

        # Update the input image that is going to be used in the next decomposition order
        input_image = iteration_result[0]

    return result
