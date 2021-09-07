import sys
import cv2

from DecompositionFilter import *
from InterpolationFilter import *

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: ", sys.argv[0], " image_path decomposition_order")
        exit(1)

    assert int(sys.argv[2]) >= 0, "Decomposition order cannot be negative"

    image_path = sys.argv[1]
    decomposition_order = int(sys.argv[2])

    image = cv2.imread(image_path)
    print("Input image size: ", image.shape)
    result_images = decomposition_bank_filter(image, decomposition_order)

    for i in range(0, len(result_images)):
        index = 1
        for res_image in result_images[i]:
            cv2.imshow("Image {}".format(index), res_image)
            index += 1
        cv2.waitKey(0)

    ill_interpolated = interpolation_fft(result_images[1][0], 5)
    print("ILL interpolated image size: ", ill_interpolated.shape)
    cv2.imshow("Original image", image)
    cv2.imshow("ILL Interpolated", ill_interpolated)
    cv2.waitKey(0)
