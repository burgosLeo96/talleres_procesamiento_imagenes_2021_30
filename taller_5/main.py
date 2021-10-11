import os
import re
import sys
import cv2
import numpy as np

points_list = []


# Helper method to capture the clicked pixel coordinates
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points_list.append((x, y))


if __name__ == '__main__':
    # Validate the argv length
    if len(sys.argv) < 2:
        print("Usage: python ", sys.argv[0], " images_directory")
        exit(1)

    # Get the initial data
    images_path = sys.argv[1]
    images_list = os.listdir(images_path)
    image_cant = len(images_list)

    # Assert that the directory received is not empty
    assert image_cant > 0, "The directory received is empty."
    print(image_cant, " images founded in the directory.")

    # Capture reference image index
    reference_image_index = int(input("Please specify the index of the image that is going to be used as reference: "))

    # Validate that the reference image index
    assert 1 <= reference_image_index <= image_cant, "The reference image index must be between 1 and the images cant"

    # Initialize helper list to store the chosen points for each image
    images_points_list = []

    # Create visualization window
    cv2.namedWindow("Images", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Images", click_event)

    # Sort images list by numbers
    images_list.sort(key=lambda image_name: int(re.search(r'[0-9]+', image_name).group()))

    cv2_images_list = list(map(lambda image_name: cv2.imread(images_path + "/" + image_name), images_list))

    # Iterate over the images list to read the points
    for i in range(1, len(cv2_images_list)):

        # Read a pair of images
        left_image = cv2_images_list[i - 1]
        right_image = cv2_images_list[i]

        # Concatenate horizontally the images
        concatenated_images = cv2.hconcat([left_image, right_image])

        offset = left_image.shape[1]

        rendered_points = 0
        while True:
            cv2.imshow("Images", concatenated_images)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('x'):
                if len(points_list) < 8:
                    print("Please select at least 4 points per image.")
                else:
                    left_image_points = points_list[0::2]
                    right_image_points = points_list[1::2]
                    right_image_points = list(
                        map(lambda coordinates: (coordinates[0] - offset, coordinates[1]), right_image_points))
                    images_points_list.append(left_image_points)
                    images_points_list.append(right_image_points)
                    points_list = []
                    break
            if len(points_list) > rendered_points:
                rendered_points = len(points_list)

                # If the points list length is odd, then the last points belongs to left image
                if len(points_list) % 2 != 0:
                    cv2.circle(concatenated_images, (points_list[-1][0], points_list[-1][1]), 3, [0, 0, 255], -1)

                # If the points list length is even, then the last points belongs to right image
                else:
                    cv2.circle(concatenated_images, (points_list[-1][0], points_list[-1][1]), 3, [255, 0, 0], -1)

    cv2.destroyWindow("Images")
    homography_list = []

    # Calculate homography matrix per each couple of points
    indexes = np.arange(0, len(images_points_list), 2)
    for i in indexes:
        N = min(len(images_points_list[i]), len(images_points_list[i + 1]))
        source_points = np.array(images_points_list[i][:N])
        target_points = np.array(images_points_list[i + 1][:N])

        homography, _ = cv2.findHomography(source_points, target_points, method=cv2.RANSAC)
        homography_list.append(homography)

    # Steps to calculate images perspectives:
    #  1. iterate over all the images
    #  2. analise the image index: three scenarios:
    #     a. image index is equal to reference image index -> add image to result list
    #     b. image index is less than reference image -> iterate from index to (reference index - 1) accumulating
    #        homography multiplications
    #     c. image index is greater that reference image -> iterate from (reference index) to index - 1 accumulating
    #        homography multiplications. At the end of the cycle, compute inverse
    #  4. Compute wrapPerspective
    #  5. Add image to result list
    result_images = []
    reference_image_index -= 1
    for i in range(len(cv2_images_list)):
        res_image = cv2_images_list[i]

        if i < reference_image_index:
            matrix_accum = homography_list[i]
            for j in range(i + 1, reference_image_index):
                matrix_accum = np.matmul(matrix_accum, homography_list[j])

            res_image = cv2.warpPerspective(res_image, matrix_accum, (res_image.shape[1], res_image.shape[0]))

        elif i > reference_image_index:
            matrix_accum = homography_list[reference_image_index]

            for j in range(reference_image_index + 1, i):
                matrix_accum = np.matmul(matrix_accum, homography_list[j])

            inverse_homography = np.linalg.inv(matrix_accum)
            res_image = cv2.warpPerspective(res_image, inverse_homography, (res_image.shape[1], res_image.shape[0]))

        result_images.append(res_image)

    # Finally, compute average
    # Convert results to numpy array
    np_res_images = np.array(result_images, dtype=float)

    # Convert zero values to np.nan
    np_res_images[np_res_images == 0] = np.nan

    # Compute average ignoring nan values across all images in BGR channels
    perspective_result = np.nanmean(np_res_images, axis=0)

    # Round values to int
    perspective_result = np.rint(perspective_result)

    # Save results
    cv2.imwrite("result_image.jpg", perspective_result)
    print("Output file --result_image.jpg-- created.")

    result = cv2.imread("result_image.jpg")

    cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
    cv2.imshow("Result", result)
    cv2.waitKey(0)
