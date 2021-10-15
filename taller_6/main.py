import os
import re
import sys
import cv2
import numpy as np


# main method
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

    # Capture interest points detection technique
    interest_points_tec = input("Please choose the technique to identify the interest points (SIFT/ORB): ")

    assert interest_points_tec == "SIFT" or interest_points_tec == "ORB", "The technique specified is invalid"

    # Initialize helper list to store the chosen points for each image
    images_points_list = []

    # Sort images list by numbers
    images_list.sort(key=lambda image_name: int(re.search(r'[0-9]+', image_name).group()))

    cv2_images_list = list(map(lambda image_name: cv2.imread(images_path + "/" + image_name), images_list))

    # In this point, start to identify the interest points for each pair of images

    # Iterate over the images list to read the points
    for i in range(1, len(cv2_images_list)):

        # Read a pair of images
        left_image = cv2_images_list[i - 1]
        right_image = cv2_images_list[i]

        # Compute interest points using SIFT technique
        if interest_points_tec == 'SIFT':
            sift = cv2.SIFT_create(nfeatures=100)
            bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
            kp_1, left_image_desc = sift.detectAndCompute(cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY), None)
            kp_2, right_image_desc = sift.detectAndCompute(cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY), None)

        # Compute interest points using ORB technique
        else:
            orb = cv2.ORB_create(nfeatures=100)
            bf = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)
            kp_1, left_image_desc = orb.detectAndCompute(cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY), None)
            kp_2, right_image_desc = orb.detectAndCompute(cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY), None)

        # Then, find matching key points
        matched_points = bf.knnMatch(left_image_desc, right_image_desc, k=1)
        matches_image = cv2.drawMatchesKnn(left_image, kp_1, right_image, kp_2, matched_points, None)

        # Then, retrieve matched points and store them
        left_image_points = []
        right_image_points = []
        for source_index, match in enumerate(matched_points):
            if len(match) > 0:
                target_index = match[0].trainIdx
                left_image_points.append(kp_1[source_index].pt)
                right_image_points.append(kp_2[target_index].pt)

        images_points_list.append(left_image_points)
        images_points_list.append(right_image_points)

    homography_list = []
    # Calculate homography matrix per each couple of points
    indexes = np.arange(0, len(images_points_list), 2)
    for i in indexes:
        # Select the minimum quantity of points between the pair of points_list that are going to be processed
        N = min(len(images_points_list[i]), len(images_points_list[i + 1]))
        source_points = np.array(images_points_list[i][:N])
        target_points = np.array(images_points_list[i + 1][:N])

        # Compute the homography between the two points_lists
        homography, _ = cv2.findHomography(source_points, target_points, method=cv2.RANSAC)

        # Append homography matrix to homography_list
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
        # Default scenario: image index is equal to current index
        res_image = cv2_images_list[i]

        # Scenario 2: image index is less than the reference image index
        if i < reference_image_index:
            matrix_accum = homography_list[i]
            for j in range(i + 1, reference_image_index):
                matrix_accum = np.matmul(matrix_accum, homography_list[j])

            res_image = cv2.warpPerspective(res_image, matrix_accum, (res_image.shape[1], res_image.shape[0]))

        # Scenario 3: image index is greater that the reference image index
        elif i > reference_image_index:
            matrix_accum = homography_list[reference_image_index]

            for j in range(reference_image_index + 1, i):
                matrix_accum = np.matmul(matrix_accum, homography_list[j])

            inverse_homography = np.linalg.inv(matrix_accum)
            res_image = cv2.warpPerspective(res_image, inverse_homography, (res_image.shape[1], res_image.shape[0]))

        # Append result image to result_images list
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
    cv2.imwrite(images_path + "/result_image.jpg", perspective_result)
    print("Output file --result_image.jpg-- created.")

    result = cv2.imread(images_path + "/result_image.jpg")

    # Show result image through "Result" window
    cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
    cv2.imshow("Result", result)
    cv2.waitKey(0)
