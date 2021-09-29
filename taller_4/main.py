import sys

import cv2
import numpy as np
from numpy.linalg import LinAlgError

from Quadrilateral import Quadrilateral


def intersection(line1, line2):
    rho1 = line1[0]
    theta1 = line1[1]

    rho2 = line2[0]
    theta2 = line2[1]

    m = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])

    # solve the linear matrix equation
    try:
        x0, y0 = np.linalg.solve(m, b)
        return [int(np.round(x0)), int(np.round(y0))]
    except LinAlgError as e:
        print("Error: ", e)
        print("Lines does not intersect")
        return []


def detectCorners(quad_image):
    high_thresh = 300

    # Get Canny for input image
    bw_edges = cv2.Canny(quad_image, high_thresh * 0.3, high_thresh, L2gradient=True)

    image_draw = np.copy(quad_image)

    # Then, calculate hough lines using opencv HoughLines method
    hough_lines = cv2.HoughLines(bw_edges, 1, np.pi / 180, round(0.25 * image_draw.shape[0]))

    cv2.imshow("BW_edges", bw_edges)
    cv2.waitKey(0)

    # Paint the hough lines
    for line in hough_lines:
        for rho, theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 3000 * (-b))
            y1 = int(y0 + 3000 * a)
            x2 = int(x0 - 3000 * (-b))
            y2 = int(y0 - 3000 * a)
            cv2.line(image_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # After that, find intersection between hough lines
    intersections = []
    hough_lines = hough_lines[:, -1]
    for i, line_1 in enumerate(hough_lines):
        for line_2 in hough_lines[i + 1:, :]:
            intersections.append(intersection(line_1, line_2))

    max_x = quad_image.shape[0]
    max_y = quad_image.shape[1]

    # Remove empty values
    intersections[:] = [inter for inter in intersections if len(inter) > 0]

    # Remove values outside the image
    intersections[:] = [inter for inter in intersections if (max_x > inter[0] >= 0) or (max_y > inter[1] >= 0)]

    for point in intersections:
        cv2.circle(quad_image, (point[0], point[1]), 5, (0, 255, 255), -1)

    cv2.imshow("Image draw", image_draw)
    cv2.waitKey(0)

    return quad_image, intersections


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: ", sys.argv[0], " image_size")
        exit(1)

    image_size = int(sys.argv[1])

    # Assert image size is even
    assert image_size % 2 == 0, 'image_size have to be even'

    # Initialize quadrilateral object
    quadrilateral = Quadrilateral(image_size)
    quadrilateral_image = quadrilateral.generate()
    image, intersection_list = detectCorners(quadrilateral_image)

    cv2.imshow("Result image", image)
    cv2.waitKey(0)
