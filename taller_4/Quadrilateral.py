import cv2
import random
import numpy as np


class Quadrilateral:

    image_size = None

    def __init__(self, image_size):
        self.image_size = image_size

    def generate(self):

        # First of all, create cyan image background
        image = np.zeros((self.image_size, self.image_size, 3), np.uint8)
        image[:, :, 0] = 255
        image[:, :, 1] = 255

        p1_x = random.randint(0, (self.image_size / 2) - 1)
        p1_y = random.randint(0, (self.image_size / 2) - 1)

        p2_x = random.randint((self.image_size / 2) - 1, self.image_size - 1)
        p2_y = random.randint(0, (self.image_size / 2) - 1)

        p3_x = random.randint((self.image_size / 2) - 1, self.image_size - 1)
        p3_y = random.randint((self.image_size / 2) - 1, self.image_size - 1)

        p4_x = random.randint(0, (self.image_size / 2) - 1)
        p4_y = random.randint((self.image_size / 2) - 1, self.image_size - 1)

        # Create edges between each point pair
        cv2.line(image, (p1_x, p1_y), (p2_x, p2_y), (255, 0, 255), 1)
        cv2.line(image, (p2_x, p2_y), (p3_x, p3_y), (255, 0, 255), 1)
        cv2.line(image, (p3_x, p3_y), (p4_x, p4_y), (255, 0, 255), 1)
        cv2.line(image, (p4_x, p4_y), (p1_x, p1_y), (255, 0, 255), 1)

        cv2.imshow('Image', image)
        cv2.waitKey(0)

        return image
