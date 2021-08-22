import cv2
import numpy as np


class BasicColor:

    def __init__(self, path):
        self.image = cv2.imread(path)

    def display_properties(self):
        return {
            'image_size': self.image.size,
            'image_channels': np.array(self.image[1, 1, :]).size
        }

    def make_bw(self):
        image_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        ret, ibw = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return ibw

    def colorize(self, h):
        if 0 <= h <= 179:
            image_hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
            hue, s, v = cv2.split(image_hsv)
            hue = h * np.ones_like(hue)
            colorized_image = cv2.merge((hue, s, v))
            bgr_colorized_image = cv2.cvtColor(colorized_image, cv2.COLOR_HSV2RGB)
            return bgr_colorized_image
        else:
            return 'invalid input'
