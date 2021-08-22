import sys
import cv2

from BasicColor import BasicColor


if __name__ == '__main__':

    image_path = sys.argv[1]
    h = sys.argv[2]

    basicColor = BasicColor(image_path)
    cv2.imshow('Original Image', basicColor.image)
    cv2.waitKey(0)

    image_properties = basicColor.display_properties()
    print('Image size: %2d' % (image_properties['image_size']))
    print('Image channels: %2d' % (image_properties['image_channels']))

    cv2.imshow('Black-White Image', basicColor.make_bw())
    cv2.waitKey(0)

    cv2.imshow('Colorized Image', basicColor.colorize(int(h)))
    cv2.waitKey(0)
