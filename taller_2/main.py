
import sys
import cv2
from ThetaFilter import *

if __name__ == '__main__':

    if len(sys.argv) < 4:
        print('Usage: %s image_path theta delta' % sys.argv[0])
        sys.exit(1)

    image_path = sys.argv[1]
    theta = float(sys.argv[2])
    delta = float(sys.argv[3])

    assert 0 <= theta <= 360
    assert 0 <= delta <= 360

    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Point 1
    fftFilter = ThetaFilter(gray_image)
    fftFilter.set_theta(theta, delta)
    fftFilter.filtering(True)

    # Point 2
    filter_bank = [0, 45, 90, 135]

    for i in filter_bank:
        fftFilter.set_theta(i, 5)
        fftFilter.filtering(True)
