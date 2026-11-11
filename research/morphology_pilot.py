"""Image processing variants used by the RQ2 recovery experiment."""
import cv2
import numpy as np


def variants(image):
    rgb = np.asarray(image)
    kernel = np.ones((3, 3), np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    overlay = cv2.addWeighted(rgb, 1.0, cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB), 0.25, 0)
    return {'erosie': cv2.erode(rgb, kernel), 'dilatatie': cv2.dilate(rgb, kernel),
            'canny_overlay': overlay}
