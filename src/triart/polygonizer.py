import cv2
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

def polygonize_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    points = np.column_stack(np.where(edges > 0))
    tri = Delaunay(points)
    img_polygonized = np.zeros_like(image)

    for simplex in tri.simplices:
        pts = points[simplex]
        rect = cv2.boundingRect(np.array([pts]))
        mask = np.zeros((rect[3], rect[2]), dtype=np.uint8)
        pts_shifted = pts - np.array([rect[:2]])
        cv2.fillConvexPoly(mask, pts_shifted, 255)
        mean_color = cv2.mean(image, mask=mask)
        cv2.fillConvexPoly(img_polygonized, pts, mean_color[:3])

    return img_polygonized

def save_image(image, path):
    cv2.imwrite(path, image)
