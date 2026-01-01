import cv2
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

def polygonize_image(image_path, num_points=6000):
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    # エッジ点を取得してサンプリング
    edge_points = np.column_stack(np.where(edges > 0))
    if len(edge_points) > num_points:
        indices = np.random.choice(len(edge_points), num_points, replace=False)
        edge_points = edge_points[indices]

    # 四隅と境界点を追加
    corners = np.array([[0, 0], [0, w-1], [h-1, 0], [h-1, w-1]])
    points = np.vstack([edge_points, corners])

    tri = Delaunay(points)
    img_polygonized = np.zeros_like(image)

    for simplex in tri.simplices:
        pts_yx = points[simplex]  # (y, x) format from np.where
        pts_xy = pts_yx[:, ::-1]  # convert to (x, y) for OpenCV
        x, y, w, h = cv2.boundingRect(pts_xy)
        mask = np.zeros((h, w), dtype=np.uint8)
        pts_shifted = pts_xy - np.array([[x, y]])
        cv2.fillConvexPoly(mask, pts_shifted, 255)
        roi = image[y:y+h, x:x+w]
        mean_color = cv2.mean(roi, mask=mask)
        cv2.fillConvexPoly(img_polygonized, pts_xy, mean_color[:3])

    return img_polygonized

def save_image(image, path):
    cv2.imwrite(path, image)
