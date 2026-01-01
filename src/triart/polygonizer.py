import cv2
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

def polygonize_image(image_path, num_points=6000):
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Cannyエッジ検出
    edges = cv2.Canny(gray, 100, 200)
    edge_points = np.column_stack(np.where(edges > 0))

    # Harrisコーナー検出（輪郭の特徴点を保持）
    harris = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
    harris_threshold = 0.01 * harris.max()
    corner_points = np.column_stack(np.where(harris > harris_threshold))

    # コーナー点を優先的に保持し、残りをエッジ点からサンプリング
    num_corners = min(len(corner_points), num_points // 3)
    if len(corner_points) > num_corners:
        indices = np.random.choice(len(corner_points), num_corners, replace=False)
        corner_points = corner_points[indices]

    num_edges = num_points - num_corners
    if len(edge_points) > num_edges:
        indices = np.random.choice(len(edge_points), num_edges, replace=False)
        edge_points = edge_points[indices]

    # 画像境界に沿った点を追加（上下左右の辺）
    num_boundary = 20
    top = np.column_stack([np.zeros(num_boundary), np.linspace(0, w-1, num_boundary)])
    bottom = np.column_stack([np.full(num_boundary, h-1), np.linspace(0, w-1, num_boundary)])
    left = np.column_stack([np.linspace(0, h-1, num_boundary), np.zeros(num_boundary)])
    right = np.column_stack([np.linspace(0, h-1, num_boundary), np.full(num_boundary, w-1)])
    boundary_points = np.vstack([top, bottom, left, right]).astype(int)

    # 全ての点を結合
    points = np.vstack([edge_points, corner_points, boundary_points])
    points = np.unique(points, axis=0)  # 重複を除去

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
