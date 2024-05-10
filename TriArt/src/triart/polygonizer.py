import cv2
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt


def polygonize_image(
    image_path, edge_threshold1=100, edge_threshold2=200, num_random_points=500
):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, edge_threshold1, edge_threshold2)
    points = np.column_stack(np.where(edges > 0))

    # 追加のランダムポイント（数を減らす）
    additional_points = np.random.randint(
        0, min(image.shape[:2]), (num_random_points, 2)
    )
    points = np.vstack([points, additional_points])

    tri = Delaunay(points)
    img_polygonized = np.zeros_like(image)

    for simplex in tri.simplices:
        pts = points[simplex]
        rect = cv2.boundingRect(np.array([pts]))
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, pts, 255)
        mean_color = cv2.mean(image, mask=mask)
        cv2.fillConvexPoly(img_polygonized, pts, mean_color[:3])

    return img_polygonized


def enhanced_polygonize_image(
    image_path, edge_threshold1=150, edge_threshold2=250, num_random_points=300
):
    image = cv2.imread(image_path)
    blurred_image = cv2.GaussianBlur(image, (7, 7), 0)  # ぼかしを加える
    gray = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, edge_threshold1, edge_threshold2)
    points = np.column_stack(np.where(edges > 0))

    # 追加のランダムポイント
    additional_points = np.random.randint(
        0, min(image.shape[:2]), (num_random_points, 2)
    )
    points = np.vstack([points, additional_points])

    tri = Delaunay(points)
    img_polygonized = np.zeros_like(image)

    for simplex in tri.simplices:
        pts = points[simplex]
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, pts, 255)
        mean_color = cv2.mean(image, mask=mask)
        cv2.fillConvexPoly(img_polygonized, pts, mean_color[:3])

    return img_polygonized


def enhanced_polygonize_image2(
    image_path, edge_threshold1=75, edge_threshold2=150, num_random_points=800
):
    image = cv2.imread(image_path)
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)  # ぼかしを少し減らす
    gray = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, edge_threshold1, edge_threshold2)
    points = np.column_stack(np.where(edges > 0))

    # 追加のランダムポイントの数を調整
    additional_points = np.random.randint(
        0, min(image.shape[:2]), (num_random_points, 2)
    )
    points = np.vstack([points, additional_points])

    tri = Delaunay(points)
    img_polygonized = np.zeros_like(image)

    for simplex in tri.simplices:
        pts = points[simplex]
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, pts, 255)
        mean_color = cv2.mean(image, mask=mask)
        cv2.fillConvexPoly(img_polygonized, pts, mean_color[:3])

    return img_polygonized


def enhanced_polygonize_image3(
    image_path, edge_threshold1=150, edge_threshold2=250, num_random_points=1200
):
    image = cv2.imread(image_path)
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)  # 画像を適度にぼかす
    gray = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, edge_threshold1, edge_threshold2)
    points = np.column_stack(np.where(edges > 0))

    # 追加のランダムポイントを増やしてポリゴンを細かくする
    additional_points = np.random.randint(
        0, min(image.shape[:2]), (num_random_points, 2)
    )
    points = np.vstack([points, additional_points])

    tri = Delaunay(points)
    img_polygonized = np.zeros_like(image)

    for simplex in tri.simplices:
        pts = points[simplex]
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, pts, 255)
        mean_color = cv2.mean(image, mask=mask)
        cv2.fillConvexPoly(img_polygonized, pts, mean_color[:3])

    return img_polygonized


def save_image(image, path):
    cv2.imwrite(path, image)
