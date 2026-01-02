import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import Delaunay, KDTree


def poisson_disk_sampling(width, height, min_dist, fixed_points=None, k=30):
    """
    Bridson's algorithm for Poisson disk sampling.
    Generates points with minimum distance constraint.

    Fixed points are ALWAYS preserved (never removed).
    New points are added only where they don't violate min_dist from other NEW points.

    Args:
        width: Image width
        height: Image height
        min_dist: Minimum distance between points
        fixed_points: Points that must be preserved (y, x format) - e.g., edge/corner points
        k: Number of attempts before rejecting a point
    """
    cell_size = min_dist / np.sqrt(2)
    grid_width = int(np.ceil(width / cell_size))
    grid_height = int(np.ceil(height / cell_size))

    # Grid stores index of point in that cell (-1 = empty)
    grid = np.full((grid_height, grid_width), -1, dtype=int)
    points = []  # New points only (for grid-based collision)
    active_list = []

    # Fixed points are stored separately and always preserved
    fixed_list = []
    if fixed_points is not None and len(fixed_points) > 0:
        fixed_list = [(pt[0], pt[1]) for pt in fixed_points
                      if 0 <= pt[0] < height and 0 <= pt[1] < width]

    # Build KDTree for fixed points to check distance efficiently
    fixed_tree = None
    if len(fixed_list) > 0:
        fixed_tree = KDTree(np.array(fixed_list))

    def grid_coords(point):
        """Convert point (y, x) to grid coordinates."""
        return int(point[0] / cell_size), int(point[1] / cell_size)

    def is_valid(point):
        """Check if point is valid (within bounds and respects min_dist from NEW points only)."""
        y, x = point
        if y < 0 or y >= height or x < 0 or x >= width:
            return False

        # Check against fixed points - use smaller threshold to allow points near edges
        if fixed_tree is not None:
            dist, _ = fixed_tree.query([y, x])
            if dist < min_dist * 0.5:  # Allow closer to edges for better detail
                return False

        gy, gx = grid_coords(point)
        # Check neighboring cells (5x5 neighborhood) for NEW points only
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                ny, nx = gy + dy, gx + dx
                if 0 <= ny < grid_height and 0 <= nx < grid_width:
                    idx = grid[ny, nx]
                    if idx != -1:
                        other = points[idx]
                        dist = np.sqrt((point[0] - other[0])**2 + (point[1] - other[1])**2)
                        if dist < min_dist:
                            return False
        return True

    def add_point(point):
        """Add a NEW point to the sampling."""
        points.append(point)
        idx = len(points) - 1
        gy, gx = grid_coords(point)
        if 0 <= gy < grid_height and 0 <= gx < grid_width:
            grid[gy, gx] = idx
        return idx

    # Start with a random point
    for _ in range(100):  # Try to find a valid starting point
        initial = (np.random.uniform(0, height), np.random.uniform(0, width))
        if is_valid(initial):
            add_point(initial)
            active_list.append(0)
            break

    while active_list:
        idx = np.random.randint(len(active_list))
        point = points[active_list[idx]]

        found = False
        for _ in range(k):
            # Generate random point in annulus [min_dist, 2*min_dist]
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(min_dist, 2 * min_dist)
            new_point = (
                point[0] + radius * np.cos(angle),
                point[1] + radius * np.sin(angle)
            )

            if is_valid(new_point):
                new_idx = add_point(new_point)
                active_list.append(new_idx)
                found = True
                break

        if not found:
            active_list.pop(idx)

    # Combine fixed points (preserved) with new points
    all_points = fixed_list + points
    return np.array(all_points)


def adaptive_poisson_sampling(width, height, edge_map, base_min_dist, fixed_points=None, k=30):
    """
    Adaptive Poisson disk sampling with variable min_dist based on edge distance.

    Near edges: smaller min_dist (denser points)
    Far from edges: larger min_dist (sparser points)

    Args:
        width: Image width
        height: Image height
        edge_map: Binary edge map (255 = edge, 0 = background)
        base_min_dist: Base minimum distance (used far from edges)
        fixed_points: Points that must be preserved (y, x format)
        k: Number of attempts before rejecting a point
    """
    # Compute distance transform from edges
    edge_binary = (edge_map > 0).astype(np.uint8)
    dist_from_edge = distance_transform_edt(1 - edge_binary)

    # Normalize distance to [0, 1]
    max_dist = dist_from_edge.max() if dist_from_edge.max() > 0 else 1
    dist_normalized = dist_from_edge / max_dist

    def get_min_dist_at(y, x):
        """Get adaptive min_dist at position (y, x)."""
        if 0 <= int(y) < height and 0 <= int(x) < width:
            # Near edge (dist=0): min_dist = base * 0.3
            # Far from edge (dist=1): min_dist = base * 1.5
            factor = 0.3 + 1.2 * dist_normalized[int(y), int(x)]
            return base_min_dist * factor
        return base_min_dist

    # Use smaller cell size for grid (based on minimum possible min_dist)
    min_cell_dist = base_min_dist * 0.3
    cell_size = min_cell_dist / np.sqrt(2)
    grid_width = int(np.ceil(width / cell_size))
    grid_height = int(np.ceil(height / cell_size))

    grid = np.full((grid_height, grid_width), -1, dtype=int)
    points = []
    active_list = []

    fixed_list = []
    if fixed_points is not None and len(fixed_points) > 0:
        fixed_list = [(pt[0], pt[1]) for pt in fixed_points
                      if 0 <= pt[0] < height and 0 <= pt[1] < width]

    fixed_tree = None
    if len(fixed_list) > 0:
        fixed_tree = KDTree(np.array(fixed_list))

    def grid_coords(point):
        return int(point[0] / cell_size), int(point[1] / cell_size)

    def is_valid(point):
        y, x = point
        if y < 0 or y >= height or x < 0 or x >= width:
            return False

        local_min_dist = get_min_dist_at(y, x)

        if fixed_tree is not None:
            dist, _ = fixed_tree.query([y, x])
            if dist < local_min_dist * 0.3:
                return False

        gy, gx = grid_coords(point)
        search_radius = int(np.ceil(base_min_dist * 1.5 / cell_size)) + 1
        for dy in range(-search_radius, search_radius + 1):
            for dx in range(-search_radius, search_radius + 1):
                ny, nx = gy + dy, gx + dx
                if 0 <= ny < grid_height and 0 <= nx < grid_width:
                    idx = grid[ny, nx]
                    if idx != -1:
                        other = points[idx]
                        dist = np.sqrt((point[0] - other[0])**2 + (point[1] - other[1])**2)
                        other_min_dist = get_min_dist_at(other[0], other[1])
                        if dist < min(local_min_dist, other_min_dist):
                            return False
        return True

    def add_point(point):
        points.append(point)
        idx = len(points) - 1
        gy, gx = grid_coords(point)
        if 0 <= gy < grid_height and 0 <= gx < grid_width:
            grid[gy, gx] = idx
        return idx

    for _ in range(100):
        initial = (np.random.uniform(0, height), np.random.uniform(0, width))
        if is_valid(initial):
            add_point(initial)
            active_list.append(0)
            break

    while active_list:
        idx = np.random.randint(len(active_list))
        point = points[active_list[idx]]
        local_min_dist = get_min_dist_at(point[0], point[1])

        found = False
        for _ in range(k):
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(local_min_dist, 2 * local_min_dist)
            new_point = (
                point[0] + radius * np.cos(angle),
                point[1] + radius * np.sin(angle)
            )

            if is_valid(new_point):
                new_idx = add_point(new_point)
                active_list.append(new_idx)
                found = True
                break

        if not found:
            active_list.pop(idx)

    all_points = fixed_list + points
    return np.array(all_points)


def weighted_poisson_sampling(width, height, edge_map, base_min_dist, fixed_points=None, k=30):
    """
    Weighted Poisson disk sampling based on edge strength.

    High edge strength: more points (rejection sampling favors edge areas)
    Low edge strength: fewer points

    Args:
        width: Image width
        height: Image height
        edge_map: Edge strength map (0-255)
        base_min_dist: Base minimum distance
        fixed_points: Points that must be preserved (y, x format)
        k: Number of attempts before rejecting a point
    """
    # Create probability map from edge strength
    # Higher edge strength = higher probability of accepting point
    edge_normalized = edge_map.astype(float) / 255.0

    # Apply Gaussian blur to smooth the edge map
    edge_smooth = cv2.GaussianBlur(edge_normalized, (15, 15), 0)

    # Create acceptance probability: base + edge_boost
    # Even non-edge areas have some probability (0.3), edge areas get up to 1.0
    acceptance_prob = 0.3 + 0.7 * edge_smooth

    cell_size = base_min_dist / np.sqrt(2)
    grid_width = int(np.ceil(width / cell_size))
    grid_height = int(np.ceil(height / cell_size))

    grid = np.full((grid_height, grid_width), -1, dtype=int)
    points = []
    active_list = []

    fixed_list = []
    if fixed_points is not None and len(fixed_points) > 0:
        fixed_list = [(pt[0], pt[1]) for pt in fixed_points
                      if 0 <= pt[0] < height and 0 <= pt[1] < width]

    fixed_tree = None
    if len(fixed_list) > 0:
        fixed_tree = KDTree(np.array(fixed_list))

    def grid_coords(point):
        return int(point[0] / cell_size), int(point[1] / cell_size)

    def get_acceptance_prob(y, x):
        if 0 <= int(y) < height and 0 <= int(x) < width:
            return acceptance_prob[int(y), int(x)]
        return 0.3

    def is_valid(point, check_probability=True):
        y, x = point
        if y < 0 or y >= height or x < 0 or x >= width:
            return False

        # Probabilistic acceptance based on edge strength
        if check_probability:
            if np.random.random() > get_acceptance_prob(y, x):
                return False

        if fixed_tree is not None:
            dist, _ = fixed_tree.query([y, x])
            if dist < base_min_dist * 0.5:
                return False

        gy, gx = grid_coords(point)
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                ny, nx = gy + dy, gx + dx
                if 0 <= ny < grid_height and 0 <= nx < grid_width:
                    idx = grid[ny, nx]
                    if idx != -1:
                        other = points[idx]
                        dist = np.sqrt((point[0] - other[0])**2 + (point[1] - other[1])**2)
                        if dist < base_min_dist:
                            return False
        return True

    def add_point(point):
        points.append(point)
        idx = len(points) - 1
        gy, gx = grid_coords(point)
        if 0 <= gy < grid_height and 0 <= gx < grid_width:
            grid[gy, gx] = idx
        return idx

    # Start with a random point (no probability check for initial)
    for _ in range(100):
        initial = (np.random.uniform(0, height), np.random.uniform(0, width))
        if is_valid(initial, check_probability=False):
            add_point(initial)
            active_list.append(0)
            break

    while active_list:
        idx = np.random.randint(len(active_list))
        point = points[active_list[idx]]

        found = False
        for _ in range(k * 2):  # More attempts to find edge-favored points
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(base_min_dist, 2 * base_min_dist)
            new_point = (
                point[0] + radius * np.cos(angle),
                point[1] + radius * np.sin(angle)
            )

            if is_valid(new_point, check_probability=True):
                new_idx = add_point(new_point)
                active_list.append(new_idx)
                found = True
                break

        if not found:
            active_list.pop(idx)

    all_points = fixed_list + points
    return np.array(all_points)


def polygonize_image(image_path, num_points=6000, sampling_mode="poisson"):
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
    n = 20
    x_range = np.linspace(0, w - 1, n)
    y_range = np.linspace(0, h - 1, n)
    top = np.column_stack([np.zeros(n), x_range])
    bottom = np.column_stack([np.full(n, h - 1), x_range])
    left = np.column_stack([y_range, np.zeros(n)])
    right = np.column_stack([y_range, np.full(n, w - 1)])
    boundary_points = np.vstack([top, bottom, left, right]).astype(int)

    # エッジ・コーナー・境界の重要な点を結合
    feature_points = np.vstack([edge_points, corner_points, boundary_points])
    feature_points = np.unique(feature_points, axis=0)

    # min_distは画像サイズとnum_pointsから計算
    area = h * w
    target_density = num_points / area
    min_dist = max(5, int(1.0 / np.sqrt(target_density * 1.5)))

    if sampling_mode == "poisson":
        # ハイブリッド方式: 特徴点を保持しつつポアソンディスクサンプリングで補間
        points = poisson_disk_sampling(w, h, min_dist, fixed_points=feature_points)
        points = points.astype(int)
    elif sampling_mode == "adaptive":
        # 適応的サンプリング: エッジ付近は密、遠いところは疎
        points = adaptive_poisson_sampling(w, h, edges, min_dist, fixed_points=feature_points)
        points = points.astype(int)
    elif sampling_mode == "weighted":
        # 重み付きサンプリング: エッジ強度に応じて点密度を変える
        points = weighted_poisson_sampling(w, h, edges, min_dist, fixed_points=feature_points)
        points = points.astype(int)
    else:
        # 従来方式: ランダムサンプリング (random or any other value)
        points = feature_points

    tri = Delaunay(points)
    img_polygonized = np.zeros_like(image)

    for simplex in tri.simplices:
        pts_yx = points[simplex]  # (y, x) format from np.where
        pts_xy = pts_yx[:, ::-1]  # convert to (x, y) for OpenCV
        x, y, w, h = cv2.boundingRect(pts_xy)
        mask = np.zeros((h, w), dtype=np.uint8)
        pts_shifted = pts_xy - np.array([[x, y]])
        cv2.fillConvexPoly(mask, pts_shifted, 255)
        roi = image[y : y + h, x : x + w]
        mean_color = cv2.mean(roi, mask=mask)
        cv2.fillConvexPoly(img_polygonized, pts_xy, mean_color[:3])

    return img_polygonized


def save_image(image, path):
    cv2.imwrite(path, image)
