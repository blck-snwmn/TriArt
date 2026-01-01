import os
import tempfile

import cv2
import numpy as np
import pytest

from triart.polygonizer import polygonize_image, save_image


@pytest.fixture
def sample_image_path():
    """Create a temporary test image."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        # Create a simple test image with some edges
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(img, (20, 20), (80, 80), (255, 255, 255), -1)
        cv2.circle(img, (50, 50), 20, (0, 0, 255), -1)
        cv2.imwrite(f.name, img)
        yield f.name
    os.unlink(f.name)


class TestPolygonizeImage:
    def test_output_shape_matches_input(self, sample_image_path):
        """Output image should have the same shape as input."""
        original = cv2.imread(sample_image_path)
        result = polygonize_image(sample_image_path, num_points=500)

        assert result.shape == original.shape

    def test_output_is_numpy_array(self, sample_image_path):
        """Output should be a numpy array."""
        result = polygonize_image(sample_image_path, num_points=500)

        assert isinstance(result, np.ndarray)

    def test_output_has_three_channels(self, sample_image_path):
        """Output should have 3 color channels (BGR)."""
        result = polygonize_image(sample_image_path, num_points=500)

        assert len(result.shape) == 3
        assert result.shape[2] == 3

    def test_output_dtype_is_uint8(self, sample_image_path):
        """Output should have uint8 dtype."""
        result = polygonize_image(sample_image_path, num_points=500)

        assert result.dtype == np.uint8

    def test_different_num_points(self, sample_image_path):
        """Should work with different num_points values."""
        for num_points in [100, 500, 1000]:
            result = polygonize_image(sample_image_path, num_points=num_points)
            assert result is not None
            assert result.shape[0] > 0

    def test_invalid_image_path_raises_error(self):
        """Should raise error for non-existent image."""
        with pytest.raises(Exception):
            polygonize_image("non_existent_image.jpg")


class TestSaveImage:
    def test_save_creates_file(self, sample_image_path):
        """save_image should create a file."""
        result = polygonize_image(sample_image_path, num_points=500)

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            output_path = f.name

        try:
            save_image(result, output_path)
            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_saved_image_is_readable(self, sample_image_path):
        """Saved image should be readable by OpenCV."""
        result = polygonize_image(sample_image_path, num_points=500)

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            output_path = f.name

        try:
            save_image(result, output_path)
            loaded = cv2.imread(output_path)
            assert loaded is not None
            assert loaded.shape == result.shape
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
