import pytest
import numpy as np

from dotned import pool

POOL_2D = {
    "3x3_filter": (3, 1, np.array([[12, 12], [12, 12]])),
    "2x2_filter": (2, 2, np.array([[9, 12], [4, 8]])),
    "2x1_filter": (2, 1, np.array([[9, 12, 12], [9, 12, 12], [4, 5, 8]]))
}


@pytest.fixture
def image() -> np.ndarray:
    return np.array([
        [3,4,5,6],
        [9,4,12,6],
        [0,1,5,8],
        [1,4,0,3],
    ])


class TestPooling:

    @pytest.mark.parametrize("kernel, stride, expected", POOL_2D.values(), ids=POOL_2D.keys())
    def test_max_pool_2d(self, image, kernel, stride, expected):
        result = pool.max_pool2D(image, kernel, stride)

        assert (result == expected).all()

    def test_negative(self, image):
        with pytest.raises(IndexError):
            pool.max_pool2D(image, 3, 2)