import pytest
import numpy as np

from dotned import conv

TEST_PADDING = {
    "padding_same": (True, (8, 8)),
    "padding_valid": (False, (6, 6)),
    "padding_weird": (123, (6, 6))
}

TEST_SUBMATRIX = {
    "3x3_submatrix": ([0,3,0,3], np.array([[10, 10, 10], [10, 10, 10], [10, 10, 10]])),
    "2x2_submatrix": ([0,2,0,2], np.array([[10, 10,], [10, 10]]))
}

TEST_CONV2D = {
    "padding_same": ('same', np.array([
        [-20, 0, 20, 20, 0, 0],
        [-30, 0, 30, 30, 0, 0],
        [-30, 0, 30, 30, 0, 0],
        [-30, 0, 30, 30, 0, 0],
        [-30, 0, 30, 30, 0, 0],
        [-20, 0, 20, 20, 0, 0],
    ])),
    "padding_valid": ('valid', np.array([
        [0, 30, 30, 0],
        [0, 30, 30, 0],
        [0, 30, 30, 0],
        [0, 30, 30, 0]
    ])),
}


@pytest.fixture
def image() -> np.array:
    return np.array([
        [10, 10, 10, 0, 0, 0],
        [10, 10, 10, 0, 0, 0],
        [10, 10, 10, 0, 0, 0],
        [10, 10, 10, 0, 0, 0],
        [10, 10, 10, 0, 0, 0],
        [10, 10, 10, 0, 0, 0],
    ])

@pytest.fixture
def submatrix() -> np.array:
    return np.array([
        [10, 10, 10],
        [10, 10, 10],
        [10, 10, 10]
    ])


@pytest.fixture
def kernel() -> np.array:
    return  np.array([
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1],
    ])

class TestConvUtils:

    def test_calc_final_shape(self, image, kernel):
        result = conv.calc_final_shape(image, kernel)

        assert isinstance(result, tuple)
        assert result == (4, 4)

    def test_calc_filter(self, submatrix, kernel):
        result = conv.calc_filter(submatrix, kernel)

        assert result == 0

    @pytest.mark.parametrize("positions, expected", TEST_SUBMATRIX.values(), ids=TEST_SUBMATRIX.keys())
    def test_get_submatrix(self, image, positions, expected):
        result = conv.get_submatrix(positions, image)

        assert (result == expected).all()

    @pytest.mark.parametrize("padding, expected", TEST_PADDING.values(), ids=TEST_PADDING.keys())
    def test_padding(self, image, kernel, padding, expected):
        result = conv.padding(image, kernel, padding)

        assert result.shape == expected

    @pytest.mark.parametrize("pad, expected", TEST_CONV2D.values(), ids=TEST_CONV2D.keys())
    def test_conv_2D(self, image, kernel, pad, expected):
        result = conv.conv2D(image, kernel, pad)

        assert (result == expected).all()
