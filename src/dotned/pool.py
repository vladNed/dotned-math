"""
Math modules for AI
Author: Vlad Nedelcu
Version: 0.1

Math related to images processing
and filtering
"""
import numpy as np

from .utils import get_submatrix

def max_pool2D(image: np.ndarray, kernel: int, stride: int = 1, pad: int = 0) -> np.ndarray:
    """
    Max pool a 2D image
    """
    shape = np.int32(((image.shape[0] + 2*pad - kernel)/stride) + 1)
    result = np.zeros((shape, shape), dtype=int)

    start_row_pos = kernel
    start_col_pos = kernel

    rows = 0
    cols = 0

    for i in range(0, image.shape[0]-(kernel-stride), stride):
        for j in range(0, image.shape[1]-(kernel-stride), stride):
            poz = np.array([i, start_row_pos, j, start_col_pos])
            submatrix = get_submatrix(poz, image)
            result[rows][cols] = submatrix.max()
            start_col_pos += stride
            cols += 1
        cols = 0
        rows += 1
        start_col_pos = kernel
        start_row_pos += stride

    return result