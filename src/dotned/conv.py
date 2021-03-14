"""
Math modules for AI
Author: Vlad Nedelcu
Version: 0.1

Math related to images processing
and filtering
"""
from typing import Tuple, List

import numpy as np


def calc_final_shape(image: np.ndarray, kernel: np.ndarray) -> Tuple[int, int]:
    """
    calculates the shape for convolution

    matrix_1 = n*n
    matrix_2 = f*f

    Formula: [(n-f)+1, (n-f)+1]
    """
    rows = image.shape[0]-kernel.shape[1]+1
    cols = image.shape[1]-kernel.shape[1]+1

    return rows, cols


def calc_filter(sub_matrix: np.ndarray, kernel: np.ndarray) -> int:
    return np.sum(sub_matrix * kernel)


def get_submatrix(positions: List[int], image: np.ndarray) -> np.ndarray:
    """
    Fetches the submatrix given by the position list

    postions = [i, start_row_pos, j, start_col_pos]
    """
    get_rows = np.array(range(positions[0], positions[1]))
    get_cols = np.array(range(positions[2], positions[3]))
    filter_matrix = np.ix_(get_rows, get_cols)

    return image[filter_matrix]


def padding(image: np.ndarray, kernel: np.ndarray, same: bool) -> np.ndarray:
    """
    Pads the image if the same argument is set to True
    """
    if same is True:
        padding = int((kernel.shape[0] - 1)/2)
        return np.pad(image, padding)

    return image


def conv2D(image: np.ndarray, kernel: np.ndarray, pad: str) -> np.ndarray:
    """
    Calculates a convolution
    """
    pad = True if padding == 'same' else False
    image = padding(image, kernel, pad)
    result = np.zeros(calc_final_shape(image, kernel), dtype=int)

    filter_max_row, filter_max_col = kernel.shape
    start_row_pos = filter_max_row
    start_col_pos = filter_max_col


    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            poz = [i, start_row_pos, j, start_col_pos]
            product = calc_filter(get_submatrix(poz, image), kernel)
            result[i][j] = product
            start_col_pos += 1
        start_col_pos = filter_max_col
        start_row_pos += 1

    return result
