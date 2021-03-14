"""
Math modules for AI
Author: Vlad Nedelcu
Version: 0.1

Math related to images processing
and filtering
"""
import numpy as np
from typing import List


def get_submatrix(positions: List[int], image: np.ndarray) -> np.ndarray:
    """
    Fetches the submatrix given by the position list

    postions = [i, start_row_pos, j, start_col_pos]
    """
    get_rows = np.array(range(positions[0], positions[1]))
    get_cols = np.array(range(positions[2], positions[3]))
    filter_matrix = np.ix_(get_rows, get_cols)

    return image[filter_matrix]