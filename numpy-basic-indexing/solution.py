import numpy as np


def get_element_1d(arr: np.ndarray, index: int):
    """
    Return the element of 1D array `arr` at position `index`.
    `index` may be negative.
    """
    return arr[index]


def get_element_2d(arr: np.ndarray, row: int, col: int):
    """
    Return the element of 2D array `arr` at the given `row` and
    `col`, using the arr[row, col] syntax (not arr[row][col]).
    """
    return arr[row,col]


def get_row(arr: np.ndarray, row: int) -> np.ndarray:
    """
    Return the entire row at index `row` from 2D array `arr`,
    using a partial index.
    """
    return arr[row,]
