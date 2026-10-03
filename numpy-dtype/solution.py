import numpy as np


def get_dtype_name(arr: np.ndarray) -> str:
    """
    Return the dtype of `arr` as a string (e.g. "int64").
    """
    return arr.dtype


def create_with_dtype(values: list, dtype) -> np.ndarray:
    """
    Create an ndarray from `values`, explicitly using the given
    `dtype`, and return it.
    """
    return np.array(values, dtype=dtype)


def convert_dtype(arr: np.ndarray, new_dtype) -> np.ndarray:
    """
    Return a new array with the same values as `arr`, converted
    to `new_dtype`, without modifying the original array `arr`.
    """
    return arr.astype(new_dtype)
