import numpy as np


def reshape_to(arr: np.ndarray, new_shape: tuple) -> np.ndarray:
    """
    Return `arr` reshaped to `new_shape`, using .reshape().
    Assume new_shape's product matches arr.size.
    """
    return arr.reshape(new_shape)
    


def reshape_with_inferred_dim(arr: np.ndarray, known_dim: int) -> np.ndarray:
    """
    Reshape `arr` (1D) into a 2D array with `known_dim` columns,
    letting NumPy infer the number of rows automatically using -1.
    """
    return arr.reshape((-1, known_dim))


def reshape_shares_memory(arr: np.ndarray, new_shape: tuple) -> bool:
    """
    Reshape `arr` to `new_shape`, and return True if the result
    shares memory with `arr` (a view was returned), False if it
    does not (a copy was returned). Use np.shares_memory to check
    the actual outcome rather than assuming.
    """
    result = arr.reshape(new_shape)
    return bool(np.shares_memory(arr, result))


def flatten_safe(arr: np.ndarray) -> np.ndarray:
    """
    Return a flattened (1D) version of `arr` that is guaranteed
    to be independent of `arr`, mutating the result must never
    affect `arr`. Use flatten(), not ravel().
    """
    return arr.flatten()


def flatten_efficient(arr: np.ndarray) -> np.ndarray:
    """
    Return a flattened (1D) version of `arr`, using ravel(),
    which may or may not share memory with `arr` depending on
    its layout.
    """
    return arr.ravel()


def compare_flatten_ravel(arr: np.ndarray) -> dict:
    f = arr.flatten()
    r = arr.ravel()
    return {
        "flatten_result": f,
        "ravel_result": r,
        "flatten_shares_memory": bool(np.shares_memory(arr, f)),
        "ravel_shares_memory": bool(np.shares_memory(arr, r)),
    }
