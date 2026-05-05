"""
Helpers for converting numpy arrays and other ML outputs to JSON-serializable
Python structures. FastAPI's JSONResponse cannot serialize numpy types directly.
"""

import numpy as np


def ndarray_to_list(arr: np.ndarray) -> list:
    """Recursively convert ndarray to nested Python lists."""
    return arr.tolist()


def float32(v) -> float:
    """Safely convert any numeric type to a Python float."""
    return float(v)


def safe_dict(d: dict) -> dict:
    """
    Walk a dict and convert any numpy scalars / arrays to Python native types.
    Safe for nested dicts and lists.
    """
    out = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = float(v)
        elif isinstance(v, dict):
            out[k] = safe_dict(v)
        elif isinstance(v, list):
            out[k] = safe_list(v)
        else:
            out[k] = v
    return out


def safe_list(lst: list) -> list:
    out = []
    for v in lst:
        if isinstance(v, np.ndarray):
            out.append(v.tolist())
        elif isinstance(v, (np.integer,)):
            out.append(int(v))
        elif isinstance(v, (np.floating,)):
            out.append(float(v))
        elif isinstance(v, dict):
            out.append(safe_dict(v))
        elif isinstance(v, list):
            out.append(safe_list(v))
        else:
            out.append(v)
    return out


def confusion_matrix_to_dict(cm: np.ndarray, class_names: list[str]) -> dict:
    """Convert confusion matrix to a frontend-friendly format."""
    return {
        "matrix": cm.tolist(),
        "class_names": class_names,
        "n_classes": len(class_names),
    }
