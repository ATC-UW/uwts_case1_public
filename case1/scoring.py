"""Official pooled, unweighted, uncentered R². Zero prediction scores zero."""
from __future__ import annotations

import numpy as np

from .contract import TARGET_COLUMNS, ValidationError


def _arrays(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    if np.iscomplexobj(y_true) or np.iscomplexobj(y_pred):
        raise ValidationError("Scores require real numeric arrays.")
    try:
        actual = np.asarray(y_true, dtype=np.float64)
        predicted = np.asarray(y_pred, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise ValidationError("Scores require numeric arrays.") from exc
    if actual.shape != predicted.shape or actual.size == 0 or actual.ndim not in (1, 2):
        raise ValidationError("Actual and predicted arrays must have the same nonempty 1D or 2D shape.")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValidationError("Scores require finite actual and predicted values.")
    return actual, predicted


def score(y_true, y_pred) -> float:
    """Compute 1 - sum((y - prediction)²) / sum(y²), pooling all entries."""
    actual, predicted = _arrays(y_true, y_pred)
    if not np.any(actual):
        raise ValidationError("Score is undefined because the true returns have zero sum of squares.")
    # Scale both arrays together, preserving the ratio while avoiding squaring overflow.
    scale = max(float(np.max(np.abs(actual))), float(np.max(np.abs(predicted))))
    actual = actual / scale
    predicted = predicted / scale
    denominator = float(np.sum(actual * actual))
    if denominator == 0:
        raise ValidationError("Prediction magnitudes are too large to represent a finite score.")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        result = 1.0 - float(np.sum((actual - predicted) ** 2)) / denominator
    if not np.isfinite(result):
        raise ValidationError("Prediction magnitudes are too large to represent a finite score.")
    return result


def scores_by_target(y_true, y_pred) -> dict[str, float]:
    actual, predicted = _arrays(y_true, y_pred)
    if actual.ndim != 2 or actual.shape[1] != len(TARGET_COLUMNS):
        raise ValidationError("Per-target scores require nine columns in target_0…target_8 order.")
    return {name: score(actual[:, index], predicted[:, index])
            for index, name in enumerate(TARGET_COLUMNS)}
