"""Validate prediction CSVs and align them to the expected row IDs."""
from __future__ import annotations

import pandas as pd

from .contract import (ID_COLUMN, PREDICTION_COLUMNS, TARGET_COLUMNS, ValidationError,
                       numeric_columns, read_csv_strict, validate_ids)


def validate_predictions(frame: pd.DataFrame, expected_ids) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise ValidationError("Predictions must be a pandas DataFrame.")
    if any(not isinstance(column, str) for column in frame.columns):
        raise ValidationError("Prediction column names must be strings.")
    if not frame.columns.is_unique:
        raise ValidationError("Predictions have duplicate columns.")
    missing = sorted(set(PREDICTION_COLUMNS) - set(frame.columns))
    extra = sorted(set(frame.columns) - set(PREDICTION_COLUMNS))
    if missing or extra:
        raise ValidationError(f"Incorrect prediction columns; missing={missing}, unexpected={extra}.")
    expected = validate_ids(expected_ids, name="Expected row IDs")
    received = validate_ids(frame[ID_COLUMN], name="Prediction row IDs")
    missing_ids = set(expected) - set(received)
    extra_ids = set(received) - set(expected)
    if missing_ids or extra_ids:
        raise ValidationError(
            f"Prediction row IDs do not match: {len(missing_ids)} missing, {len(extra_ids)} unexpected."
        )
    result = frame.loc[:, PREDICTION_COLUMNS].copy()
    result[TARGET_COLUMNS] = numeric_columns(result, TARGET_COLUMNS)
    return result.set_index(ID_COLUMN).loc[expected].reset_index()


def read_predictions(path_or_file, expected_ids) -> pd.DataFrame:
    return validate_predictions(read_csv_strict(path_or_file, PREDICTION_COLUMNS), expected_ids)
