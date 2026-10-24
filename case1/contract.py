"""The small, explicit file contract shared by the case and its judge."""
from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import IO
import warnings

import numpy as np
import pandas as pd

FORMAT_VERSION = 1
HORIZON_SECONDS = 5
ID_COLUMN = "row_id"
TIME_COLUMN = "timestamp"
FEATURE_COLUMNS = [f"feature_{i:02d}" for i in range(79)]
TARGET_COLUMNS = [f"target_{i}" for i in range(9)]
PREDICTION_COLUMNS = [ID_COLUMN, *TARGET_COLUMNS]
FEATURE_FILE_COLUMNS = [ID_COLUMN, TIME_COLUMN, *FEATURE_COLUMNS]
LABELED_FILE_COLUMNS = [*FEATURE_FILE_COLUMNS, *TARGET_COLUMNS]
DATA_FILES = ("train.csv", "validation.csv", "test.csv", "sample_submission.csv")


class ValidationError(ValueError):
    """A file does not satisfy the published competition contract."""


@dataclass(frozen=True)
class Dataset:
    manifest: dict
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame

    @property
    def dataset_id(self) -> str:
        return self.manifest["dataset_id"]


def read_csv_strict(source: str | Path | IO, expected_columns: list[str]) -> pd.DataFrame:
    """Read UTF-8 CSV, retaining IDs exactly and rejecting ambiguous headers/rows."""
    try:
        if hasattr(source, "read"):
            raw = source.read()
            text = raw.decode("utf-8-sig") if isinstance(raw, bytes) else raw.lstrip("\ufeff")
        else:
            text = Path(source).read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as exc:
        raise ValidationError(f"Cannot read CSV: {exc}") from exc
    if "\x00" in text:
        raise ValidationError("CSV contains a NUL byte.")
    try:
        records = csv.reader(io.StringIO(text), strict=True)
        header = next(records, None)
        if not header:
            raise ValidationError("CSV is empty or has no header.")
        if len(header) != len(set(header)):
            raise ValidationError("CSV has duplicate column names.")
        missing = sorted(set(expected_columns) - set(header))
        extra = sorted(set(header) - set(expected_columns))
        if missing or extra:
            raise ValidationError(f"Incorrect CSV columns; missing={missing}, unexpected={extra}.")
        for line_number, row in enumerate(records, start=2):
            if len(row) != len(header):
                raise ValidationError(f"CSV record {line_number} has {len(row)} fields; expected {len(header)}.")
        with warnings.catch_warnings():
            warnings.simplefilter("error", pd.errors.ParserWarning)
            frame = pd.read_csv(
                io.StringIO(text), dtype={ID_COLUMN: str}, keep_default_na=False,
                float_precision="round_trip", index_col=False,
            )
    except (csv.Error, pd.errors.ParserError, pd.errors.ParserWarning, UnicodeError) as exc:
        raise ValidationError(f"Malformed CSV: {exc}") from exc
    return frame.loc[:, expected_columns]


def validate_ids(values, *, name: str = "row IDs") -> list[str]:
    ids = list(values)
    if not ids:
        raise ValidationError(f"{name} cannot be empty.")
    if any(not isinstance(value, str) or not value or value != value.strip() for value in ids):
        raise ValidationError(f"{name} must be nonempty strings with no surrounding whitespace.")
    if len(set(ids)) != len(ids):
        raise ValidationError(f"{name} contain duplicates.")
    return ids


def numeric_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    try:
        converted = frame.loc[:, columns].apply(pd.to_numeric, errors="raise")
        if any(np.iscomplexobj(converted[column].to_numpy()) for column in columns):
            raise ValidationError("Data columns must contain real numeric values, not complex numbers.")
        values = converted.to_numpy(dtype=np.float64)
    except ValidationError:
        raise
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValidationError("Data columns must contain numeric values.") from exc
    if not np.isfinite(values).all():
        raise ValidationError("Data columns must contain only finite values; no blanks, NaN, or infinity.")
    # Normalize all accepted values, including booleans, to the actual CSV wire type.
    return pd.DataFrame(values, index=frame.index, columns=columns)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_dataset(data_dir: str | Path) -> Dataset:
    """Validate manifest, checksums, schema, chronology and the purged boundaries."""
    data_dir = Path(data_dir)
    try:
        manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"Cannot read data/manifest.json: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValidationError("Dataset manifest must be an object.")
    expected = {
        "format_version": FORMAT_VERSION, "horizon_seconds": HORIZON_SECONDS,
        "id_column": ID_COLUMN, "time_column": TIME_COLUMN,
        "feature_columns": FEATURE_COLUMNS, "target_columns": TARGET_COLUMNS,
    }
    for key, value in expected.items():
        if manifest.get(key) != value or (type(value) is int and type(manifest.get(key)) is not int):
            raise ValidationError(f"Unsupported dataset manifest field: {key}.")
    if not isinstance(manifest.get("dataset_id"), str) or not manifest["dataset_id"].strip():
        raise ValidationError("Dataset manifest needs a nonempty dataset_id.")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != set(DATA_FILES):
        raise ValidationError(f"Dataset manifest must describe exactly {list(DATA_FILES)}.")
    frames = {}
    for name in DATA_FILES:
        info = files[name]
        if not isinstance(info, dict) or not isinstance(info.get("sha256"), str):
            raise ValidationError(f"Missing checksum for {name}.")
        if type(info.get("rows")) is not int or info["rows"] <= 0:
            raise ValidationError(f"Invalid row count for {name}.")
        path = data_dir / name
        if not path.is_file() or path.is_symlink():
            raise ValidationError(f"Dataset file is missing or is a symbolic link: {name}.")
        if sha256_file(path) != info["sha256"]:
            raise ValidationError(f"Checksum mismatch for {name}; restore the original dataset.")
        columns = (PREDICTION_COLUMNS if name == "sample_submission.csv" else
                   FEATURE_FILE_COLUMNS if name == "test.csv" else LABELED_FILE_COLUMNS)
        frame = read_csv_strict(path, columns)
        if len(frame) != info["rows"]:
            raise ValidationError(f"Row count mismatch for {name}.")
        validate_ids(frame[ID_COLUMN], name=f"{name} row IDs")
        numbers = [column for column in columns if column != ID_COLUMN]
        frame[numbers] = numeric_columns(frame, numbers)
        if TIME_COLUMN in frame:
            times = frame[TIME_COLUMN].to_numpy(dtype=np.float64)
            if np.any(np.diff(times) <= 0) or np.any(times != np.floor(times)):
                raise ValidationError(f"{name} timestamps must be strictly increasing integer seconds.")
        frames[name] = frame
    labeled = [frames[name] for name in ("train.csv", "validation.csv", "test.csv")]
    validate_ids([value for frame in labeled for value in frame[ID_COLUMN]], name="Dataset row IDs")
    for earlier, later in zip(labeled, labeled[1:]):
        if earlier[TIME_COLUMN].iloc[-1] + HORIZON_SECONDS >= later[TIME_COLUMN].iloc[0]:
            raise ValidationError("Chronological split boundaries require a gap longer than the target horizon.")
    # Import here to keep the CSV contract independent of submission-specific rules.
    from .submission import validate_predictions
    validate_predictions(frames["sample_submission.csv"], frames["test.csv"][ID_COLUMN])
    return Dataset(manifest, frames["train.csv"], frames["validation.csv"], frames["test.csv"])


def validate_dataset(data_dir: str | Path) -> dict:
    return load_dataset(data_dir).manifest
