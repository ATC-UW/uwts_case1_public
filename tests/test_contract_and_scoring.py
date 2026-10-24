from __future__ import annotations

import hashlib
import io
import json

import numpy as np
import pandas as pd
import pytest

from case1.contract import (PREDICTION_COLUMNS, TARGET_COLUMNS, ValidationError,
                           load_dataset, validate_dataset)
from case1.scoring import score, scores_by_target
from case1.submission import read_predictions, validate_predictions


def predictions(ids):
    frame = pd.DataFrame(0.0, index=range(len(ids)), columns=TARGET_COLUMNS)
    frame.insert(0, "row_id", ids)
    return frame


def test_metric_is_pooled_uncentered_and_not_target_average():
    actual = np.array([[1.0, 10.0], [1.0, 10.0]])
    predicted = np.array([[1.0, 0.0], [1.0, 0.0]])
    assert score(actual, predicted) == pytest.approx(1 - 200 / 202)
    assert score(actual, np.zeros_like(actual)) == 0
    assert score(actual, actual) == 1
    assert score(actual, -actual) == -3
    assert score(np.array([1e-300]), np.array([0.0])) == 0


@pytest.mark.parametrize("actual,predicted", [
    ([0, 0], [0, 0]), ([1], [1, 2]), ([], []), ([1], [np.nan]), ([np.inf], [0]),
])
def test_metric_rejects_undefined_or_malformed_inputs(actual, predicted):
    with pytest.raises(ValidationError):
        score(actual, predicted)


def test_per_target_names_and_shape():
    actual = np.ones((5, 9))
    assert scores_by_target(actual, actual) == dict.fromkeys(TARGET_COLUMNS, 1)
    with pytest.raises(ValidationError):
        scores_by_target(actual[:, :8], actual[:, :8])


def test_prediction_alignment_and_string_id_roundtrip():
    frame = predictions(["002", "001", "NA"])
    frame.loc[0, "target_0"] = 0.012345678901234567
    aligned = read_predictions(io.StringIO(frame.to_csv(index=False, float_format="%.17g")),
                               ["001", "NA", "002"])
    assert aligned["row_id"].tolist() == ["001", "NA", "002"]
    assert aligned.loc[2, "target_0"] == frame.loc[0, "target_0"]


@pytest.mark.parametrize("kind", ["missing", "extra", "duplicate", "nan", "infinity", "text", "column"])
def test_bad_predictions_rejected(kind):
    frame = predictions(["a", "b"])
    if kind == "missing":
        frame = frame.iloc[:1]
    elif kind == "extra":
        frame = predictions(["a", "b", "c"])
    elif kind == "duplicate":
        frame["row_id"] = ["a", "a"]
    elif kind == "nan":
        frame.loc[0, "target_0"] = np.nan
    elif kind == "infinity":
        frame.loc[0, "target_0"] = np.inf
    elif kind == "text":
        frame["target_0"] = ["bad", "1"]
    elif kind == "column":
        frame["extra"] = 0
    with pytest.raises(ValidationError):
        validate_predictions(frame, ["a", "b"])


def test_duplicate_raw_headers_and_ragged_records_rejected():
    header = ",".join(PREDICTION_COLUMNS)
    for text in [header + ",target_0\n", header + "\na,1,2\n", header + "\n\n"]:
        with pytest.raises(ValidationError):
            read_predictions(io.StringIO(text), ["a"])


def test_dataset_checks_schema_checksums_and_leading_zero_ids(dataset_dir):
    dataset = load_dataset(dataset_dir)
    assert dataset.dataset_id == "unit-fixture"
    assert dataset.train["row_id"].iloc[0] == "00000000"
    with (dataset_dir / "test.csv").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValidationError, match="Checksum"):
        validate_dataset(dataset_dir)


def test_dataset_rejects_target_windows_touching_next_split(dataset_dir):
    path = dataset_dir / "validation.csv"
    frame = pd.read_csv(path, dtype={"row_id": str})
    frame["timestamp"] -= 6  # Train ends at 29, its target at 34 now reaches validation.
    frame.to_csv(path, index=False)
    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["validation.csv"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValidationError, match="gap"):
        validate_dataset(dataset_dir)


def test_binary_csv_decode_errors_are_actionable():
    with pytest.raises(ValidationError, match="Cannot read CSV"):
        read_predictions(io.BytesIO(b"\xff\xfe"), ["a"])


def test_complex_predictions_are_not_silently_discarded():
    with pytest.raises(ValidationError, match="real"):
        score([1.0], [1.0 + 2.0j])


def test_numeric_overflow_is_a_validation_error():
    with pytest.raises(ValidationError):
        score([1.0], [10 ** 1000])
