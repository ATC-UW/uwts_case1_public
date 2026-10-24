from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from case1.contract import FEATURE_COLUMNS, TARGET_COLUMNS


@pytest.fixture
def dataset_dir(tmp_path):
    directory = tmp_path / "data"
    directory.mkdir()
    rng = np.random.default_rng(81)
    frames = {}
    for name, start, rows in (("train", 0, 30), ("validation", 40, 12), ("test", 60, 12)):
        frame = pd.DataFrame(rng.normal(size=(rows, 79)), columns=FEATURE_COLUMNS)
        frame.insert(0, "timestamp", np.arange(start, start + rows))
        frame.insert(0, "row_id", [f"{start + i:08d}" for i in range(rows)])
        if name != "test":
            for target in TARGET_COLUMNS:
                frame[target] = frame["feature_00"] * 0.0001 + rng.normal(0, 0.0001, rows)
        frames[f"{name}.csv"] = frame
    sample = pd.DataFrame(0.0, index=range(12), columns=TARGET_COLUMNS)
    sample.insert(0, "row_id", frames["test.csv"]["row_id"])
    frames["sample_submission.csv"] = sample
    for name, frame in frames.items():
        frame.to_csv(directory / name, index=False, float_format="%.17g")
    manifest = {
        "format_version": 1, "dataset_id": "unit-fixture", "horizon_seconds": 5,
        "id_column": "row_id", "time_column": "timestamp", "feature_columns": FEATURE_COLUMNS,
        "target_columns": TARGET_COLUMNS,
        "files": {name: {"sha256": hashlib.sha256((directory / name).read_bytes()).hexdigest(),
                         "rows": len(frame)} for name, frame in frames.items()},
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return directory
