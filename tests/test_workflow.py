from __future__ import annotations

import io
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import pandas as pd
import pytest

from case1.contract import TARGET_COLUMNS, ValidationError, load_dataset
from case1.packaging import create_submission
from case1.submission import read_predictions
from organizer.build_release import SOURCE_FILES, build_release

ROOT = Path(__file__).resolve().parents[1]


def test_submission_contains_exactly_allowlisted_files(tmp_path):
    source = tmp_path / "solution.py"
    source.write_text("print('source is retained, not executed')\n")
    (tmp_path / "answer_key.csv").write_text("must never be included")
    frame = pd.DataFrame(0.0, index=range(2), columns=TARGET_COLUMNS)
    frame.insert(0, "row_id", ["02", "01"])
    path, submission_id = create_submission(
        frame, ["01", "02"], team_name="A / B", dataset_id="test",
        solution_path=source, output_dir=tmp_path / "out",
    )
    assert len(submission_id) == 32
    assert path.name == f"submission_a_b_{submission_id}.zip"
    with zipfile.ZipFile(path) as archive:
        assert set(archive.namelist()) == {"predictions.csv", "solution.py", "submission.json"}
        metadata = json.loads(archive.read("submission.json"))
        assert metadata == {"format_version": 1, "dataset_id": "test", "team_name": "A / B",
                            "submission_id": submission_id}
        assert read_predictions(io.BytesIO(archive.read("predictions.csv")), ["01", "02"])["row_id"].tolist() == ["01", "02"]


def test_release_allowlist_ignores_private_decoys(dataset_dir, tmp_path):
    (dataset_dir / "answer_key.csv").write_text("private")
    (dataset_dir / "generator.py").write_text("private")
    path = build_release(dataset_dir, tmp_path / "releases", version="test")
    with zipfile.ZipFile(path) as archive:
        names = {name.split("/", 1)[1] for name in archive.namelist()}
        assert names == set(SOURCE_FILES) | {
            "data/manifest.json", "data/train.csv", "data/validation.csv", "data/test.csv",
            "data/sample_submission.csv", "release.json",
        }
        assert "answer_key.csv" not in " ".join(names)
        assert "generator.py" not in " ".join(names)
    with pytest.raises(FileExistsError):
        build_release(dataset_dir, tmp_path / "releases", version="test")


def test_release_rejects_private_metadata(dataset_dir, tmp_path):
    path = dataset_dir / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["seed"] = 123
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValidationError, match="unexpected fields"):
        build_release(dataset_dir, tmp_path / "releases")


def test_extracted_starter_runs_from_an_unrelated_directory(dataset_dir, tmp_path):
    archive_path = build_release(dataset_dir, tmp_path / "releases", version="smoke")
    unpacked = tmp_path / "unpacked"
    # This is our own freshly built archive with fixed allowlisted paths.
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(unpacked)
    starter = unpacked / "uwts_case1_starter_smoke"
    unrelated = tmp_path / "elsewhere"
    unrelated.mkdir()
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    run = subprocess.run(
        [sys.executable, str(starter / "solution.py"), "--team", "Smoke Team"],
        cwd=unrelated, env=env, text=True, capture_output=True, timeout=60,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    assert "Validation score:" in run.stdout
    assert "Submission ID:" in run.stdout
    outputs = list((starter / "submissions").glob("*.zip"))
    assert len(outputs) == 1
    with zipfile.ZipFile(outputs[0]) as archive:
        frame = read_predictions(io.BytesIO(archive.read("predictions.csv")),
                                 load_dataset(starter / "data").test["row_id"])
        assert frame.shape == (12, 10)
        assert set(archive.namelist()) == {"predictions.csv", "solution.py", "submission.json", "requirements.txt"}


def test_packaging_collision_never_deletes_an_existing_submission(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import case1.packaging as packaging

    monkeypatch.setattr(packaging, "uuid4", lambda: SimpleNamespace(hex="a" * 32))
    source = tmp_path / "solution.py"
    source.write_text("pass\n")
    path = tmp_path / ("submission_team_" + "a" * 32 + ".zip")
    path.write_bytes(b"existing submission")
    frame = pd.DataFrame(0.0, index=[0], columns=TARGET_COLUMNS)
    frame.insert(0, "row_id", ["1"])
    with pytest.raises(FileExistsError):
        create_submission(frame, ["1"], team_name="Team", dataset_id="test",
                          solution_path=source, output_dir=tmp_path)
    assert path.read_bytes() == b"existing submission"


@pytest.mark.parametrize("source_text", ["", " \n\t"])
def test_packager_rejects_empty_source(tmp_path, source_text):
    source = tmp_path / "solution.py"
    source.write_text(source_text)
    frame = pd.DataFrame(0.0, index=[0], columns=TARGET_COLUMNS)
    frame.insert(0, "row_id", ["1"])
    with pytest.raises(ValidationError, match="nonempty source"):
        create_submission(frame, ["1"], team_name="Team", dataset_id="test",
                          solution_path=source, output_dir=tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("kind", ["complex", "team_surrogate", "dataset_surrogate"])
def test_packager_rejects_values_that_cannot_reach_the_judge(tmp_path, kind):
    source = tmp_path / "solution.py"
    source.write_text("pass\n")
    frame = pd.DataFrame(0.0, index=[0], columns=TARGET_COLUMNS)
    frame.insert(0, "row_id", ["1"])
    team, dataset = "Team", "test"
    if kind == "complex":
        frame["target_0"] = [1.0 + 2.0j]
    elif kind == "team_surrogate":
        team = "Bad" + chr(0xD800)
    else:
        dataset = "Bad" + chr(0xD800)
    with pytest.raises(ValidationError, match="real|Unicode"):
        create_submission(frame, ["1"], team_name=team, dataset_id=dataset,
                          solution_path=source, output_dir=tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_boolean_predictions_are_serialized_as_numeric_values(tmp_path):
    source = tmp_path / "solution.py"
    source.write_text("pass\n")
    frame = pd.DataFrame(True, index=[0], columns=TARGET_COLUMNS)
    frame.insert(0, "row_id", ["1"])
    path, _ = create_submission(frame, ["1"], team_name="Team", dataset_id="test",
                                solution_path=source, output_dir=tmp_path / "out")
    with zipfile.ZipFile(path) as archive:
        actual = read_predictions(io.BytesIO(archive.read("predictions.csv")), ["1"])
    assert (actual[TARGET_COLUMNS].to_numpy() == 1.0).all()
