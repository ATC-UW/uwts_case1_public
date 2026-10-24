"""Create a small submission archive by explicitly naming every included file."""
from __future__ import annotations

import json
from pathlib import Path
import re
from uuid import uuid4
import zipfile

import pandas as pd

from .contract import FORMAT_VERSION, ValidationError
from .submission import validate_predictions


def _text_file(path: str | Path, *, limit: int, label: str, allow_empty: bool = True) -> bytes:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValidationError(f"{label} must be a regular file.")
    if path.stat().st_size > limit:
        raise ValidationError(f"{label} exceeds the {limit:,}-byte limit.")
    data = path.read_bytes()
    try:
        text = data.decode("utf-8")
    except UnicodeError as exc:
        raise ValidationError(f"{label} must be UTF-8 text.") from exc
    if "\x00" in text:
        raise ValidationError(f"{label} contains a NUL byte.")
    if not allow_empty and not text.strip():
        raise ValidationError(f"{label} must contain nonempty source text.")
    return data


def create_submission(
    predictions: pd.DataFrame, expected_ids, *, team_name: str, dataset_id: str,
    solution_path: str | Path, output_dir: str | Path,
    requirements_path: str | Path | None = None,
) -> tuple[Path, str]:
    """Validate, package, and return (archive path, unique submission ID)."""
    if not isinstance(team_name, str) or not team_name.strip() or len(team_name) > 120:
        raise ValidationError("Team name must contain 1–120 characters.")
    team_name = team_name.strip()
    if any(ord(char) < 32 or ord(char) == 127 for char in team_name):
        raise ValidationError("Team name cannot contain control characters.")
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise ValidationError("A dataset_id from data/manifest.json is required.")
    aligned = validate_predictions(predictions, expected_ids)
    submission_id = uuid4().hex
    metadata = {
        "format_version": FORMAT_VERSION, "dataset_id": dataset_id,
        "team_name": team_name, "submission_id": submission_id,
    }
    payloads = {
        "predictions.csv": aligned.to_csv(index=False, float_format="%.17g").encode("utf-8"),
        "solution.py": _text_file(solution_path, limit=1_000_000, label="solution.py", allow_empty=False),
        "submission.json": (json.dumps(metadata, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
    }
    if requirements_path is not None:
        payloads["requirements.txt"] = _text_file(
            requirements_path, limit=64_000, label="requirements.txt"
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "_", team_name.lower()).strip("_")[:48] or "team"
    path = output_dir / f"submission_{slug}_{submission_id}.zip"
    created = False
    try:
        with path.open("xb") as stream:
            created = True
            with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
                for name, data in payloads.items():
                    archive.writestr(name, data)
    except Exception:
        # Delete only a file this call created; never an existing submission.
        if created:
            path.unlink(missing_ok=True)
        raise
    return path, submission_id
