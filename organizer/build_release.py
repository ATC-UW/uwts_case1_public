"""Assemble a participant ZIP using an explicit public-file allowlist."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import zipfile

# Supports running the script directly from a fresh checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from case1.contract import DATA_FILES, ValidationError, validate_dataset

SOURCE_FILES = (
    "README.md", "AGENTS.md", "solution.py", "requirements.txt",
    "case1/__init__.py", "case1/contract.py", "case1/scoring.py",
    "case1/submission.py", "case1/packaging.py",
)
MANIFEST_FIELDS = {
    "format_version", "dataset_id", "horizon_seconds", "id_column", "time_column",
    "feature_columns", "target_columns", "files",
}


def build_release(data_dir: Path, output_dir: Path, *, version: str = "v1",
                  source_dir: Path | None = None) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,39}", version):
        raise ValidationError("Release version must contain 1–40 letters, digits, underscores, or hyphens.")
    source_dir = source_dir or Path(__file__).resolve().parents[1]
    manifest = validate_dataset(data_dir)
    # Prevent an accidental private metadata field from entering a release.
    if set(manifest) != MANIFEST_FIELDS:
        raise ValidationError("Public dataset manifest contains missing or unexpected fields.")
    for info in manifest["files"].values():
        if set(info) != {"sha256", "rows"}:
            raise ValidationError("Public file metadata may contain only sha256 and rows.")
    payloads = {}
    for relative in SOURCE_FILES:
        source = source_dir / relative
        if source.is_symlink() or not source.is_file():
            raise ValidationError(f"Public source file is missing or is a symbolic link: {relative}.")
        payloads[relative] = source.read_bytes()
    for name in (*DATA_FILES, "manifest.json"):
        source = data_dir / name
        if source.is_symlink() or not source.is_file():
            raise ValidationError(f"Dataset file is missing or is a symbolic link: {name}.")
        payloads[f"data/{name}"] = source.read_bytes()
    release_record = {
        "version": version, "dataset_id": manifest["dataset_id"],
        "files": {name: hashlib.sha256(data).hexdigest() for name, data in sorted(payloads.items())},
    }
    payloads["release.json"] = (json.dumps(release_record, indent=2) + "\n").encode()
    output_dir.mkdir(parents=True, exist_ok=True)
    root_name = f"uwts_case1_starter_{version}"
    archive_path = output_dir / f"{root_name}.zip"
    with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, data in sorted(payloads.items()):
            entry = zipfile.ZipInfo(f"{root_name}/{name}", date_time=(2026, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o100644 << 16
            archive.writestr(entry, data)
    return archive_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Generator's public export directory.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--version", default="v1")
    args = parser.parse_args()
    try:
        path = build_release(args.data_dir, args.output_dir, version=args.version)
    except (ValidationError, OSError) as exc:
        parser.exit(2, f"Release failed: {exc}\n")
    print(path.resolve())


if __name__ == "__main__":
    main()
