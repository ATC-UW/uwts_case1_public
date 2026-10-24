# UWTS Case 1: five-second return forecasting

Predict the next five-second mid-price return for nine anonymized instruments from 79 anonymized market features. Each row is a point in time. Your output is nine decimal returns for that row; for example, `0.001` means a 0.1% return. Different models for different targets are allowed.

## Get started

Use Python **3.12 or 3.13** on an ordinary laptop. Fork or clone this repository, or choose **Code → Download ZIP** and unzip it. Open the resulting folder in your terminal or editor. The repository includes the complete participant dataset in `data/`; no separate data download is needed. If the organizer gives you a starter ZIP instead, unzip it and follow the same steps below.

```sh
python -m venv .venv
```

Activate it with `source .venv/bin/activate` on macOS/Linux, or `.venv\Scripts\Activate.ps1` in Windows PowerShell. Then:

```sh
python -m pip install -r requirements.txt
python solution.py --team "Your team name"
```

The starter trains a regularized linear model, prints validation scores, refits on the labeled data, and writes a submission ZIP into `submissions/`. It uses CPUs only. A successful run prints the ZIP path and its unique submission ID.

Edit `build_model()` in `solution.py` to start researching. You may change the rest of that file as needed; keep your solution self-contained in that one file and use the supplied helpers for validation and packaging. Update `requirements.txt` if you add dependencies. Do not include trained model binaries, datasets, credentials, or additional archives in your submission.

## Data and chronology

| File | Contents |
| --- | --- |
| `data/train.csv` | Earlier features and all nine labels |
| `data/validation.csv` | Later features and all nine labels for model selection |
| `data/test.csv` | Final-period features; labels are private |
| `data/sample_submission.csv` | Required test IDs and nine zero predictions |
| `data/manifest.json` | Dataset version, exact columns, row counts, and checksums |

Feature files contain `row_id`, `timestamp`, and `feature_00` through `feature_78`. Labeled files also contain `target_0` through `target_8`. `timestamp` is an ordered count of seconds, not a calendar date. Rows are spaced one second apart within the supplied periods. `row_id` is an identifier, not a predictive feature. Preserve it exactly as text, including any leading zeroes. Feature scales vary; inspect their distributions before modeling.

Each target is `(mid_price[t + 5] - mid_price[t]) / mid_price[t]`. The raw price paths remain private. Splits are chronological, with rows purged at boundaries so an earlier split's five-second label does not reach the next split. Successive labels within a split overlap in time; use chronological validation, not random row splits.

For a prediction at time `t`, use only features available at or before `t`. The whole test file is supplied for convenience; future rows remain off-limits for that prediction. Do not use negative shifts, centered rolling windows, future interpolation, or fit preprocessing on the test period. Fit preprocessing only on the training data when evaluating validation. After choosing your method, you may refit on training plus validation labels before predicting test. The baseline follows this workflow.

## Official score

The score pools **every row and all nine targets**:

```text
score = 1 - sum((actual - prediction)²) / sum(actual²)
```

This is unweighted, uncentered R², not the default `sklearn.metrics.r2_score`. Perfect predictions score 1, all-zero predictions score 0, and negative scores are valid. Targets with larger squared returns contribute more to the overall score. Per-target scores are diagnostics, not an average used for ranking. Use `case1.scoring.score` for the official calculation.

## Submit

Upload the generated `submission_<team>_<unique-id>.zip` to the organizer's Google Form. Paste the printed **submission ID** into its matching form field and use the same team name each time. The organizer supplies the form link and closing time. Sign into a Google account to upload.

The ZIP contains only:

- `predictions.csv`: `row_id,target_0,…,target_8`, one finite numeric prediction per target per test row.
- `solution.py`: your self-contained solution source, retained for review.
- `submission.json`: format version, dataset ID, team name, and unique submission ID, generated automatically.
- `requirements.txt`: the dependencies used by your solution, included by the starter.

Do not edit or repackage the ZIP after generation. Each run creates a new ID and leaves earlier ZIPs intact. Upload the ZIP you intend to submit. The latest valid submission received before closing counts; a later invalid upload does not replace an earlier valid submission. The form's receipt time is authoritative, not file creation time.

Row order in your CSV may differ from the test file; the judge aligns by ID. Missing, extra, or duplicate IDs, incorrect columns, nonnumeric values, NaN, and infinity are rejected. The starter checks these before packaging. Predictions are judged directly; source code is not run during batch scoring. Keep test scores private until the organizer releases results.

For custom data/output locations:

```sh
python solution.py --team "Your team name" --data /path/to/data --output /path/to/submissions
```

## Toolkit development (repository checkout only)

This section is for maintainers working in the full repository checkout. The participant starter ZIP omits `pyproject.toml`, `tests/`, and `organizer/`; participants do not need these development commands.

The public code has one small module per responsibility: `contract` defines and reads the files, `scoring` computes the metric, `submission` checks predictions, and `packaging` writes an explicitly enumerated archive. No generator internals or test labels belong in this repository.

```sh
python -m pip install -e '.[dev]'
python -m pytest
```

From the repository checkout, the organizer's release script includes only allowlisted participant files and a validated public data export:

```sh
python organizer/build_release.py --data-dir /path/to/public/export --output-dir /path/to/releases --version v1
```

It refuses to overwrite an existing ZIP and writes `release.json` with hashes of its contents. The five frozen participant files in `data/` are tracked in Git. Additional generated data, submissions, and release ZIPs remain untracked. The separate private organizer repository handles data generation and judging.
