"""Edit build_model() to research your strategy; the surrounding workflow is ready."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from case1.contract import FEATURE_COLUMNS, ID_COLUMN, TARGET_COLUMNS, ValidationError, load_dataset
from case1.packaging import create_submission
from case1.scoring import score, scores_by_target


def build_model():
    """Return an estimator with fit(X, y) and predict(X); y has nine target columns.

    This baseline uses the current row's features. Fit preprocessing inside the
    training pipeline. For lagged features, only use observations at or before t.
    Separate models per target are also allowed.
    """
    return make_pipeline(StandardScaler(), Ridge(alpha=100.0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train, validate, and package a UWTS Case 1 submission.")
    parser.add_argument("--team", required=True, help="Use the same team name on every submission.")
    parser.add_argument("--data", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "submissions")
    args = parser.parse_args()
    try:
        dataset = load_dataset(args.data)
        train, validation, test = dataset.train, dataset.validation, dataset.test
        model = build_model()
        model.fit(train[FEATURE_COLUMNS], train[TARGET_COLUMNS])
        validation_predictions = model.predict(validation[FEATURE_COLUMNS])
        actual = validation[TARGET_COLUMNS].to_numpy()
        print(f"Validation score: {score(actual, validation_predictions):.6f}")
        for target, value in scores_by_target(actual, validation_predictions).items():
            print(f"  {target}: {value:.6f}")

        # Labels from both released periods are known before the test period.
        # A fresh pipeline refits preprocessing using these labeled periods only.
        final_model = build_model()
        labeled = pd.concat([train, validation], ignore_index=True)
        final_model.fit(labeled[FEATURE_COLUMNS], labeled[TARGET_COLUMNS])
        predictions = pd.DataFrame(final_model.predict(test[FEATURE_COLUMNS]), columns=TARGET_COLUMNS)
        predictions.insert(0, ID_COLUMN, test[ID_COLUMN].to_numpy())
        path, submission_id = create_submission(
            predictions, test[ID_COLUMN], team_name=args.team, dataset_id=dataset.dataset_id,
            solution_path=Path(__file__), output_dir=args.output,
            requirements_path=Path(__file__).with_name("requirements.txt"),
        )
    except (ValidationError, OSError) as exc:
        parser.exit(2, f"Could not create a submission: {exc}\n")
    print(f"\nReady to upload: {path.resolve()}")
    print(f"Submission ID: {submission_id}")
    print("Upload this ZIP to the organizer's form and paste the submission ID into its matching field.")
    print("Keep the ZIP unchanged. A new run creates a new submission ID.")


if __name__ == "__main__":
    main()
