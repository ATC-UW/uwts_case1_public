# Working on UWTS Case 1

This is the participant-facing repository. Keep the workflow small enough for a three-hour research round. Python 3.12 or 3.13 is required by the tested dependency pins.

- `solution.py` is the participant entry point. Run `python solution.py --team "Example"`; it validates data, reports a chronological validation score, predicts, and packages the submission.
- `case1.contract` is the file-format authority: 79 features, nine targets, five-second horizon, string row IDs, numeric second timestamps.
- `case1.scoring.score` is the scoring authority: `1 - sum((y - prediction)^2) / sum(y^2)`. Pool all entries; never substitute centered R² or average target scores.
- Preserve causal inference: preprocessing for validation fits training only; final fitting may use training plus validation labels. Features at time t must not use future rows.
- `case1.packaging` explicitly includes predictions, source, metadata, and optional requirements. Never zip a whole project.
- In the full repository checkout, `organizer/build_release.py` has a strict public-file allowlist. Never add generator code, seeds, latent states, answer keys, or private qualification reports here.
- For repository maintainers only: run `python -m pytest` after consequential toolkit changes. Favor tests for data leakage boundaries, malformed inputs, ID alignment, scoring, and release contents. The participant starter ZIP omits `tests/`, `pyproject.toml`, and `organizer/`; agents working from that ZIP should run the participant solution to validate their changes, not attempt missing development commands.
- Keep the five frozen participant files in `data/` tracked and unchanged. Additional generated data, `submissions/`, and release archives stay out of Git. Do not contact external services or publish changes without the organizer's instruction.

To explore a modeling idea, edit the participant solution; do not change the metric or contract to improve a score. Use ordinary functions and explicit arguments. Avoid unnecessary abstractions and configuration layers.
