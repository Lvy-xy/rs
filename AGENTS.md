# Repository Guidelines

## Project Structure & Module Organization
- Root currently holds the trained YOLO weights (`model/yolo_rs.pt`) and a reference output image (`Generated_Image.png`).
- Add code that consumes or updates the model under a new `src/` directory; place notebooks under `notebooks/`; keep evaluation artifacts in `reports/` to avoid cluttering the root.
- Store large artifacts only in `model/` and document provenance (training data, hyperparams) alongside the file in a short `README` within the same folder.

## Build, Test, and Development Commands
- No build pipeline is defined yet. If you add Python code, prefer a virtual environment and a simple workflow: `python -m venv .venv && .\.venv\Scripts\activate` then `python -m pip install -e .[dev]` once a `setup.cfg`/`pyproject.toml` exists.
- When tests are present, run `python -m pytest` from the repo root; add `-q` for quicker feedback.
- For lint/format, adopt `ruff check` and `ruff format` once a `ruff.toml` is added; keep configuration in version control.

## Coding Style & Naming Conventions
- Python: PEP 8 with 4-space indentation; prefer type hints and `dataclass`es for structured data; function names snake_case, classes PascalCase, constants UPPER_SNAKE.
- Keep scripts small and composable; factor shared logic into modules under `src/`.
- Persist model checkpoints with semantic version tags (e.g., `yolo_rs_v1.1.pt`) and update any consumer scripts accordingly.

## Testing Guidelines
- Use `pytest` for unit/integration coverage; name test files `test_*.py` and mirror the `src/` layout.
- For model updates, include a lightweight regression check (e.g., comparing sample detections) and log expected outputs in fixtures so diffs are reviewable.
- Document any external data dependencies or random seeds in the test docstrings to keep runs reproducible.

## Commit & Pull Request Guidelines
- Commit messages: short imperative subject (`Add inference wrapper`), optional body for rationale and edge cases.
- Pull requests should describe model changes, data used, evaluation metrics, and sample commands to reproduce results. Attach before/after images (like `Generated_Image.png`) when behavior changes.
- Link issues or tasks, note breaking changes clearly, and request review when tests (if present) are green.

## Security & Configuration Tips
- Do not commit private datasets, API keys, or training secrets. Use environment variables and a non-tracked `.env.example` for expected keys.
- Large binaries should stay in `model/`; consider Git LFS if size grows. Validate downloads with checksums when sharing weights.
