
# CC Group Joker Engine — Python Tooling Rules (uv-first)

Scope
- These rules apply at the workspace level for all Python work in this repo (CC Group Joker Engine and related ML utilities).
- Kilo must follow these rules when creating, modifying, or running Python code, scripts, configs, and tooling.

References (for Kilo; no action required)
- Kilo Custom Rules (project rules live in .kilocode/rules; mode-specific allowed): https://kilocode.ai/docs/advanced-usage/custom-rules
- uv (Astral) projects, lockfiles, workspaces, tools:
    - Projects: https://docs.astral.sh/uv/guides/projects/
    - Features: https://docs.astral.sh/uv/getting-started/features/
    - Workspaces: https://docs.astral.sh/uv/concepts/projects/workspaces/
    - Scripts: https://docs.astral.sh/uv/guides/scripts/

## 1) Package and environment management
- Use uv for everything (init, add/remove deps, run, lock, sync, build, publish). Do not use pip/poetry/conda/virtualenv directly.
- Always create projects with `uv init` (or `uv init --package` for installable packages).
- Enforce a project-local virtual environment at ./.venv (uv default). Never check .venv into git.
- Manage dependencies exclusively via:
    - `uv add PKG` / `uv remove PKG`
    - `uv lock` / `uv sync`
    - `uv run` for commands in the project env
- Check in uv.lock. Do not edit uv.lock manually.
- If a single repo contains multiple packages, use a uv workspace:
    - At repo root pyproject.toml, add:
    ```
    [tool.uv.workspace]
    members = ["packages/*", "apps/*"]
    ```
    - Prefer `[tool.uv.sources] foo = { workspace = true }` for intra-workspace deps.

## 2) Python version and reproducibility
- Pin a workspace-wide Python version via `.python-version` (e.g., 3.12) and expose it in `pyproject.toml`:
```
[project]
requires-python = ">=3.12,<3.13"
```
- Prefer `uv python install 3.12` and `uv python pin 3.12` if needed.
- For scripts, prefer inline script metadata with uv (PEP 723). Use `uv run --script` shebangs when useful.
- Ensure CI uses `uv run` (which checks lockfile/env) to avoid drift.

## 3) Project layout (single-package)
- New ML project (cc-joker-engine) layout:
```
cc-joker-engine/
├── pyproject.toml
├── README.md
├── .python-version
├── src/ccjoker/
│   ├── init.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── eval.py
│   └── utils.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── models/
├── notebooks/
└── tests/
```
- If you later split into multiple packages, move libraries under packages/* and apps under apps/* and enable a uv workspace at the root.

## 4) Naming and packaging
- Top-level package: `ccjoker` (src layout).
- CLI entry points via `[project.scripts]` in pyproject.toml, e.g.:
```
[project.scripts]
ccj-train = "ccjoker.train:main"
ccj-eval  = "ccjoker.eval:main"
```

- Keep import-safe names (no hyphens in package modules).

## 5) Dependencies
- Baseline stack (add with `uv add`):
- torch (CPU baseline; CUDA variants added per machine as needed)
- numpy, scipy
- pydantic (config/schema), pyyaml or tomli for configs
- rich and tqdm for logs/progress
- orjson / ujson for fast JSON
- typer (CLI) or click
- pytest + pytest-cov for testing
- Optional:
- lightning, optuna (tuning), onnxruntime / torch.compile for inference
- Lock with `uv lock` and commit uv.lock.

## 6) Code style and quality
- Linters/formatters:
- Ruff for lint + format (preferred), or Black + Ruff if you insist.
- mypy for typing (incremental adoption allowed).
- Add scripts to pyproject.toml where convenient:
```
[tool.ruff]
line-length = 100

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
```

- Kilo must run linters/formatters via `uv run ruff format` and `uv run ruff check`.

## 7) Data handling
- Do not commit large datasets; use data/raw/.gitignore. Small sample JSONL may live under tests/fixtures.
- Exported JSONL from Triplecargo goes in data/raw/.
- Preprocessed tensors go in data/processed/.
- Models in data/models/.

## 8) Execution, scripts, and tools
- Use `uv run` for all Python execution within the project env.
- For ad-hoc tools, prefer `uvx` (uv tool runner) over global pipx.
- If making a utility CLI, support `uv tool install -e .` for local editable installs.
- For notebooks, use the project kernel (point to .venv).

## 9) CI/CD
- CI should:
- Set up uv (curl install or package).
- `uv python install 3.12 && uv sync`
- `uv run pytest`
- `uv run ruff check` (and optionally `ruff format --check`)
- Caches:
- Cache ~/.cache/uv and the workspace .venv if beneficial.

## 10) Kilo operational rules
- When Kilo creates Python code or scaffolding:
- Default to `uv init --package` (src layout), add deps with `uv add`.
- Provide runnable commands with `uv run ...`.
- Update pyproject.toml (project metadata, scripts, optional tool configs).
- Never suggest `pip install` / `virtualenv` / `conda` in this repo.
- When Kilo needs to add a CLI:
- Use `[project.scripts]` and demonstrate `uv run <script-name>` and `uv tool install -e .`.
- When Kilo needs to split into multiple subpackages:
- Create uv workspace at root with `[tool.uv.workspace]` members.
- Move shared libs to `packages/`, apps to `apps/`, and wire `[tool.uv.sources]`.

## 11) Security and secrets
- Never read or write real secrets. Respect .env.example with dummy values.
- Do not commit `.venv`, data/raw large files, or secrets.

## 12) Documentation
- Keep README.md updated with:
- `uv` quickstart (init, add, run).
- Data directory expectations.
- How to train/evaluate with `uv run`.
- Kilo should add usage snippets using `uv run`.