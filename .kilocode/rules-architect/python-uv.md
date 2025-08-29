# Mode-specific: Python (uv-first)

- In this workspace, prefer uv for all Python tasks.
- When proposing scaffolds or commands, default to:
  - `uv init --package`
  - `uv add ...`
  - `uv run ...`
  - `uv lock` / `uv sync`
- Use src/ layout, `[project.scripts]` entry points, and local .venv.
- For multi-package repos, create a uv workspace via `[tool.uv.workspace]` and wire intra-workspace deps with `[tool.uv.sources]`.