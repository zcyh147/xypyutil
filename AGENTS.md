# AGENTS.md

## Overview

`xypyutil` is a flat Python utility library built around one main module: `xypyutil.py`.
It aims to provide thin, low-dependency wrappers for common sysadmin and cross-platform tasks.

## Repository Layout

- `xypyutil.py` — main library module and primary edit surface
- `xypyutil_helper/` — packaged helper assets that ship with the library
- `xypyutil_helper/windows/` — Windows-specific helper scripts
- `test/test_default.py` — main regression test file
- `test/_org/` — input fixtures and sample source material for tests
- `test/_ref/` — expected/reference outputs for tests
- `pyproject.toml` — Poetry metadata, build config, and dev dependencies
- `README.md` — short project overview and install snippet
- `dist/` — built artifacts; not a source directory

## Working Rules

- Treat `xypyutil.py` as the canonical implementation surface. This repo does not use a `src/` layout or a multi-package structure.
- Preserve the project's low-dependency direction. The README explicitly says to avoid third-party runtime dependencies.
- Keep helper assets in `xypyutil_helper/` aligned with packaging rules in `pyproject.toml`.
- Do not treat `dist/`, `.venv/`, or ignored generated files as source-of-truth.

## Testing Conventions

- Tests live in `test/`, not `tests/`.
- The test suite imports the project directly from the repository root by modifying `sys.path` in `test/test_default.py`.
- `test/_org/` contains source fixtures and sample files.
- `test/_ref/` contains expected outputs used for assertions.
- `test/_gen/` is generated during testing and is gitignored.
- `test/skip_slow_tests.cfg.txt` is an optional local toggle for skipping slow tests and is gitignored.

## Tooling Present

- Packaging/build: Poetry via `pyproject.toml`
- Supported Python: `^3.9`
- Dev dependencies explicitly declared: `pytest`, `coverage`
- Poetry is configured for an in-project virtualenv via `poetry.toml` (`.venv/` at repo root)

## Tooling Not Present

Do not assume repo-managed automation exists for any of the following unless you add it explicitly:

- GitHub Actions / CI workflows
- Ruff, Black, Flake8, Mypy, Pyright, tox, nox, or pre-commit config
- A documented console-script entry point
- A docs or examples tree

## Editing Guidance

- Prefer small, targeted edits. This repo is centralized in one large module, so unrelated cleanup can create unnecessary risk.
- Match the existing style and naming patterns in nearby code instead of introducing a new structure.
- Be careful with cross-platform helpers; check existing platform branches before changing behavior.
- Do not casually change newline, path, or lazy helper behavior when code comments mark them as intentional.
- When touching helper scripts or packaging-related paths, confirm they are still included correctly in `pyproject.toml`.

## Validation

Use the simplest repo-local validation that matches the change:

- Run targeted `pytest` coverage for affected behavior when practical.
- If packaging changes are involved, verify `pyproject.toml` still reflects shipped assets.
- Ignore diagnostics from `.venv/` and non-source generated artifacts when assessing your own changes.

## Notes for Agents

- The most important architectural fact in this repo is its flat structure: one main Python module plus packaged helper assets.
- If a change looks like it deserves a new submodule, confirm that intent first instead of silently restructuring the project.
