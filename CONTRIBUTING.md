# Contributing

## Prerequisites

- [just](https://just.systems/) — command runner
- [rustup](https://rustup.rs/) — Rust toolchain manager (`rust-toolchain.toml` pins the version automatically)
- [uv](https://docs.astral.sh/uv/) — Python package manager

On Linux, the following system packages are required before running `just setup` (scipy depends on them):

```bash
sudo apt-get update
sudo apt-get install cmake libblas-dev liblapack-dev gfortran
```

On macOS, the Accelerate framework provides these dependencies automatically.

## Setup

After cloning, run once:

```bash
just setup
```

`just setup` installs Python dependencies and installs pre-commit hooks.
Before running tests or importing the package, build the Rust extension:

```bash
just build
```

Run this again whenever Rust source changes.

## Running tests and linters

```bash
# Python tests only
uv run pytest

# Rust tests only
cargo test

# Both
just test

# Rust and Python linters, check-only (matches CI lint gates)
just lint
```

Pre-commit hooks installed by `just setup` run additional hygiene checks
(trailing whitespace, end-of-file, YAML/TOML validation, markdown lint)
automatically on each `git commit`. To run them across all files manually:

```sh
uv run pre-commit run --all-files
```

## Running examples

```bash
just run-examples
```

## Docker (optional)

A Docker environment is available as an alternative to the local setup. It is useful on Linux where
installing BLAS/LAPACK/gfortran locally is inconvenient. The image builds the Rust toolchain and
all system dependencies at image-build time.

Start the container and work from inside:

```bash
# Build and start container
docker compose up -d

# Enter the container
docker compose exec rustgression-dev bash

# Run commands from inside the container (same as local)
just test
just lint
```

To reset the environment:

```bash
# Rebuild image after Dockerfile changes
docker compose up --build -d

# Remove container and volumes (full reset)
docker compose down -v
```

## Version management

Use `just version-update` to update all version files consistently:

```bash
# Regular version update (e.g., 0.2.0 → 0.2.1)
just version-update 0.2.1

# Alpha release
just version-update 0.3.0-alpha.1

# Beta release
just version-update 0.3.0-beta.1
```

This command updates the following files:

- `Cargo.toml` — Rust package version
- `pyproject.toml` — Python package version
- `src-py/rustgression/__init__.py` — Package version constant
- `Cargo.lock` — Rust dependency lock file
- `uv.lock` — Python dependency lock file

Check version consistency across all files:

```bash
just version-check 0.2.1
```

Always run tests after version updates and verify consistency before releases.

## Testing guidelines

This project follows **Red → Green → Refactor** (Detroit-school TDD):

- Write a failing test first, then implement.
- Use real objects; mocks are only permitted at external boundaries.
- Test names describe **what business rule** is verified, not how.

## Commit format

Follow [Conventional Commits](https://www.conventionalcommits.org/). Project-specific additions:

- **Extra type**: `tidy` — small, safe cleanup (< 2 min; no behavior change)
- **Scopes**: use `rust`, `python`, `deps`, or `deps-dev` when the change targets a specific layer; omit for project-wide changes
- **Prefer to separate Rust and Python changes** — mixing is fine when the concern is a single cohesive unit

## Pull request process

1. Fork the repository and create a branch: `feat/xxx`, `fix/xxx`, `docs/xxx`.
2. Follow the Red → Green → Refactor cycle.
3. Run `just lint` and `just test`, and ensure all pass.
4. Open a pull request — CI runs Rust and Python test suites automatically.

## Code style

- No code comments unless the **why** is genuinely non-obvious.
- Prefer immutability; avoid mutable state unless necessary.
- All identifiers, test names, error messages, and doc comments must be in **English**.
