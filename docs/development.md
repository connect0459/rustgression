# Development Environment

## Prerequisites

- [rustup](https://rustup.rs/) — Rust toolchain manager (`rust-toolchain.toml` pins the version automatically)
- [uv](https://docs.astral.sh/uv/) — Python package manager
- [just](https://just.systems/) — command runner

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

This installs Python dependencies and installs pre-commit hooks.
Before running tests or importing the package, build the Rust extension:

```bash
just build
```

Run this again whenever Rust source changes.

## Running Tests and Linters

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

## Running Examples

```bash
just run-examples
```

## Docker (Optional)

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

## Version Management

Use `scripts/version-update.sh` to update all version files consistently:

```bash
# Regular version update (e.g., 0.2.0 → 0.2.1)
./scripts/version-update.sh 0.2.1

# Alpha release
./scripts/version-update.sh 0.3.0-alpha.1

# Beta release
./scripts/version-update.sh 0.3.0-beta.1
```

This script updates the following files:

- `Cargo.toml` — Rust package version
- `pyproject.toml` — Python package version
- `src-py/rustgression/__init__.py` — Package version constant
- `Cargo.lock` — Dependency lock file

Check version consistency across all files:

```bash
./scripts/version-check.sh 0.2.1
```

Always run tests after version updates and verify consistency before releases.
