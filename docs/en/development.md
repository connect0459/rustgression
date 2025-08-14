# Development Environment

## Installing apt Packages

The following dependencies are required to run Rust:

- cmake
- BLAS and LAPACK development libraries
- Fortran compiler

For Debian-based OS, install using the following commands:

```bash
sudo apt-get update
sudo apt-get install cmake libblas-dev liblapack-dev gfortran
```

## Setting Up Python Environment

Place `pyproject.toml` in the project root and install with the following command:

```bash
uv sync
```

## Testing the Created Package

Use `maturin` to perform a clean build from Rust. Install the necessary dependencies and build with the following commands:

```bash
# Confirm installation of necessary tools
uv add maturin
# Clean build
uv run maturin develop
```

## Docker Development Environment

You can set up a development environment using Docker Compose. With volume mounting, file changes on the host are reflected in real-time.

### Starting the Development Environment

```bash
# Start container in background
docker compose up -d

# Or start with logs displayed
docker compose up
```

### Running Commands in the Container

```bash
# Run tests
docker compose exec -w /workspace rustgression-dev uv run pytest

# Run linting
docker compose exec -w /workspace rustgression-dev uv run ruff check

# Rebuild the package (after file changes)
docker compose exec -w /workspace rustgression-dev uv run maturin develop
```

### Development Environment Management

```bash
# Stop container
docker compose stop

# Stop and remove container
docker compose down

# Rebuild image and recreate container
docker compose up --build

# Complete removal including volumes
docker compose down -v
```

## Version Management

### Version Update Procedure

Use the `scripts/version-update.sh` script to manage project versions consistently.

```bash
# Regular version updates (e.g., 0.2.0 → 0.2.1)
docker compose exec -w /workspace rustgression-dev ./scripts/version-update.sh 0.2.1

# Alpha release
docker compose exec -w /workspace rustgression-dev ./scripts/version-update.sh 0.3.0-alpha.1

# Beta release  
docker compose exec -w /workspace rustgression-dev ./scripts/version-update.sh 0.3.0-beta.1
```

This script automatically updates the following files:

- `Cargo.toml` - Rust package version
- `pyproject.toml` - Python package version
- `rustgression/__init__.py` - Package version constant
- `Cargo.lock` - Dependency lock file

### Version Verification

Check version consistency across all files:

```bash
docker compose exec -w /workspace rustgression-dev ./scripts/version-check.sh 0.2.1
```

### Important Notes

- Always run tests after version updates
- Verify version consistency before releases
- Follow Semantic Versioning (SemVer):
  - MAJOR: Breaking changes
  - MINOR: Backward-compatible feature additions
  - PATCH: Backward-compatible bug fixes
