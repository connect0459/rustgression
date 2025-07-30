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
docker compose exec rustgression-dev -w /workspace uv run pytest

# Run linting
docker compose exec rustgression-dev -w /workspace uv run ruff check

# Rebuild the package (after file changes)
docker compose exec rustgression-dev -w /workspace uv run maturin develop
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

# Run commands from outside container
docker compose exec rustgression-container -w /workspace uv run pytest
```
