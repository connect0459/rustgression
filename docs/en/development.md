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

You can also set up a development environment using Docker containers.

### Building the Image

```bash
docker build -t rustgression-dev .
```

### Starting the Container

```bash
# Start container in background
docker run -d --name rustgression-container rustgression-dev

# Or start in interactive mode
docker run -it --name rustgression-container rustgression-dev
```

### Running Commands from Outside the Container

```bash
# Run tests
docker exec rustgression-container uv run pytest

# Run linting
docker exec rustgression-container uv run ruff check

# Rebuild the package
docker exec rustgression-container uv run maturin develop

# Enter bash shell
docker exec -it rustgression-container /bin/bash
```

### Container Management

```bash
# Stop container
docker stop rustgression-container

# Start container
docker start rustgression-container

# Remove container
docker rm rustgression-container

# Remove image
docker rmi rustgression-dev
```
