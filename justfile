# First-time setup: install deps and install pre-commit hook
setup:
    uv sync --frozen
    uv run pre-commit install

# Run Rust and Python linters (check-only; matches CI lint gates)
lint:
    cargo fmt --all --check
    cargo clippy --all-targets --all-features -- -D warnings
    uv run ruff check
    uv run ruff format --check

# Run Rust and Python formatting
fmt:
    cargo fmt
    uv run ruff format
    uv run ruff check --fix

# Build the Rust extension
build:
    uv run maturin develop

# Run Rust and Python tests (rebuilds extension first to avoid stale bindings)
test: build
    cargo test
    uv run pytest

# Run example scripts
run-examples: build
    uv run --extra examples python examples/simple_example.py
    uv run --extra examples python examples/scientific_example.py

# Check version consistency across all files
version-check VERSION:
    ./scripts/version-check.sh {{VERSION}}

# Update version across all files (Cargo.toml, pyproject.toml, __init__.py, lock files)
version-update VERSION:
    ./scripts/version-update.sh {{VERSION}}
