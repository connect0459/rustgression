# Claude Code Instructions

## Primary Directive

- Think in English. For user interaction language, follow the setting in the user's global CLAUDE.md or CLAUDE.local.md.

## Language Convention

This project may be released publicly. All of the following must be written in **English**:

- Commit messages
- Code comments
- Documentation (including `CLAUDE.md`, `README.md`, etc.)
- Test names
- Error messages

## Commit Conventions

Follow the project commit conventions defined in [`docs/COMMIT_CONVENTIONS.md`](docs/COMMIT_CONVENTIONS.md).

## Test Commands

### Python Tests

```bash
docker compose exec -w /workspace rustgression-dev uv run pytest
```

### Rust Tests

```bash
docker compose exec -w /workspace rustgression-dev cargo test
```

### Rust Coverage

```bash
# Install cargo-llvm-cov (first time only)
docker compose exec -w /workspace rustgression-dev cargo install cargo-llvm-cov

# Generate coverage report (HTML)
docker compose exec -w /workspace rustgression-dev cargo llvm-cov --html

# Generate coverage report (lcov format)
docker compose exec -w /workspace rustgression-dev cargo llvm-cov --lcov --output-path lcov.info
```

## Project Information

This is a Rust-Python hybrid project that implements fast Total Least Squares (TLS) regression using a Rust backend with Python bindings.

### Key Files

- `rustgression/__init__.py` - Python package version and imports
- `Cargo.toml` - Rust package configuration
- `pyproject.toml` - Python package configuration
- `scripts/version-update.sh` - Version update script with alpha/beta support

### Version Management

Use the version update script to update all version files consistently:

```bash
./scripts/version-update.sh 0.2.1
./scripts/version-update.sh 0.3.0-alpha.1
```

Check version consistency across all files:

```bash
./scripts/version-check.sh 0.2.1
```

The update script handles different version formats:

- Rust format (Cargo.toml): `0.2.0-alpha.6`
- Python format (`__init__.py`, `pyproject.toml`): `0.2.0a6`
