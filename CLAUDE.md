# Claude Code Instructions

## Test Commands

### Python Tests

```bash
docker compose exec -w /workspace rustgression-dev uv run pytest
```

### Rust Tests

```bash
docker compose exec -w /workspace rustgression-dev cargo test
```

## Project Information

This is a Rust-Python hybrid project that implements fast Total Least Squares (TLS) regression using a Rust backend with Python bindings.

### Key Files

- `rustgression/__init__.py` - Python package version and imports
- `Cargo.toml` - Rust package configuration
- `pyproject.toml` - Python package configuration
- `scripts/update-version.sh` - Version update script with alpha/beta support

### Version Management

Use the version update script to update all version files consistently:

```bash
./scripts/update-version.sh 0.2.1
./scripts/update-version.sh 0.3.0-alpha.1
```

The script handles different version formats:

- Rust format (Cargo.toml): `0.2.0-alpha.6`
- Python format (`__init__.py`, `pyproject.toml`): `0.2.0a6`
