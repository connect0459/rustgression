# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.1] - 2026-05-28

### Miscellaneous

- **docs**: rewrite OLS and TLS descriptions with mathematical precision; remove redundant Usage section
- **docs**: replace relative documentation links with full GitHub URLs for PyPI compatibility
- **docs**: add CI status and MIT license badges to README

## [0.5.0] - 2026-05-28

### Added

- **TlsRegressor**: expose `p_value()`, `stderr()`, and `intercept_stderr()` methods

### Fixed

- **TLS**: remove epsilon filter from singular vector selection for more accurate results
- **TLS**: raise `RuntimeError` when null-vector y-component falls below numerical stability threshold

### Miscellaneous

- **ci**: add Rust test step alongside Python tests in CI pipeline
- **docs**: add `CONTRIBUTING.md` with development guidelines and commit conventions
- **test**: rename tests to describe business rules rather than implementation mechanisms

## [0.4.1] - 2025-09-26

### Fixed

- **ci**: resolve manylinux wheel compatibility issues
- **ci**: fix license metadata for PEP 639 compliance and PyPI upload

## [0.4.0] - 2025-09-05

### Changed

- **Rust**: split regression module into focused sub-modules (`utils`, IEEE 754 handling, statistics)

### Fixed

- **TLS**: improve SVD stability and sign consistency
- **TLS**: improve numerical precision with Kahan summation algorithm

### Miscellaneous

- **deps**: bump nalgebra to 0.34.1 and statrs to 0.18.0
- **ci**: add version check as a separate job

## [0.3.1] - 2025-08-14

### Fixed

- **pyproject.toml**: fix typo in package description

## [0.3.0] - 2025-07-31

### Added

- Multilingual documentation (`docs/en/`, `docs/ja/`)
- GitHub Issue and Pull Request templates
- PyPI download statistics badge

## [0.2.2] - 2025-07-31

### Added

- **scripts**: version validation to `version-update.sh` and `version-check.sh`
- **scripts**: auto-update `tests/test_imports.py` on version bump

## [0.2.1] - 2025-07-31

Initial public release.

## [0.2.0] - 2025-07-31

Initial public release.
