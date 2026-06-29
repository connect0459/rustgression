# Changelog

<!--
When cutting a new release, update THREE places in this file:

1. Rename [Unreleased] to [X.Y.Z] with today's date (above).
2. Update the reference links at the very bottom of this file:
    - Change [Unreleased] to compare the new tag against HEAD.
    - Add [X.Y.Z] comparing the new tag against the previous tag.
3. After the PR is merged, create a GitHub Release (this creates the remote
   tag). Pull main first so HEAD is the merge commit, then use `--target main`
   or pass the full 40-character SHA — the GitHub API rejects abbreviated SHAs:

    ```console
    git checkout main && git pull origin main
    gh release create vX.Y.Z --title "vX.Y.Z" \
      --notes-file path/to/gh-release-draft.md \
      --target main
    ```

see: <https://github.com/connect0459/rustgression/pull/200>
-->

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.6.0] - 2026-06-29

### Added

- **OlsMultiRegressor**: new class for multi-predictor OLS regression backed by a Rust
  implementation (`calculate_ols_multi_regression`); supports `predict()`, `r_squared()`,
  `f_statistic()`, `p_value()`, and `get_params()`; also accessible via `create_regressor(x, y, method="ols_multi")`
- **BaseRegressor**: `r_squared()` — returns the coefficient of determination (R²)
- **BaseRegressor**: `residuals()` — returns vertical residuals `y − ŷ`
- **OlsRegressor**: `confidence_interval(alpha)` — t-based confidence intervals for slope and intercept
- **OlsRegressor**: `prediction_interval(x_new, alpha)` — t-based prediction intervals for new observations
  (`TlsRegressor` raises `NotImplementedError` for both; TLS bootstrap intervals are deferred)
- **NumericalWarning**: new warning type emitted when subnormal input values are detected;
  exported from the package public API and routed through Python's `warnings` module

### Fixed

- **TLS**: replace OLS stderr formula with the Deming orthogonal-regression formula for more
  accurate slope and intercept standard errors
- **OLS**: fix product underflow in `r_value()` for sub-epsilon x variance using a scale-aware
  `compute_r_value` approach
- **OLS**: add overflow guard for intercept on extreme-scale inputs for consistency with slope
- **TlsRegressor**: `.pyi` stub now correctly declares `NoReturn` on `confidence_interval` and
  `prediction_interval`
- **deps**: add `scipy` to runtime dependencies (required by `confidence_interval` and
  `prediction_interval` but was previously missing)
- **`__init__.pyi`**: add missing exports (`NumericalWarning`, `OlsMultiRegressor`,
  `OlsMultiRegressionParams`) and `"ols_multi"` literal to `create_regressor` overload;
  resolves type errors for mypy/pyright users (#198)

### Changed

- **src layout**: Python package moved from `rustgression/` to `src-py/rustgression/`
- **test layout**: tests reorganized under `tests-py/integration/` and `tests-py/e2e/` tiers
- **NumericalWarning**: numerical diagnostics previously emitted via `eprintln!` are now
  routed through Python's `warnings` module so callers can suppress or escalate them

### Miscellaneous

- **ci**: improve paths-filter entries and clean up redundant workflow steps
- **dev**: simplify Docker dev environment to a local-first workflow
- **chore**: add `just version-update` and `just version-check` recipes; include `CHANGELOG.md` in source distribution
- **docs**: consolidate development and commit guidelines into `CONTRIBUTING.md`
- **chore**: remove redundant `scipy` from `optional-dependencies.examples` after its
  promotion to core runtime dependency (#198)

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

---

[Unreleased]: <https://github.com/connect0459/rustgression/compare/v0.6.0...HEAD>
[0.6.0]: <https://github.com/connect0459/rustgression/compare/v0.5.1...v0.6.0>
[0.5.1]: <https://github.com/connect0459/rustgression/compare/v0.5.0...v0.5.1>
[0.5.0]: <https://github.com/connect0459/rustgression/compare/v0.4.1...v0.5.0>
[0.4.1]: <https://github.com/connect0459/rustgression/compare/v0.4.0...v0.4.1>
[0.4.0]: <https://github.com/connect0459/rustgression/compare/v0.3.1...v0.4.0>
[0.3.1]: <https://github.com/connect0459/rustgression/compare/v0.3.0...v0.3.1>
[0.3.0]: <https://github.com/connect0459/rustgression/compare/v0.2.2...v0.3.0>
[0.2.2]: <https://github.com/connect0459/rustgression/compare/v0.2.1...v0.2.2>
[0.2.1]: <https://github.com/connect0459/rustgression/compare/v0.2.0...v0.2.1>
[0.2.0]: <https://github.com/connect0459/rustgression/releases/tag/v0.2.0>
