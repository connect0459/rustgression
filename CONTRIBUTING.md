# Contributing

## Prerequisites

- [just](https://just.systems/) — command runner
- [rustup](https://rustup.rs/) — Rust toolchain manager
- [uv](https://docs.astral.sh/uv/) — Python package manager

## Setup

```sh
git clone https://github.com/connect0459/rustgression
cd rustgression
just setup
```

`just setup` installs Python dependencies and installs pre-commit hooks.
Run `just build` before first use and after any Rust source changes.

## Development workflow

See [docs/development.md](docs/development.md) for environment setup, test commands, and version management details.

Before opening a pull request, ensure linters and tests pass:

```sh
just lint
just test
```

Pre-commit hooks installed by `just setup` run additional hygiene checks
(trailing whitespace, end-of-file, YAML/TOML validation, markdown lint)
automatically on each `git commit`. To run them across all files manually:

```sh
uv run pre-commit run --all-files
```

## Testing guidelines

This project follows **Red → Green → Refactor** (Detroit-school TDD):

- Write a failing test first, then implement.
- Use real objects; mocks are only permitted at external boundaries.
- Test names describe **what business rule** is verified, not how.

## Commit format

Follow the conventions defined in [docs/COMMIT_CONVENTIONS.md](docs/COMMIT_CONVENTIONS.md).

```text
<type>(<scope>): <subject>
```

**Types**: `feat`, `fix`, `docs`, `style`, `refactor`, `tidy`, `test`, `chore`, `ci`, `perf`

**Scopes**: `rust`, `python`, `deps`, `deps-dev`; omit for project-wide changes.

**Subject**: imperative mood, 72 characters max, no trailing period.

**Never mix Rust and Python in one commit.** Make two commits if a feature requires both.

Examples:

```text
feat(rust): add weighted TLS regression support
fix(python): correct axis label in plot_regression output
test(rust): express slope estimation as a business rule spec
chore(deps-dev): bump pytest from 7.x to 8.x
```

## Pull request process

1. Fork the repository and create a branch: `feature/xxx`, `fix/xxx`, `docs/xxx`.
2. Follow the Red → Green → Refactor cycle.
3. Run `just lint` and `just test`, and ensure all pass.
4. Open a pull request — CI runs Rust and Python test suites automatically.

## Code style

- No code comments unless the **why** is genuinely non-obvious.
- Prefer immutability; avoid mutable state unless necessary.
- All identifiers, test names, error messages, and doc comments must be in **English**.
