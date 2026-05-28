# Contributing

## Prerequisites

- [pre-commit](https://pre-commit.com/) — git hook manager
- [Docker](https://www.docker.com/) *(optional)* — isolated dev container
- Rust toolchain + [uv](https://docs.astral.sh/uv/) *(if not using Docker)*

## Setup

```sh
git clone https://github.com/connect0459/rustgression
cd rustgression
```

**With Docker (recommended):**

```sh
docker compose up -d
```

**Without Docker:** install the Rust toolchain and uv locally, then run commands directly (omit the `docker compose exec -w /workspace rustgression-dev` prefix).

### pre-commit hooks

```sh
pip install pre-commit   # or: brew install pre-commit
pre-commit install
```

To run all hooks manually:

```sh
pre-commit run --all-files
```

## Development workflow

See [`docs/development.md`](docs/development.md) for environment setup, test commands, and version management details.

Before opening a pull request, ensure all hooks and tests pass:

```sh
pre-commit run --all-files
```

## Testing guidelines

This project follows **Red → Green → Refactor** (Detroit-school TDD):

- Write a failing test first, then implement.
- Use real objects; mocks are only permitted at external boundaries.
- Test names describe **what business rule** is verified, not how.

## Commit format

Follow the conventions defined in [`docs/COMMIT_CONVENTIONS.md`](docs/COMMIT_CONVENTIONS.md).

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
3. Run `pre-commit run --all-files` and ensure all tests pass.
4. Open a pull request — CI runs Rust and Python test suites automatically.

## Code style

- No code comments unless the **why** is genuinely non-obvious.
- Prefer immutability; avoid mutable state unless necessary.
- All identifiers, test names, error messages, and doc comments must be in **English**.
