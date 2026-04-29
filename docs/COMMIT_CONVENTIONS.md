# Commit Conventions

## Format

```text
<type>(<scope>): <subject>

<body>

<footer>
```

## Types

| Type | Description |
| :--- | :--- |
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `style` | Code style (formatting, whitespace) |
| `refactor` | Code change that is neither a fix nor a feature |
| `tidy` | Small, safe cleanup (< 2 min; no behavior change) |
| `test` | Adding or updating tests |
| `chore` | Build process, tooling, config, CI/CD pipeline changes |
| `perf` | Performance improvement |

## Scopes

Scope is **required** when the change targets a specific layer; omit it only for project-wide changes (e.g., `docs: update README`).

| Scope | When to use |
| :--- | :--- |
| `python` | Python changes |
| `rust` | Rust changes |
| `deps` | Runtime dependency updates (`Cargo.toml`, `pyproject.toml`) |
| `deps-dev` | Dev-only dependency updates |

## Rules

### Never mix Rust and Python in one commit

Changes to the Rust core and Python bindings must be committed separately.
If a feature requires both, make two commits: one with `(rust)` and one with `(python)`.

### Subject line

- Use the imperative mood: "add", "fix", "remove" — not "added" or "adds"
- 72 characters max
- No trailing period

### Body (optional)

- Wrap at 72 characters
- Explain **why**, not what — the diff already shows what changed
- Leave one blank line between subject and body

### Footer (optional)

- `BREAKING CHANGE: <description>` for breaking changes
- `Closes #123` or `Fixes #456` to link issues

## Examples

```text
feat(rust): add weighted TLS regression support

Weighted regression allows callers to down-weight noisy measurements.
This avoids adding a separate API endpoint for the common use case.

Closes #42
```

```text
fix(python): correct axis label in plot_regression output
```

```text
test(rust): express slope estimation as a business rule spec

Renamed from numeric-assertion tests to describe what statistical behavior
is being verified, following the Evergreen test naming principle.
```

```text
chore(deps-dev): bump pytest from 7.x to 8.x
```

```text
docs: add commit conventions guide
```
