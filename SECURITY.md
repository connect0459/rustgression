# Security Policy

## Supported Versions

Only the latest minor release receives security fixes. Pre-release versions
(`alpha`, `beta`) are not covered.

## Reporting a Vulnerability

Please use [GitHub's private vulnerability reporting](https://github.com/connect0459/rustgression/security/advisories/new)
to disclose security issues. This keeps the disclosure out of public issues
until a fix is ready.

**What to include:**

- A minimal, self-contained code sample that reproduces the issue
- The version of `rustgression` and the Python / OS environment
- A description of the impact (panic, wrong result, memory unsafety, etc.)

You can expect an acknowledgement within 7 days and a status update within 30 days.

## Scope

The following classes of issues are in scope:

- **Rust panics / unsound FFI** — any input that triggers a Rust `panic!`,
  unwrap failure, or undefined behavior through the PyO3 boundary (e.g.,
  NaN / Inf arrays, zero-length inputs, dimension mismatches that are not
  caught before the Rust layer executes).
- **Incorrect numerical results** — silent data corruption or wrong
  regression output (OLS / TLS) that violates documented numerical guarantees
  in a security-relevant context.
- **Dependency vulnerabilities** — security advisories in upstream Rust
  crates (`cargo audit`) or Python dependencies that affect `rustgression`
  users.

Pure performance regressions and feature requests are out of scope for
security reports; please open a regular issue instead.
