<!-- # PULL_REQUEST_TEMPLATE -->

<!-- Remove unnecessary sections to keep the review focused -->

## Related Links

- Issues
  - <!-- <https://github.com/connect0459/rustgression/issues/xxx> -->
- PRs
  - <!-- <https://github.com/connect0459/rustgression/pull/xxx> -->

## [Required] Overview

- Describe the problem being solved, its background, and what changes when this PR is merged.
- Links to specs, design documents, or other references are welcome.

```txt
It is difficult to review without knowing the specifications and background.
```

## Scope of Change

- [ ] Rust core
- [ ] Python bindings
- [ ] Tooling / CI
- [ ] Documentation

## Breaking Changes

- [ ] No breaking changes
- [ ] Breaking changes (describe below)

<!--
If this changes the public API, describe what breaks and why the breakage is justified.
-->

## Deferred Items and TODOs

- Items intentionally deferred and the reasons why.

```txt
If you deferred something due to time constraints, document it here.
Reviewers cannot tell whether something was intentionally skipped or overlooked
without this information.
```

## Test Items

- Describe any test considerations beyond unit tests.
- Note whether both Rust tests (`cargo test`) and Python tests (`uv run pytest`) were validated.

## [Required] Quality Checklist

**Please check all items before merging.**

- [ ] **CI Workflow Execution**: Full quality check completed by manually running `Run workflow` in [Actions](../actions/workflows/ci.yml)
- [ ] **Code Comments**: Code comments and doc-comments are in sync with the changes
- [ ] **Reference Docs**: `docs/api.md` is updated for any public API change
- [ ] **Version Update** (for release PRs): Executed `./scripts/version-update.sh <new-version>` to update all version files consistently

> **Important**: This checklist ensures quality. Please verify all items before requesting review.
