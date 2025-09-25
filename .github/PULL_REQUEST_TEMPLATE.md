※Remove unnecessary sections to make review easier

## Related URLs

- Related Issues
- URLs to accessible screens, etc.

## [Required] Overview

- Describe the problem you want to solve, its background, and how things will change when this PR is released.
- Links to wiki, tickets, or project overview documents are also acceptable.

```txt
It's difficult to review without knowing the specifications and background.
```

## [Required] Release

- Release date
  - Please provide an approximate timeline even if not finalized.
- Release considerations
  - Dependencies on other PRs being released first, etc.

## Target Devices/Environments

- [ ] PC (Windows, Mac)
  - [ ] Chrome
  - [ ] Firefox
  - [ ] Edge
  - [ ] Safari
- [ ] Mobile (Android, iPhone)
  - [ ] Chrome
  - [ ] Safari

```txt
Check devices/environments that may be affected and should be tested.
This helps reviewers determine whether device-specific bugs should be addressed
or can be ignored because they're not target devices.
```

## Related Database

- Related tables

## External Integrations

- Links to partner API specifications if applicable.

```txt
For external integration cases, integration specifications are necessary for review.
```

## Deferred Items and TODOs

- Items that were deferred and the reasons why.

```txt
If you intentionally deferred something due to time constraints, please document it.
If reviewers can't tell whether something was intentionally deferred or overlooked,
they may make unnecessary comments.
```

## Test Items

- Describe any tests beyond unit tests. This should be documented before testing begins.
- Consider what you would test in production after release.

## Release Preparation

- Document any special release procedures if applicable.

## Post-Release Verification

- Post-release verification steps
- Rollback procedures if things aren't working correctly

```txt
Describe what conditions indicate a successful release completion.
If you can verify that the production implementation works correctly,
the same items as test items are fine.
Also document rollback procedures in case of issues, e.g., Revert.
```

## Areas That Need Special Review Attention

- Design decisions you're struggling with or struggled with.
- Areas you're somewhat concerned about.
- Feel free to highlight parts you think went well.

## ✅ Quality Checklist (Required)

### Please check all items before merging

- [ ] **CI Workflow Execution**: Full quality check completed by manually running `Run workflow` in [Actions](../actions/workflows/test-and-build.yml)
- [ ] Coverage is maintained (target 80%+)
- [ ] Design decisions documented in ADR (if applicable)
- [ ] **Version Update** (for release PRs): Executed `./scripts/version-update.sh <new-version>` to update all version files consistently

> 💡 **Important**: Since this is a private repository, this checklist ensures quality. Please verify all items before requesting review.

## Work Time

- Estimate
  - xx hours
- Actual time spent
  - xx hours
- Reason for time difference (over/under estimate)
  - More work than anticipated, etc.

```txt
Please record approximate work time estimates and actuals until PR closure.
This helps with retrospectives and similar future work.
```
