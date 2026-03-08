# ADR-012: Standardization of GitHub Templates and Documentation Structure

## Status

- [x] Proposed
- [x] Accepted
- [ ] Deprecated

## Context

The rustgression project required improvements to contribution quality and internationalization of documentation. The situation was as follows:

### Current Problems

1. **Absence of Issue/PR templates**: Without a standardized format, contribution quality was inconsistent.
2. **Insufficient documentation internationalization**: Only developer-facing documentation had English/Japanese support; user-facing documentation was incomplete.
3. **Lack of project visibility**: Project statistics such as PyPI download counts were not displayed.
4. **Unclear contribution guidelines**: Information that new contributors needed to reference was scattered.

## Decision

Standardize GitHub templates and documentation structure to improve project visibility and contribution quality.

### 1. Introduce GitHub Templates

#### Issue Templates

- `BUG_REPORT.md`: Template for bug reports
- `DEVELOP_REQUEST.md`: Template for feature development and improvement requests (later renamed to `FEATURE_REQUEST.md`)

#### Pull Request Template

- `PULL_REQUEST_TEMPLATE.md`: Comprehensive PR review template
  - Related URLs, summary, release information
  - Target devices/environments, test items
  - Quality checklist, work time tracking

### 2. Multilingual Documentation Structure

```txt
docs/
├── adrs/           # Architecture Decision Records
├── en/             # English documentation
│   ├── README.md   # English user guide
│   └── development.md  # English developer guide
└── ja/             # Japanese documentation
    ├── README.md   # Japanese user guide
    └── development.md  # Japanese developer guide
```

### 3. Improve Project Visibility

- Add pepy.tech badge to display PyPI download statistics
- Organize and add important project URLs

### 4. Documentation Design Principles

#### User-facing documentation (`docs/en/README.md`, `docs/ja/README.md`)

- Installation instructions
- Basic usage examples
- API reference overview
- Clear links to developer documentation

#### Developer-facing documentation (existing `development.md`)

- Development environment setup
- Contribution guidelines
- Architecture details

## Consequences

### Expected Benefits

1. **Improved contribution quality**
   - Consistent Issue/PR creation through standardized templates
   - More efficient review process

2. **Improved project visibility**
   - Download statistics displayed via pepy badge
   - Visualization of project adoption

3. **Strengthened internationalization**
   - Documentation provided in both English and Japanese
   - Improved accessibility for global users

4. **Lower barrier for new users and contributors**
   - Clear documentation structure
   - Gradual learning path (user → developer)

### Implementation Details

#### Template Language

- Unified in English (following the standard for international OSS projects)
- Japanese comments removed and translated to English

#### Quality Assurance

- Include quality checklist in PR template
- Integration with CI/CD workflows

#### Gradual Migration

- Maintain compatibility with existing documentation
- Add appropriate links to new documentation

## References

- [GitHub Issue Templates](https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests/about-issue-and-pull-request-templates)
- [pepy - PyPI download statistics](https://pepy.tech/)
- [Open Source Guide - Best Practices for Maintainers](https://opensource.guide/best-practices/)

## Related File Paths

### Initial implementation (2025-07-31)

#### GitHub Templates

- `.github/ISSUE_TEMPLATE/BUG_REPORT.md` (new)
- `.github/ISSUE_TEMPLATE/DEVELOP_REQUEST.md` (new, later renamed to FEATURE_REQUEST)
- `.github/PULL_REQUEST_TEMPLATE.md` (new)

#### Documentation Structure

- `docs/en/README.md` (new)
- `docs/ja/README.md` (new)
- `README.md` (add pepy badge and URLs)

#### ADR

- `docs/adrs/adr-012-github-templates-and-documentation-structure.md` (new)
