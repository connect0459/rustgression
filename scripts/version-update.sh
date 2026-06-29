#!/bin/bash

# Version update script
# Usage: ./scripts/version-update.sh <new_version>
# Example: ./scripts/version-update.sh 0.2.1

set -e

_VERSION_UTILS="$(dirname "$0")/libs/version-utils.sh"
if [ ! -f "$_VERSION_UTILS" ]; then
    echo "Error: required library not found: $_VERSION_UTILS" >&2
    exit 1
fi
# shellcheck source=libs/version-utils.sh
source "$_VERSION_UTILS"

if [ $# -ne 1 ]; then
    error "Usage: $0 <new_version>"
    echo "Example: $0 0.2.1"
    exit 1
fi

NEW_VERSION=$1

# Check version format (semver format, supports prerelease)
if ! [[ $NEW_VERSION =~ ^[0-9]+\.[0-9]+\.[0-9]+(-alpha\.[0-9]+|-beta\.[0-9]+|-rc\.[0-9]+)?$ ]]; then
    error "Error: Version must be in semver format"
    echo "Examples: 1.0.0, 1.0.0-alpha.1, 1.0.0-beta.2, 1.0.0-rc.1"
    exit 1
fi

# Check for version downgrade by comparing with current Cargo.toml version
CURRENT_CARGO_VERSION=$(grep '^version = ' Cargo.toml | cut -d '"' -f2)
if [ -n "$CURRENT_CARGO_VERSION" ]; then
    if ! compare_versions "$NEW_VERSION" "$CURRENT_CARGO_VERSION"; then
        warning "WARNING: Version downgrade detected!"
        warning "  Current version: $CURRENT_CARGO_VERSION"
        warning "  New version:     $NEW_VERSION"
        echo ""
        read -p "Do you want to continue with the downgrade? (y/N): " confirm
        if [[ ! $confirm =~ ^[Yy]$ ]]; then
            info "Version update cancelled."
            exit 1
        fi
    fi
fi

echo "Updating version to $NEW_VERSION..."

# 1. Update __version__ in src-py/rustgression/__init__.py (convert alpha/beta format)
echo "Updating src-py/rustgression/__init__.py..."
PYTHON_VERSION=$(to_python_version "$NEW_VERSION")
sed -i.bak "s/__version__ = \".*\"/__version__ = \"$PYTHON_VERSION\"/" src-py/rustgression/__init__.py

# 2. Update version in Cargo.toml
echo "Updating Cargo.toml..."
sed -i.bak "s/^version = \".*\"/version = \"$NEW_VERSION\"/" Cargo.toml

# 3. Update version in pyproject.toml (convert alpha/beta format)
echo "Updating pyproject.toml..."
sed -i.bak "s/^version = \".*\"/version = \"$PYTHON_VERSION\"/" pyproject.toml

# 4. Update Cargo.lock (check project with cargo command)
echo "Updating Cargo.lock..."
cargo check --quiet

# 5. Update uv.lock (sync project dependencies with uv command)
echo "Updating uv.lock..."
if command -v uv >/dev/null 2>&1; then
    uv lock
else
    warning "Warning: uv command not found. Skipping uv.lock update."
fi

# Remove backup files
rm -f src-py/rustgression/__init__.py.bak Cargo.toml.bak pyproject.toml.bak

success "SUCCESS: Version update completed: $NEW_VERSION"
echo ""
echo "Updated files:"
echo "- src-py/rustgression/__init__.py (Python format: $PYTHON_VERSION)"
echo "- Cargo.toml (Rust format: $NEW_VERSION)"
echo "- pyproject.toml (Python format: $PYTHON_VERSION)"
echo "- Cargo.lock"
if command -v uv >/dev/null 2>&1; then
    echo "- uv.lock"
fi
echo ""
info "Next steps:"
info "1. Review changes: git diff"
info "2. Run tests: cargo test && uv run pytest"
info "3. Commit changes: git add . && git commit -m \"chore: bump version to $NEW_VERSION\""
