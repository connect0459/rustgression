#!/bin/bash

# Version update script
# Usage: ./scripts/version-update.sh <new_version>
# Example: ./scripts/version-update.sh 0.2.1

set -e

if [ $# -ne 1 ]; then
    echo "Usage: $0 <new_version>"
    echo "Example: $0 0.2.1"
    exit 1
fi

NEW_VERSION=$1

# Function to compare semantic versions (returns 0 if v1 >= v2, 1 if v1 < v2)
compare_versions() {
    local v1=$1
    local v2=$2

    # Extract version parts (major.minor.patch) and prerelease
    local v1_base=$(echo "$v1" | cut -d'-' -f1)
    local v2_base=$(echo "$v2" | cut -d'-' -f1)
    local v1_pre=$(echo "$v1" | cut -d'-' -f2- | sed 's/^'$v1_base'$//')
    local v2_pre=$(echo "$v2" | cut -d'-' -f2- | sed 's/^'$v2_base'$//')

    # Split version parts
    IFS='.' read -r v1_major v1_minor v1_patch <<< "$v1_base"
    IFS='.' read -r v2_major v2_minor v2_patch <<< "$v2_base"

    # Compare major.minor.patch
    if [ "$v1_major" -gt "$v2_major" ]; then return 0; fi
    if [ "$v1_major" -lt "$v2_major" ]; then return 1; fi
    if [ "$v1_minor" -gt "$v2_minor" ]; then return 0; fi
    if [ "$v1_minor" -lt "$v2_minor" ]; then return 1; fi
    if [ "$v1_patch" -gt "$v2_patch" ]; then return 0; fi
    if [ "$v1_patch" -lt "$v2_patch" ]; then return 1; fi

    # If base versions are equal, compare prerelease
    # No prerelease (stable) > prerelease
    if [ -z "$v1_pre" ] && [ -n "$v2_pre" ]; then return 0; fi
    if [ -n "$v1_pre" ] && [ -z "$v2_pre" ]; then return 1; fi

    # Both have prerelease or both stable
    if [ -z "$v1_pre" ] && [ -z "$v2_pre" ]; then return 0; fi  # Equal

    # Compare prerelease types and numbers
    # alpha < beta < rc
    local v1_pre_type=$(echo "$v1_pre" | cut -d'.' -f1)
    local v2_pre_type=$(echo "$v2_pre" | cut -d'.' -f1)
    local v1_pre_num=$(echo "$v1_pre" | cut -d'.' -f2)
    local v2_pre_num=$(echo "$v2_pre" | cut -d'.' -f2)

    if [ "$v1_pre_type" != "$v2_pre_type" ]; then
        case "$v1_pre_type-$v2_pre_type" in
            alpha-beta|alpha-rc|beta-rc) return 1 ;;
            beta-alpha|rc-alpha|rc-beta) return 0 ;;
        esac
    fi

    # Same prerelease type, compare numbers
    if [ "$v1_pre_num" -ge "$v2_pre_num" ]; then return 0; else return 1; fi
}

# Check version format (semver format, supports prerelease)
if ! [[ $NEW_VERSION =~ ^[0-9]+\.[0-9]+\.[0-9]+(-alpha\.[0-9]+|-beta\.[0-9]+|-rc\.[0-9]+)?$ ]]; then
    echo "Error: Version must be in semver format"
    echo "Examples: 1.0.0, 1.0.0-alpha.1, 1.0.0-beta.2, 1.0.0-rc.1"
    exit 1
fi

# Check for version downgrade by comparing with current Cargo.toml version
CURRENT_CARGO_VERSION=$(grep '^version = ' Cargo.toml | cut -d '"' -f2)
if [ -n "$CURRENT_CARGO_VERSION" ]; then
    if ! compare_versions "$NEW_VERSION" "$CURRENT_CARGO_VERSION"; then
        echo "WARNING: Version downgrade detected!"
        echo "  Current version: $CURRENT_CARGO_VERSION"
        echo "  New version:     $NEW_VERSION"
        echo ""
        read -p "Do you want to continue with the downgrade? (y/N): " confirm
        if [[ ! $confirm =~ ^[Yy]$ ]]; then
            echo "Version update cancelled."
            exit 1
        fi
    fi
fi

echo "Updating version to $NEW_VERSION..."

# 1. Update __version__ in rustgression/__init__.py (convert alpha/beta format)
echo "Updating rustgression/__init__.py..."
PYTHON_VERSION=$(echo "$NEW_VERSION" | sed 's/-alpha\./.a/' | sed 's/-beta\./.b/' | sed 's/-rc\./.rc/')
sed -i.bak "s/__version__ = \".*\"/__version__ = \"$PYTHON_VERSION\"/" rustgression/__init__.py

# 2. Update version test in tests/test_imports.py
echo "Updating tests/test_imports.py..."
sed -i.bak "s/assert rustgression.__version__ == \".*\"/assert rustgression.__version__ == \"$PYTHON_VERSION\"/" tests/test_imports.py

# 3. Update version in Cargo.toml
echo "Updating Cargo.toml..."
sed -i.bak "s/^version = \".*\"/version = \"$NEW_VERSION\"/" Cargo.toml

# 4. Update version in pyproject.toml (convert alpha/beta format)
echo "Updating pyproject.toml..."
sed -i.bak "s/^version = \".*\"/version = \"$PYTHON_VERSION\"/" pyproject.toml

# 5. Update Cargo.lock (check project with cargo command)
echo "Updating Cargo.lock..."
cargo check --quiet

# 6. Update uv.lock (sync project dependencies with uv command)
echo "Updating uv.lock..."
if command -v uv >/dev/null 2>&1; then
    uv lock
else
    echo "Warning: uv command not found. Skipping uv.lock update."
fi

# Remove backup files
rm -f rustgression/__init__.py.bak tests/test_imports.py.bak Cargo.toml.bak pyproject.toml.bak

echo "SUCCESS: Version update completed: $NEW_VERSION"
echo ""
echo "Updated files:"
echo "- rustgression/__init__.py (Python format: $PYTHON_VERSION)"
echo "- tests/test_imports.py (Python format: $PYTHON_VERSION)"
echo "- Cargo.toml (Rust format: $NEW_VERSION)"
echo "- pyproject.toml (Python format: $PYTHON_VERSION)"
echo "- Cargo.lock"
if command -v uv >/dev/null 2>&1; then
    echo "- uv.lock"
fi
echo ""
echo "Next steps:"
echo "1. Review changes: git diff"
echo "2. Run tests: cargo test && python -m pytest"
echo "3. Commit changes: git add . && git commit -m \"chore: bump version to $NEW_VERSION\""
