#!/bin/bash

# Version consistency check script
# Usage: ./scripts/version-check.sh <version>
# Example: ./scripts/version-check.sh 0.2.2

set -e

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Color functions
error() {
    echo -e "${RED}$1${NC}"
}

warning() {
    echo -e "${YELLOW}$1${NC}"
}

success() {
    echo -e "${GREEN}$1${NC}"
}

info() {
    echo -e "${BLUE}$1${NC}"
}

# Specify version as argument
if [ $# -ne 1 ]; then
    error "Error: Version not specified"
    echo "Usage: $0 <version>"
    echo "Example: $0 0.2.2"
    exit 1
fi

ARG_VERSION=$1

if ! [[ $ARG_VERSION =~ ^[0-9]+\.[0-9]+\.[0-9]+(-alpha\.[0-9]+|-beta\.[0-9]+|-rc\.[0-9]+)?$ ]]; then
    error "Error: Version must be in semver format"
    echo "Examples: 1.0.0, 1.0.0-alpha.1, 1.0.0-beta.2, 1.0.0-rc.1"
    exit 1
fi

PYTHON_VERSION=$(echo "$ARG_VERSION" | sed 's/-alpha\./a/;s/-beta\./b/;s/-rc\./rc/')

_VERSION_UTILS="$(dirname "$0")/lib/version-utils.sh"
if [ ! -f "$_VERSION_UTILS" ]; then
    echo "Error: required library not found: $_VERSION_UTILS" >&2
    exit 1
fi
# shellcheck source=lib/version-utils.sh
source "$_VERSION_UTILS"

echo "Checking version consistency..."
echo "Expected version: $ARG_VERSION"
if [ "$ARG_VERSION" != "$PYTHON_VERSION" ]; then
    echo "Expected Python version: $PYTHON_VERSION"
fi

# Get version from src-py/rustgression/__init__.py
INIT_VERSION=$(grep '^__version__ = ' src-py/rustgression/__init__.py | cut -d '"' -f2)
echo "src-py/rustgression/__init__.py version: $INIT_VERSION"

# Get version from Cargo.toml
CARGO_VERSION=$(grep '^version = ' Cargo.toml | cut -d '"' -f2)
echo "Cargo.toml version: $CARGO_VERSION"

# Get version from pyproject.toml
PYPROJECT_VERSION=$(grep '^version = ' pyproject.toml | cut -d '"' -f2)
echo "pyproject.toml version: $PYPROJECT_VERSION"

# Get version from Cargo.lock
CARGO_LOCK_VERSION=$(grep -A 1 '^name = "rustgression"' Cargo.lock | grep '^version = ' | cut -d '"' -f2)
echo "Cargo.lock version: $CARGO_LOCK_VERSION"

# Get version from uv.lock (rustgression package version)
UV_LOCK_VERSION=$(grep -A 10 '^\[\[package\]\]' uv.lock | grep -A 10 'name = "rustgression"' | grep '^version = ' | head -1 | cut -d '"' -f2)
echo "uv.lock version: $UV_LOCK_VERSION"

CHANGELOG_FILE="CHANGELOG.md"

# Check version consistency
echo ""
echo "Version consistency check results:"

ERRORS=0

# Check for potential version downgrade
if [ -n "$CARGO_VERSION" ] && [ "$ARG_VERSION" != "$CARGO_VERSION" ]; then
    if ! compare_versions "$ARG_VERSION" "$CARGO_VERSION"; then
        warning "WARNING: Potential version downgrade detected!"
        warning "  Current Cargo.toml version: $CARGO_VERSION"
        warning "  Expected version:           $ARG_VERSION"
        echo ""
    fi
fi

if [ "$PYTHON_VERSION" != "$INIT_VERSION" ]; then
    error "[FAIL] src-py/rustgression/__init__.py version mismatch: expected=$PYTHON_VERSION, actual=$INIT_VERSION"
    ERRORS=$((ERRORS + 1))
else
    success "[PASS] src-py/rustgression/__init__.py"
fi

if [ "$ARG_VERSION" != "$CARGO_VERSION" ]; then
    error "[FAIL] Cargo.toml version mismatch: expected=$ARG_VERSION, actual=$CARGO_VERSION"
    ERRORS=$((ERRORS + 1))
else
    success "[PASS] Cargo.toml"
fi

if [ "$PYTHON_VERSION" != "$PYPROJECT_VERSION" ]; then
    error "[FAIL] pyproject.toml version mismatch: expected=$PYTHON_VERSION, actual=$PYPROJECT_VERSION"
    ERRORS=$((ERRORS + 1))
else
    success "[PASS] pyproject.toml"
fi

if [ "$ARG_VERSION" != "$CARGO_LOCK_VERSION" ]; then
    error "[FAIL] Cargo.lock version mismatch: expected=$ARG_VERSION, actual=$CARGO_LOCK_VERSION"
    ERRORS=$((ERRORS + 1))
else
    success "[PASS] Cargo.lock"
fi

if [ "$PYTHON_VERSION" != "$UV_LOCK_VERSION" ]; then
    error "[FAIL] uv.lock version mismatch: expected=$PYTHON_VERSION, actual=$UV_LOCK_VERSION"
    ERRORS=$((ERRORS + 1))
else
    success "[PASS] uv.lock"
fi

case "$ARG_VERSION" in
    *-alpha.*|*-beta.*|*-rc.*)
        info "[SKIP] $CHANGELOG_FILE (pre-release version)"
        ;;
    *)
        if [ ! -f "$CHANGELOG_FILE" ]; then
            error "[FAIL] $CHANGELOG_FILE not found"
            ERRORS=$((ERRORS + 1))
        else
            ESCAPED_VERSION=$(printf '%s' "$ARG_VERSION" | sed 's/\./\\./g')
            CHANGELOG_ENTRY=$(grep -c "^## \[$ESCAPED_VERSION\]" "$CHANGELOG_FILE" || true)
            if [ "$CHANGELOG_ENTRY" -eq 0 ]; then
                error "[FAIL] $CHANGELOG_FILE has no entry for version $ARG_VERSION"
                ERRORS=$((ERRORS + 1))
            else
                success "[PASS] $CHANGELOG_FILE"
            fi
        fi
        ;;
esac

echo ""

if [ $ERRORS -eq 0 ]; then
    if [ "$ARG_VERSION" != "$PYTHON_VERSION" ]; then
        success "SUCCESS: All versions match (Rust: $ARG_VERSION / Python: $PYTHON_VERSION)"
    else
        success "SUCCESS: All versions match: $ARG_VERSION"
    fi
    exit 0
else
    error "ERROR: Found $ERRORS version inconsistencies"
    echo ""
    info "To fix:"
    info "  ./scripts/version-update.sh $ARG_VERSION"
    exit 1
fi
