#!/bin/bash

# Version consistency check script
# Usage: ./scripts/version-check.sh <version>
# Example: ./scripts/version-check.sh 0.2.2

set -e

# Specify version as argument
if [ $# -ne 1 ]; then
    echo "Error: Version not specified"
    echo "Usage: $0 <version>"
    echo "Example: $0 0.2.2"
    exit 1
fi

ARG_VERSION=$1

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

echo "Checking version consistency..."
echo "Expected version: $ARG_VERSION"

# Get version from rustgression/__init__.py
INIT_VERSION=$(grep '^__version__ = ' rustgression/__init__.py | cut -d '"' -f2)
echo "rustgression/__init__.py version: $INIT_VERSION"

# Get test version from tests/test_imports.py
TEST_VERSION=$(grep 'assert rustgression.__version__ == ' tests/test_imports.py | cut -d '"' -f2)
echo "tests/test_imports.py test version: $TEST_VERSION"

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

# Check version consistency
echo ""
echo "Version consistency check results:"

ERRORS=0

# Check for potential version downgrade
CURRENT_CARGO_VERSION=$(grep '^version = ' Cargo.toml | cut -d '"' -f2)
if [ -n "$CURRENT_CARGO_VERSION" ] && [ "$ARG_VERSION" != "$CURRENT_CARGO_VERSION" ]; then
    if ! compare_versions "$ARG_VERSION" "$CURRENT_CARGO_VERSION"; then
        echo "WARNING: Potential version downgrade detected!"
        echo "  Current Cargo.toml version: $CURRENT_CARGO_VERSION"
        echo "  Expected version:           $ARG_VERSION"
        echo ""
    fi
fi

if [ "$ARG_VERSION" != "$INIT_VERSION" ]; then
    echo "[FAIL] rustgression/__init__.py version mismatch: expected=$ARG_VERSION, actual=$INIT_VERSION"
    ERRORS=$((ERRORS + 1))
else
    echo "[PASS] rustgression/__init__.py"
fi

if [ "$ARG_VERSION" != "$TEST_VERSION" ]; then
    echo "[FAIL] tests/test_imports.py test version mismatch: expected=$ARG_VERSION, actual=$TEST_VERSION"
    ERRORS=$((ERRORS + 1))
else
    echo "[PASS] tests/test_imports.py"
fi

if [ "$ARG_VERSION" != "$CARGO_VERSION" ]; then
    echo "[FAIL] Cargo.toml version mismatch: expected=$ARG_VERSION, actual=$CARGO_VERSION"
    ERRORS=$((ERRORS + 1))
else
    echo "[PASS] Cargo.toml"
fi

if [ "$ARG_VERSION" != "$PYPROJECT_VERSION" ]; then
    echo "[FAIL] pyproject.toml version mismatch: expected=$ARG_VERSION, actual=$PYPROJECT_VERSION"
    ERRORS=$((ERRORS + 1))
else
    echo "[PASS] pyproject.toml"
fi

if [ "$ARG_VERSION" != "$CARGO_LOCK_VERSION" ]; then
    echo "[FAIL] Cargo.lock version mismatch: expected=$ARG_VERSION, actual=$CARGO_LOCK_VERSION"
    ERRORS=$((ERRORS + 1))
else
    echo "[PASS] Cargo.lock"
fi

if [ "$ARG_VERSION" != "$UV_LOCK_VERSION" ]; then
    echo "[FAIL] uv.lock version mismatch: expected=$ARG_VERSION, actual=$UV_LOCK_VERSION"
    ERRORS=$((ERRORS + 1))
else
    echo "[PASS] uv.lock"
fi

echo ""

if [ $ERRORS -eq 0 ]; then
    echo "SUCCESS: All versions match: $ARG_VERSION"
    exit 0
else
    echo "ERROR: Found $ERRORS version inconsistencies"
    echo ""
    echo "To fix:"
    echo "  ./scripts/version-update.sh $ARG_VERSION"
    exit 1
fi
