#!/bin/bash

# バージョン一貫性チェックスクリプト
# 使用法: ./scripts/version-check.sh [git_version]
# 例: ./scripts/version-check.sh 0.2.2

set -e

# 引数が指定された場合はそれを使用、なければGitタグから取得
if [ $# -eq 1 ]; then
    GIT_VERSION=$1
else
    # Gitタグからバージョンを取得（v0.0.1 -> 0.0.1）
    if [ -n "$GITHUB_REF" ]; then
        GIT_VERSION=${GITHUB_REF#refs/tags/v}
    else
        echo "エラー: バージョンが指定されていません"
        echo "使用法: $0 <version>"
        echo "例: $0 0.2.2"
        exit 1
    fi
fi

echo "バージョン一貫性をチェックしています..."
echo "期待するバージョン: $GIT_VERSION"

# Cargo.tomlのバージョンを取得
CARGO_VERSION=$(grep '^version = ' Cargo.toml | cut -d '"' -f2)
echo "Cargo.toml バージョン: $CARGO_VERSION"

# pyproject.tomlのバージョンを取得
PYPROJECT_VERSION=$(grep '^version = ' pyproject.toml | cut -d '"' -f2)
echo "pyproject.toml バージョン: $PYPROJECT_VERSION"

# rustgression/__init__.pyのバージョンを取得
INIT_VERSION=$(grep '^__version__ = ' rustgression/__init__.py | cut -d '"' -f2)
echo "rustgression/__init__.py バージョン: $INIT_VERSION"

# tests/test_imports.pyのテストバージョンを取得
TEST_VERSION=$(grep 'assert rustgression.__version__ == ' tests/test_imports.py | cut -d '"' -f2)
echo "tests/test_imports.py テストバージョン: $TEST_VERSION"

# バージョンの一致を確認
echo ""
echo "バージョン一致チェック結果:"

ERRORS=0

if [ "$GIT_VERSION" != "$CARGO_VERSION" ]; then
    echo "❌ Cargo.toml のバージョンが一致しません: 期待値=$GIT_VERSION, 実際値=$CARGO_VERSION"
    ERRORS=$((ERRORS + 1))
else
    echo "✅ Cargo.toml"
fi

if [ "$GIT_VERSION" != "$PYPROJECT_VERSION" ]; then
    echo "❌ pyproject.toml のバージョンが一致しません: 期待値=$GIT_VERSION, 実際値=$PYPROJECT_VERSION"
    ERRORS=$((ERRORS + 1))
else
    echo "✅ pyproject.toml"
fi

if [ "$GIT_VERSION" != "$INIT_VERSION" ]; then
    echo "❌ rustgression/__init__.py のバージョンが一致しません: 期待値=$GIT_VERSION, 実際値=$INIT_VERSION"
    ERRORS=$((ERRORS + 1))
else
    echo "✅ rustgression/__init__.py"
fi

if [ "$GIT_VERSION" != "$TEST_VERSION" ]; then
    echo "❌ tests/test_imports.py のテストバージョンが一致しません: 期待値=$GIT_VERSION, 実際値=$TEST_VERSION"
    ERRORS=$((ERRORS + 1))
else
    echo "✅ tests/test_imports.py"
fi

echo ""

if [ $ERRORS -eq 0 ]; then
    echo "🎉 全てのバージョンが一致しています: $GIT_VERSION"
    exit 0
else
    echo "💥 $ERRORS 個のバージョン不整合が見つかりました"
    echo ""
    echo "修正方法:"
    echo "  ./scripts/version-update.sh $GIT_VERSION"
    exit 1
fi
