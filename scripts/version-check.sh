#!/bin/bash

# バージョン一貫性チェックスクリプト
# 使用法: ./scripts/version-check.sh <version>
# 例: ./scripts/version-check.sh 0.2.2

set -e

# 引数でバージョンを指定
if [ $# -ne 1 ]; then
    echo "エラー: バージョンが指定されていません"
    echo "使用法: $0 <version>"
    echo "例: $0 0.2.2"
    exit 1
fi

ARG_VERSION=$1

echo "バージョン一貫性をチェックしています..."
echo "期待するバージョン: $ARG_VERSION"

# rustgression/__init__.pyのバージョンを取得
INIT_VERSION=$(grep '^__version__ = ' rustgression/__init__.py | cut -d '"' -f2)
echo "rustgression/__init__.py バージョン: $INIT_VERSION"

# tests/test_imports.pyのテストバージョンを取得
TEST_VERSION=$(grep 'assert rustgression.__version__ == ' tests/test_imports.py | cut -d '"' -f2)
echo "tests/test_imports.py テストバージョン: $TEST_VERSION"

# Cargo.tomlのバージョンを取得
CARGO_VERSION=$(grep '^version = ' Cargo.toml | cut -d '"' -f2)
echo "Cargo.toml バージョン: $CARGO_VERSION"

# pyproject.tomlのバージョンを取得
PYPROJECT_VERSION=$(grep '^version = ' pyproject.toml | cut -d '"' -f2)
echo "pyproject.toml バージョン: $PYPROJECT_VERSION"

# Cargo.lockのバージョンを取得
CARGO_LOCK_VERSION=$(grep -A 1 '^name = "rustgression"' Cargo.lock | grep '^version = ' | cut -d '"' -f2)
echo "Cargo.lock バージョン: $CARGO_LOCK_VERSION"

# uv.lockのバージョンを取得（rustgressionパッケージのバージョン）
UV_LOCK_VERSION=$(grep -A 10 '^\[\[package\]\]' uv.lock | grep -A 10 'name = "rustgression"' | grep '^version = ' | head -1 | cut -d '"' -f2)
echo "uv.lock バージョン: $UV_LOCK_VERSION"

# バージョンの一致を確認
echo ""
echo "バージョン一致チェック結果:"

ERRORS=0

if [ "$ARG_VERSION" != "$INIT_VERSION" ]; then
    echo "❌ rustgression/__init__.py のバージョンが一致しません: 期待値=$ARG_VERSION, 実際値=$INIT_VERSION"
    ERRORS=$((ERRORS + 1))
else
    echo "✅ rustgression/__init__.py"
fi

if [ "$ARG_VERSION" != "$TEST_VERSION" ]; then
    echo "❌ tests/test_imports.py のテストバージョンが一致しません: 期待値=$ARG_VERSION, 実際値=$TEST_VERSION"
    ERRORS=$((ERRORS + 1))
else
    echo "✅ tests/test_imports.py"
fi

if [ "$ARG_VERSION" != "$CARGO_VERSION" ]; then
    echo "❌ Cargo.toml のバージョンが一致しません: 期待値=$ARG_VERSION, 実際値=$CARGO_VERSION"
    ERRORS=$((ERRORS + 1))
else
    echo "✅ Cargo.toml"
fi

if [ "$ARG_VERSION" != "$PYPROJECT_VERSION" ]; then
    echo "❌ pyproject.toml のバージョンが一致しません: 期待値=$ARG_VERSION, 実際値=$PYPROJECT_VERSION"
    ERRORS=$((ERRORS + 1))
else
    echo "✅ pyproject.toml"
fi

if [ "$ARG_VERSION" != "$CARGO_LOCK_VERSION" ]; then
    echo "❌ Cargo.lock のバージョンが一致しません: 期待値=$ARG_VERSION, 実際値=$CARGO_LOCK_VERSION"
    ERRORS=$((ERRORS + 1))
else
    echo "✅ Cargo.lock"
fi

if [ "$ARG_VERSION" != "$UV_LOCK_VERSION" ]; then
    echo "❌ uv.lock のバージョンが一致しません: 期待値=$ARG_VERSION, 実際値=$UV_LOCK_VERSION"
    ERRORS=$((ERRORS + 1))
else
    echo "✅ uv.lock"
fi

echo ""

if [ $ERRORS -eq 0 ]; then
    echo "🎉 全てのバージョンが一致しています: $ARG_VERSION"
    exit 0
else
    echo "💥 $ERRORS 個のバージョン不整合が見つかりました"
    echo ""
    echo "修正方法:"
    echo "  ./scripts/version-update.sh $ARG_VERSION"
    exit 1
fi
