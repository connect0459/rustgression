#!/bin/bash

# バージョン更新スクリプト
# 使用法: ./scripts/version-update.sh <new_version>
# 例: ./scripts/version-update.sh 0.2.1

set -e

if [ $# -ne 1 ]; then
    echo "使用法: $0 <new_version>"
    echo "例: $0 0.2.1"
    exit 1
fi

NEW_VERSION=$1

# バージョン形式をチェック (semver形式、プレリリース対応)
if ! [[ $NEW_VERSION =~ ^[0-9]+\.[0-9]+\.[0-9]+(-alpha\.[0-9]+|-beta\.[0-9]+|-rc\.[0-9]+)?$ ]]; then
    echo "エラー: バージョンはsemver形式で指定してください"
    echo "例: 1.0.0, 1.0.0-alpha.1, 1.0.0-beta.2, 1.0.0-rc.1"
    exit 1
fi

echo "バージョンを $NEW_VERSION に更新します..."

# 1. rustgression/__init__.py の __version__ を更新 (alpha/beta形式を変換)
echo "rustgression/__init__.py を更新中..."
PYTHON_VERSION=$(echo "$NEW_VERSION" | sed 's/-alpha\./.a/' | sed 's/-beta\./.b/' | sed 's/-rc\./.rc/')
sed -i.bak "s/__version__ = \".*\"/__version__ = \"$PYTHON_VERSION\"/" rustgression/__init__.py

# 2. Cargo.toml のバージョンを更新
echo "Cargo.toml を更新中..."
sed -i.bak "s/^version = \".*\"/version = \"$NEW_VERSION\"/" Cargo.toml

# 3. pyproject.toml のバージョンを更新 (alpha/beta形式を変換)
echo "pyproject.toml を更新中..."
sed -i.bak "s/^version = \".*\"/version = \"$PYTHON_VERSION\"/" pyproject.toml

# 4. tests/test_imports.py のバージョンテストを更新
echo "tests/test_imports.py を更新中..."
sed -i.bak "s/assert rustgression.__version__ == \".*\"/assert rustgression.__version__ == \"$PYTHON_VERSION\"/" tests/test_imports.py

# 5. Cargo.lock を更新 (cargoコマンドでプロジェクトをチェック)
echo "Cargo.lock を更新中..."
cargo check --quiet

# 6. uv.lock を更新 (uvコマンドでプロジェクトの依存関係を同期)
echo "uv.lock を更新中..."
if command -v uv >/dev/null 2>&1; then
    uv lock
else
    echo "警告: uvコマンドが見つかりません。uv.lockの更新をスキップします。"
fi

# バックアップファイルを削除
rm -f rustgression/__init__.py.bak Cargo.toml.bak pyproject.toml.bak tests/test_imports.py.bak

echo "✅ バージョン更新が完了しました: $NEW_VERSION"
echo ""
echo "更新されたファイル:"
echo "- rustgression/__init__.py (Python形式: $PYTHON_VERSION)"
echo "- Cargo.toml (Rust形式: $NEW_VERSION)"
echo "- pyproject.toml (Python形式: $PYTHON_VERSION)"
echo "- tests/test_imports.py (Python形式: $PYTHON_VERSION)"
echo "- Cargo.lock"
if command -v uv >/dev/null 2>&1; then
    echo "- uv.lock"
fi
echo ""
echo "次の手順:"
echo "1. 変更内容を確認: git diff"
echo "2. テストを実行: cargo test && python -m pytest"
echo "3. 変更をコミット: git add . && git commit -m \"chore: bump version to $NEW_VERSION\""
