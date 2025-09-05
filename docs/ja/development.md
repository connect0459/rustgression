# 開発環境

## aptパッケージのインストール

Rustを動かすために以下の依存関係が必要です。

- cmake
- BLASとLAPACKの開発ライブラリ
- Fortranコンパイラ

Debian系OSの場合は以下のコマンドでインストールしてください。

```bash
sudo apt-get update
sudo apt-get install cmake libblas-dev liblapack-dev gfortran
```

## Python環境の構築

`pyproject.toml`をプロジェクトルートに設置して、下記コマンドでインストールしてください。

```bash
uv sync
```

## 作成したパッケージをテストする

`maturin`を用いて、Rustからのクリーンビルドを行います。以下のコマンドで必要な依存関係をインストールし、ビルドを行ってください。

```bash
# 必要なツールのインストール確認
uv add maturin
# クリーンビルド
uv run maturin develop
```

## Dockerを使った開発環境

Docker Composeを使って開発環境を構築できます。ボリュームマウントによりホスト側のファイル変更がリアルタイムで反映されます。

### 開発環境の起動

```bash
# コンテナをバックグラウンドで起動
docker compose up -d

# またはログを表示しながら起動
docker compose up
```

### コンテナ内でのコマンド実行

```bash
# テストの実行
docker compose exec -w /workspace rustgression-dev uv run pytest

# Lintの実行
docker compose exec -w /workspace rustgression-dev uv run ruff check

# パッケージの再ビルド（ファイル変更後）
docker compose exec -w /workspace rustgression-dev uv run maturin develop
```

### 開発環境の管理

```bash
# コンテナの停止
docker compose stop

# コンテナの停止と削除
docker compose down

# イメージの再ビルドとコンテナの再作成
docker compose up --build

# ボリュームも含めて完全に削除
docker compose down -v
```

## バージョン管理

### バージョンアップ手順

プロジェクトのバージョンを統一的に管理するため、`scripts/version-update.sh` スクリプトを使用します。

```bash
# 通常のバージョンアップ（例: 0.2.0 → 0.2.1）
docker compose exec -w /workspace rustgression-dev ./scripts/version-update.sh 0.2.1

# アルファ版リリース
docker compose exec -w /workspace rustgression-dev ./scripts/version-update.sh 0.3.0-alpha.1

# ベータ版リリース
docker compose exec -w /workspace rustgression-dev ./scripts/version-update.sh 0.3.0-beta.1
```

このスクリプトは以下のファイルを自動更新します：

- `Cargo.toml` - Rustパッケージバージョン
- `pyproject.toml` - Pythonパッケージバージョン
- `rustgression/__init__.py` - パッケージ内バージョン定数
- `Cargo.lock` - 依存関係ロックファイル

### バージョンの確認

全ファイルのバージョン整合性を確認：

```bash
docker compose exec -w /workspace rustgression-dev ./scripts/version-check.sh 0.2.1
```

### 注意事項

- バージョン更新後は必ずテストを実行してください
- リリース前にバージョンの整合性を確認してください
- セマンティックバージョニング（SemVer）に従ってください
  - MAJOR: 互換性のない変更
  - MINOR: 後方互換性のある機能追加
  - PATCH: 後方互換性のあるバグ修正
