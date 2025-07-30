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
docker compose exec rustgression-dev -w /workspace uv run pytest

# Lintの実行
docker compose exec rustgression-dev -w /workspace uv run ruff check

# パッケージの再ビルド（ファイル変更後）
docker compose exec rustgression-dev -w /workspace uv run maturin develop
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

# コンテナ外からのコマンド実行
docker compose exec rustgression-container -w /workspace uv run pytest
```
