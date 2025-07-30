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

Dockerコンテナを使って開発環境を構築することもできます。

### イメージのビルド

```bash
docker build -t rustgression-dev .
```

### コンテナの起動

```bash
# コンテナを起動（バックグラウンド）
docker run -d --name rustgression-container rustgression-dev

# またはインタラクティブモードで起動
docker run -it --name rustgression-container rustgression-dev
```

### コンテナ外からのコマンド実行

```bash
# テストの実行
docker exec rustgression-container uv run pytest

# Lintの実行
docker exec rustgression-container uv run ruff check

# パッケージの再ビルド
docker exec rustgression-container uv run maturin develop

# Bashシェルに入る
docker exec -it rustgression-container /bin/bash
```

### コンテナの管理

```bash
# コンテナの停止
docker stop rustgression-container

# コンテナの開始
docker start rustgression-container

# コンテナの削除
docker rm rustgression-container

# イメージの削除
docker rmi rustgression-dev
```
