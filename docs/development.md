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
