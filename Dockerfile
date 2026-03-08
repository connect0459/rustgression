FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

# システムパッケージのインストール
RUN apt-get update && apt-get install -y \
  build-essential \
  cmake \
  curl \
  gfortran \
  libblas-dev \
  liblapack-dev \
  libssl-dev \
  pkg-config \
  && rm -rf /var/lib/apt/lists/*

# 先にrust-toolchain.tomlをコピーしてバージョンを指定
COPY rust-toolchain.toml .

# Rustのインストール
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# uvの設定（ハードリンクエラーを回避）
ENV UV_LINK_MODE=copy

# 作業ディレクトリの設定
WORKDIR /workspace

# プロジェクトファイルをコピー
COPY . .

# Python環境のセットアップ
RUN uv sync

# maturinでRustパッケージをビルド
RUN uv run maturin develop

# コンテナ起動時のデフォルトコマンド（バックグラウンドで起動し続ける）
CMD ["tail", "-f", "/dev/null"]
