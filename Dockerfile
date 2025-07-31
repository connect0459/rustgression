FROM python:3.11-slim

# システムパッケージのインストール
RUN apt-get update && apt-get install -y \
  cmake \
  libblas-dev \
  liblapack-dev \
  gfortran \
  curl \
  build-essential \
  pkg-config \
  libssl-dev \
  && rm -rf /var/lib/apt/lists/*

# Rustのインストール
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# uvのインストール
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

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
