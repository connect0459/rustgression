FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

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

COPY rust-toolchain.toml .

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Avoid hard link errors
ENV UV_LINK_MODE=copy

WORKDIR /workspace

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

COPY . .
RUN uv sync --frozen

RUN uv run maturin develop

# Keep running in the background
CMD ["tail", "-f", "/dev/null"]
