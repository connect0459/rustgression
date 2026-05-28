# rustgression

[![PyPI Downloads](https://static.pepy.tech/badge/rustgression)](https://pepy.tech/projects/rustgression)
[![CI](https://github.com/connect0459/rustgression/actions/workflows/ci.yml/badge.svg)](https://github.com/connect0459/rustgression/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

This project provides fast regression analysis (OLS, TLS) as a Python package.

- **WebSite**: <https://pypi.org/project/rustgression/>
- **Documentation**: <https://github.com/connect0459/rustgression/blob/main/README.md>
- **Source code**: <https://github.com/connect0459/rustgression>
- **Bug reports and security issues**: <https://github.com/connect0459/rustgression/issues>

## Overview

`rustgression` provides high-performance regression analysis tools implemented in Rust as a Python package.
It includes the following features.

- **Ordinary Least Squares (OLS)**: Traditional least squares method. Minimizes errors only in the y-direction.
- **Total Least Squares (TLS)**: Orthogonal regression. Considers errors in both variables (x-axis and y-axis).

This package targets Python version `3.11` and above.

## Installation

Using pip:

```bash
pip install rustgression
```

Using uv:

```bash
uv add rustgression
```

## Documentation

- **[API Reference](docs/api.md)**
- **[Development Guide](docs/development.md)**
- **[Changelog](CHANGELOG.md)**

## Author

[connect0459](https://github.com/connect0459)
