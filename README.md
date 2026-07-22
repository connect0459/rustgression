# rustgression

[![CI](https://github.com/connect0459/rustgression/actions/workflows/ci.yml/badge.svg)](https://github.com/connect0459/rustgression/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/connect0459/rustgression/blob/main/LICENSE)
[![PyPI Downloads](https://static.pepy.tech/badge/rustgression)](https://pepy.tech/projects/rustgression)

This project provides fast regression analysis (OLS, TLS) as a Python package.

- **Homepage**: <https://github.com/connect0459/rustgression>
- **Bug Tracker**: <https://github.com/connect0459/rustgression/issues>

## Overview

`rustgression` provides high-performance regression analysis tools implemented in Rust as a Python package.
It includes the following features.

- **Ordinary Least Squares (OLS)**: Minimizes the sum of squared vertical residuals (errors in y only). Assumes x is measured without error.
- **Total Least Squares (TLS)**: Minimizes the sum of squared orthogonal (perpendicular) distances from data points to the fitted line. Accounts for measurement errors in both x and y.

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

- [API Reference](https://github.com/connect0459/rustgression/blob/main/docs/api.md)
- [Contributing Guide](https://github.com/connect0459/rustgression/blob/main/CONTRIBUTING.md)
- [Changelog](https://github.com/connect0459/rustgression/blob/main/CHANGELOG.md)
