# 開発環境の構築

## aptパッケージのインストール

cmake

```bash
sudo apt-get update
sudo apt-get install cmake 
```

BLASとLAPACKの開発ライブラリのインストール（必要な依存関係）:

```bash
sudo apt-get install libblas-dev liblapack-dev
```

Fortranコンパイラのインストール（LAPACKビルドに必要）:

```bash
sudo apt-get install gfortran
```
