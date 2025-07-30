#!/usr/bin/env python3
"""
インポートテストと基本動作確認スクリプト

このスクリプトは以下を実行します:
1. rustgressionモジュールのインポートテスト
2. 基本的なクラスのインスタンス化テスト
3. 簡単な回帰分析の動作確認
"""

import sys
import traceback

import numpy as np


def test_import():
    """rustgressionモジュールのインポートテスト"""
    try:
        import rustgression

        print(f"Successfully imported rustgression {rustgression.__version__}")
        return True
    except ImportError as e:
        print(f"Failed to import rustgression: {e}")
        return False


def test_basic_functionality():
    """基本的なクラスのインスタンス化と動作テスト"""
    try:
        import rustgression

        # 簡単なテストデータを作成
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([2.0, 4.0, 6.0])

        # OLSレグレッサーを試す（新しいプロパティメソッドAPI）
        print("Creating OLS regressor...")
        regressor = rustgression.OlsRegressor(x, y)
        print(f"OLS slope: {regressor.slope()}")
        print(f"OLS intercept: {regressor.intercept()}")
        print(f"OLS r_value: {regressor.r_value()}")

        # TLSレグレッサーも試す（新しいプロパティメソッドAPI）
        print("Creating TLS regressor...")
        tls = rustgression.TlsRegressor(x, y)
        print(f"TLS slope: {tls.slope()}")
        print(f"TLS intercept: {tls.intercept()}")
        print(f"TLS r_value: {tls.r_value()}")

        print("All classes imported and tested successfully!")
        return True

    except Exception as e:
        print(f"Error during class testing: {e}")
        traceback.print_exc()
        return False


def main():
    """メイン実行関数"""
    print("=== Rustgression Import and Basic Functionality Test ===")

    # インポートテスト
    if not test_import():
        sys.exit(1)

    # 基本動作テスト
    if not test_basic_functionality():
        sys.exit(1)

    print("All tests passed successfully!")


if __name__ == "__main__":
    main()
