"""
TLS回帰のPythonインターフェース
"""

from typing import Tuple, Optional

import numpy as np
from . import calculate_tls_regression  # 相対インポートに変更


class TLSRegressor:
    """Total Least Squares (TLS)回帰、または直交回帰を計算するクラス。

    通常の最小二乗法(OLS)がy方向の誤差のみを最小化するのに対し、
    TLSは両方の変数(xとy)の誤差を考慮します。これは、両変数に測定誤差が
    存在する場合により適切なアプローチとなります。

    理論:
    1. データを標準化して、スケールの影響を除去
    2. 標準化データに対してTLS回帰を実行
    3. 得られたパラメータを元のスケールに戻す
    4. TLSは以下の最小化問題を解きます:
       min Σ(d_i^2), where d_i は点から直線までの直交距離
    """

    def __init__(self):
        """TLSRegressorの初期化"""
        self.slope: Optional[float] = None
        self.intercept: Optional[float] = None
        self.correlation: Optional[float] = None
        self.is_fitted: bool = False

    def fit(self, x: np.ndarray, y: np.ndarray) -> "TLSRegressor":
        """データにTLS回帰をフィットさせる

        Parameters
        ----------
        x : np.ndarray
            x軸データ (独立変数)
        y : np.ndarray
            y軸データ (従属変数)

        Returns
        -------
        self : TLSRegressor
            フィットした回帰モデル
        """
        # 1次元配列に変換
        x = np.asarray(x, dtype=np.float64).flatten()
        y = np.asarray(y, dtype=np.float64).flatten()

        if x.shape[0] != y.shape[0]:
            raise ValueError("Input arrays must have the same length")

        if x.shape[0] < 2:
            raise ValueError("At least two data points are required for regression")

        # Rust実装を呼び出し
        _, self.slope, self.intercept, self.correlation = calculate_tls_regression(x, y)
        self.is_fitted = True

        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """回帰モデルを使って予測を行う

        Parameters
        ----------
        x : np.ndarray
            予測のための入力データ

        Returns
        -------
        np.ndarray
            予測値
        """
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        x = np.asarray(x, dtype=np.float64)
        return self.slope * x + self.intercept

    def get_params(self) -> Tuple[float, float, float]:
        """回帰パラメータを取得する

        Returns
        -------
        Tuple[float, float, float]
            (傾き, 切片, 相関係数)のタプル
        """
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        return self.slope, self.intercept, self.correlation

    def __repr__(self) -> str:
        """文字列表現

        Returns
        -------
        str
            文字列表現
        """
        if self.is_fitted:
            return (
                f"TLSRegressor(slope={self.slope:.6f}, "
                f"intercept={self.intercept:.6f}, "
                f"correlation={self.correlation:.6f})"
            )
        return "TLSRegressor(unfitted)"
