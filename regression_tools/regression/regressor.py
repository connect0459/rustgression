"""
回帰分析のPythonインターフェース
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..regression_tools import calculate_ols_regression, calculate_tls_regression


@dataclass
class RegressionParams:
    """基本的な回帰パラメータを格納するデータクラス"""

    slope: float
    intercept: float
    correlation: float


@dataclass
class OlsRegressionParams(RegressionParams):
    """OLS回帰のパラメータを格納するデータクラス"""

    p_value: float
    std_err: float


class BaseRegressor(ABC):
    """回帰分析の基底クラス

    全ての回帰実装の共通インターフェースを定義します。
    """

    def __init__(self, x: np.ndarray, y: np.ndarray):
        """回帰モデルの初期化とフィッティング

        Parameters
        ----------
        x : np.ndarray
            x軸データ (独立変数)
        y : np.ndarray
            y軸データ (従属変数)
        """
        # 入力データの検証と前処理
        self.x = np.asarray(x, dtype=np.float64).flatten()
        self.y = np.asarray(y, dtype=np.float64).flatten()

        if self.x.shape[0] != self.y.shape[0]:
            raise ValueError("入力配列の長さが一致しません")

        if self.x.shape[0] < 2:
            raise ValueError("回帰には少なくとも2つのデータポイントが必要です")

        # 基本パラメータの初期化
        self.slope: float
        self.intercept: float
        self.correlation: float

        # フィッティングの実行
        self._fit()

    @abstractmethod
    def _fit(self) -> None:
        """回帰を実行する抽象メソッド"""
        pass

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
        x = np.asarray(x, dtype=np.float64)
        return self.slope * x + self.intercept

    def get_params(self) -> RegressionParams:
        """回帰パラメータを取得する

        Returns
        -------
        RegressionParams
            回帰パラメータを含むデータクラス
        """
        return RegressionParams(
            slope=self.slope, intercept=self.intercept, correlation=self.correlation
        )

    def __repr__(self) -> str:
        """文字列表現

        Returns
        -------
        str
            文字列表現
        """
        return (
            f"{self.__class__.__name__}("
            f"slope={self.slope:.6f}, "
            f"intercept={self.intercept:.6f}, "
            f"correlation={self.correlation:.6f})"
        )


class OlsRegressor(BaseRegressor):
    """Ordinary Least Squares (OLS)回帰を計算するクラス

    y方向の誤差を最小化する標準的な最小二乗法を実装します。
    """

    def __init__(self, x: np.ndarray, y: np.ndarray):
        """OLSRegressorの初期化とフィッティング

        Parameters
        ----------
        x : np.ndarray
            x軸データ (独立変数)
        y : np.ndarray
            y軸データ (従属変数)
        """
        self.p_value: float
        self.std_err: float
        super().__init__(x, y)

    def _fit(self) -> None:
        """OLS回帰を実行"""
        # Rust実装を呼び出し
        _, self.slope, self.intercept, self.correlation, self.p_value, self.std_err = (
            calculate_ols_regression(self.x, self.y)
        )

    def get_params(self) -> OlsRegressionParams:
        """回帰パラメータを取得する

        Returns
        -------
        OlsRegressionParams
            全ての回帰パラメータを含むデータクラス
        """
        return OlsRegressionParams(
            slope=self.slope,
            intercept=self.intercept,
            correlation=self.correlation,
            p_value=self.p_value,
            std_err=self.std_err,
        )


class TlsRegressor(BaseRegressor):
    """Total Least Squares (TLS)回帰、または直交回帰を計算するクラス

    通常の最小二乗法(OLS)がy方向の誤差のみを最小化するのに対し、
    TLSは両方の変数(xとy)の誤差を考慮します。これは、両変数に測定誤差が
    存在する場合により適切なアプローチとなります。
    """

    def _fit(self) -> None:
        """TLS回帰を実行"""
        # Rust実装を呼び出し
        _, self.slope, self.intercept, self.correlation = calculate_tls_regression(
            self.x, self.y
        )


def create_regressor(
    x: np.ndarray, y: np.ndarray, method: Literal["ols", "tls"] = "ols"
) -> BaseRegressor:
    """回帰モデルのファクトリ関数

    Parameters
    ----------
    x : np.ndarray
        x軸データ (独立変数)
    y : np.ndarray
        y軸データ (従属変数)
    method : str
        使用する回帰手法 ("ols" または "tls")

    Returns
    -------
    BaseRegressor
        指定された回帰モデルのインスタンス
    """
    if method == "ols":
        return OlsRegressor(x, y)
    elif method == "tls":
        return TlsRegressor(x, y)
    else:
        raise ValueError(f"未知の回帰手法です: {method}")
