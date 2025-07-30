# ADR-006: 回帰分析APIのドメイン駆動設計への移行

## ステータス

- [x] Proposed
- [ ] Accepted
- [ ] Deprecated

## コンテキスト

現在のrustgressionライブラリは、データクラス（`OlsRegressionParams`、`TlsRegressionParams`）を使用したgetter/setterパターンを採用している。このアプローチは以下の問題を抱えている：

### 現在の問題点

1. **Tell Don't Ask原則の違反**: オブジェクトの状態を問い合わせて、その結果に基づいて処理を行うパターンが多用されている
2. **Getterの使用**: `get_xxx()`のような実装パターンが存在することによって、オブジェクトの破壊を引き起こすsetterが将来的に実装される恐れがある。これを防止するために、ライブラリ設計をgetter/setterパターンからドメインオブジェクトの振る舞いとして実装するパターンに統一する。

### 現在のAPI例

```python
# 現在のアプローチ - クライアント側でドメイン知識が必要
regressor = OlsRegressor(x, y)
params = regressor.get_params()
```

## 決定事項

回帰分析APIをドメイン駆動設計（DDD）の原則に基づいて再設計し、振る舞い中心のAPIに移行する。

### 1. プロパティメソッドパターンの導入

データクラスのgetter/setterパターンを振る舞いとしてのプロパティメソッドに置き換える：

- `slope() -> float`
- `intercept() -> float`
- `r_value() -> float`
- `p_value() -> float`
- `stderr() -> float`
- `intercept_stderr() -> float`

### 2. 単一責任原則に基づく設計

Regressorクラス自体が統計値をプロパティメソッドとして提供するよう変更：

```python
class OlsRegressor:
    def __init__(self, x, y):
        # インスタンス化時に回帰分析を実行し、統計値を計算
        self._slope, self._intercept, self._r_value, self._p_value, self._stderr, self._intercept_stderr = self._calculate_regression(x, y)
        
    def slope(self) -> float:
        """回帰直線の傾きを返す"""
        return self._slope
        
    def intercept(self) -> float:
        """回帰直線の切片を返す"""
        return self._intercept
        
    def r_value(self) -> float:
        """相関係数を返す"""
        return self._r_value
        
    def p_value(self) -> float:
        """p値を返す"""
        return self._p_value
        
    def stderr(self) -> float:
        """傾きの標準誤差を返す"""
        return self._stderr
        
    def intercept_stderr(self) -> float:
        """切片の標準誤差を返す"""
        return self._intercept_stderr
```

### 3. 後方互換性の維持

既存のAPIとの互換性を保つため：

- 既存のデータクラス（`OlsRegressionParams`、`TlsRegressionParams`）は残存
- `get_params()`メソッドは非推奨として維持
- 段階的移行をサポート

### 4. 実装戦略

#### Phase 1: 基盤整備

- `OlsRegressor`, `TlsRegressor`クラスでのプロパティメソッド実装
- インスタンス化時の統計値計算処理の追加

#### Phase 2: 互換性対応

- 既存`get_params()`メソッドの非推奨化
- 段階的移行のためのドキュメント更新

#### Phase 3: テスト・ドキュメント更新

- 新APIのテスト追加
- 使用例とドキュメントの更新
- 移行ガイドの作成

## 結果

### 期待される効果

1. **単一責任原則の適用**: Regressorクラスが統計値の計算と提供を一元的に担当
2. **振る舞い中心の設計**: プロパティメソッドにより振る舞いとしてデータアクセスを提供
3. **API の一貫性向上**: 統計値へのアクセス方法が統一される
4. **パフォーマンス向上**: インスタンス化時に一度だけ計算を実行
5. **既存機能の維持**: 現在提供している全ての統計値（slope, intercept, r_value, p_value, stderr, intercept_stderr）を継続提供

### 新しいAPI例

```python
# 新しいアプローチ - Regressorクラスから直接プロパティメソッドでアクセス
regressor = OlsRegressor(x, y)  # インスタンス化時に統計値を計算

# 振る舞いとしての統計値アクセス
print(f"傾き: {regressor.slope():.3f}")
print(f"切片: {regressor.intercept():.3f}")
print(f"相関係数: {regressor.r_value():.3f}")
print(f"p値: {regressor.p_value():.6f}")
print(f"傾きの標準誤差: {regressor.stderr():.3f}")
print(f"切片の標準誤差: {regressor.intercept_stderr():.3f}")
```

### Breaking Changes

- 新しいプロパティメソッドAPIの導入
- Regressorクラスへの直接的なプロパティメソッド追加
- `get_params()`メソッドの非推奨化（段階的移行）

### 影響範囲

- 既存のクライアントコード: 互換性維持により影響最小限
- テストコード: 新APIのテスト追加が必要
- ドキュメント: 新APIの説明と移行ガイドが必要

## 参考資料

- [Domain-Driven Design: Tackling Complexity in the Heart of Software](https://www.amazon.com/Domain-Driven-Design-Tackling-Complexity-Software/dp/0321125215)
- [Tell Don't Ask Principle](https://martinfowler.com/bliki/TellDontAsk.html)
- [Anemic Domain Model Anti-pattern](https://martinfowler.com/bliki/AnemicDomainModel.html)
- [Python Enum Documentation](https://docs.python.org/3/library/enum.html)

## 関連ファイルのパス

### 初期実装時 (2025-07-30)

更新予定

- `rustgression/regression/base_regressor.py`
- `rustgression/regression/ols_regressor.py`
- `rustgression/regression/tls_regressor.py`
- `rustgression/__init__.py` (新しいクラスのエクスポート)
- `tests/test_regressor.py` (新APIのテスト追加)
- `examples/scientific_example.py` (新APIの使用例)
- `examples/simple_example.py` (新APIの使用例)
- `docs/ja/development.md` (API使用例の更新)
- `docs/en/development.md` (API使用例の更新)
