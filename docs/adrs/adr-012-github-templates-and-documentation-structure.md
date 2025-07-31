# ADR-012: GitHub テンプレートとドキュメント構造の標準化

## ステータス

- [x] Proposed
- [x] Accepted
- [ ] Deprecated

## コンテキスト

rustgressionプロジェクトにおいて、コントリビューションの品質向上とドキュメントの国際化対応が必要となった。現在の状況は以下の通り：

### 現在の問題点

1. **Issue/PRテンプレートの不在**: 標準化されたフォーマットがないため、コントリビューションの質にばらつきが生じている
2. **ドキュメントの国際化不足**: 開発者向けドキュメントのみ英語・日本語対応しているが、一般ユーザー向けドキュメントが不整備
3. **プロジェクト情報の可視性不足**: PyPIダウンロード数などのプロジェクト統計情報が表示されていない
4. **コントリビューションガイドラインの不明確さ**: 新規コントリビューターが参照すべき情報が散在している

## 決定事項

GitHub テンプレートとドキュメント構造を標準化し、プロジェクトの可視性とコントリビューション品質を向上させる。

### 1. GitHub テンプレートの導入

#### Issue テンプレート

- `BUG_REPORT.md`: バグ報告用テンプレート
- `DEVELOP_REQUEST.md`: 機能開発・改善要求用テンプレート（後に`FEATURE_REQUEST.md`に名称変更）

#### Pull Request テンプレート

- `PULL_REQUEST_TEMPLATE.md`: 包括的なPRレビュー用テンプレート
  - 関連URL、概要、リリース情報
  - 対象デバイス・環境、テスト項目
  - 品質チェックリスト、作業時間の記録

### 2. ドキュメント多言語化構造

```txt
docs/
├── adrs/           # Architecture Decision Records
├── en/             # 英語ドキュメント
│   ├── README.md   # 英語版ユーザーガイド
│   └── development.md  # 英語版開発者ガイド
└── ja/             # 日本語ドキュメント
    ├── README.md   # 日本語版ユーザーガイド
    └── development.md  # 日本語版開発者ガイド
```

### 3. プロジェクト可視性の向上

- pepy.techバッジの追加によるPyPIダウンロード統計の表示
- 重要なプロジェクトURLの整理と追加

### 4. ドキュメント設計原則

#### ユーザー向けドキュメント (`docs/en/README.md`, `docs/ja/README.md`)

- インストール方法
- 基本的な使用例
- APIリファレンス概要
- 開発者ドキュメントへの明確なリンク

#### 開発者向けドキュメント (既存の `development.md`)

- 開発環境構築
- コントリビューションガイドライン
- アーキテクチャ詳細

## 結果

### 期待される効果

1. **コントリビューション品質の向上**
   - 標準化されたテンプレートによる一貫したIssue/PR作成
   - レビュープロセスの効率化

2. **プロジェクト可視性の向上**
   - pepy バッジによるダウンロード統計の表示
   - プロジェクト採用状況の可視化

3. **国際化対応の強化**
   - 英語・日本語両方でのドキュメント提供
   - グローバルユーザーへのアクセシビリティ向上

4. **新規ユーザー・コントリビューターの参入障壁低下**
   - 明確なドキュメント構造
   - 段階的学習パスの提供（ユーザー → 開発者）

### 実装詳細

#### テンプレート言語

- 英語で統一（国際的なOSSプロジェクトとしての標準に準拠）
- 日本語コメントは削除し、英語に翻訳

#### 品質保証

- PRテンプレートに品質チェックリストを含める
- CI/CD ワークフローとの連携

#### 段階的移行

- 既存ドキュメントとの互換性維持
- 新規ドキュメントへの適切なリンク設置

## 参考資料

- [GitHub Issue Templates](https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests/about-issue-and-pull-request-templates)
- [pepy - PyPI download statistics](https://pepy.tech/)
- [Open Source Guide - Best Practices for Maintainers](https://opensource.guide/best-practices/)

## 関連ファイルのパス

### 初期実装時 (2025-07-31)

#### GitHub テンプレート

- `.github/ISSUE_TEMPLATE/BUG_REPORT.md` (新規)
- `.github/ISSUE_TEMPLATE/DEVELOP_REQUEST.md` (新規、後にFEATURE_REQUESTに変更)
- `.github/PULL_REQUEST_TEMPLATE.md` (新規)

#### ドキュメント構造

- `docs/en/README.md` (新規)
- `docs/ja/README.md` (新規)
- `README.md` (pepy バッジとURL追加)

#### ADR

- `docs/adrs/adr-012-github-templates-and-documentation-structure.md` (新規)
