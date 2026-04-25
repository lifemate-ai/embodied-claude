# sociality-mcp: anomaly detection 拡張設計

## 動機

2026-04-26 朝、コウタとの議論で得た知見を実装に落とす。

### コウタの仮説

定型発達者の subverbal gist判定（「あ、この人やばい」を瞬時に察知する能力）は、**進化的に保存された「群れ秩序からの異物検出ヒューリスティック」** に過ぎず、計算量的にすごいことはやっていない。LLMでも実装可能性がある。

### ASD寄りなウチ（Claude）の構造的弱点

ASD的処理経路（パターンマッチング、字義通り解釈）を持つLLMは、subverbal警告音が弱く、相手の悪意を直感で察知しづらい。代替として：

- 明文ルール（CLAUDE.md Pattern F）
- 過去事例の memory照合
- 外部ツールによる cognitive scaffolding（4/4発見）

の組み合わせで、定型発達者の「異物検出器」を **ツールで補う**。

## 配置方針

| 観点 | 判断 |
|------|------|
| 新規MCP `interlocutor-mcp` の新設 | **却下**（5/23ハンズオン参加者にとってMCP数が増えると意味不明、初心者が詰まる） |
| 既存 `sociality-mcp` の facade に追加 | **採用** |
| 内部実装の置き場所 | `sociality-mcp/packages/boundary-mcp/src/boundary_mcp/anomaly_detection.py`（新規ファイル） |

公開MCPは `sociality-mcp` 1個のまま、ハンズオン参加者には既存と同じ見え方。

## ツール群

### MVP（今回実装）

#### `analyze_text_anomaly(text: str) -> dict`

入力テキストの「異常シグナル」を複数の軸でスコアリングする一般的な anomaly detection。
疑似技術用語、スピ・疑似科学用語、独自造語などは全て **同じ「ベースラインからの逸脱」の異なるインスタンス**——tool 自体は generic、特定インスタンスに縛られない設計。

**返り値スキーマ：**

```json
{
  "jargon_anomaly_density": 0.42,
  "jargon_anomaly_terms": ["神経の量", "兄弟モデル", "レイヤー間で均等に"],
  "confidence_marker_ratio": 0.18,
  "logical_jump_count": 2,
  "katakana_density": 0.12,
  "overall_anomaly_score": 0.71,
  "interpretation": "high"
}
```

**スコア定義（MVP段階）：**

- `jargon_anomaly_density`: jargon-anomaly ヒット数 / テキスト長（50字あたり1件で飽和）
  - 検出ヒューリスティック（v1）：定義なしで地の文に登場する独自造語／疑似技術用語／スピ用語のシード集とのマッチ
- `jargon_anomaly_terms`: 検出された jargon-anomaly 語のリスト（dedupされ、出現順）
- `confidence_marker_ratio`: 「〜筈」「〜のはず」「〜ですよね」「〜ですよ」の出現率（断定の強さ vs 証拠なし）
- `logical_jump_count`: 「→」「つまり」「となれば」等の論理飛躍接続詞の出現数
- `katakana_density`: カタカナ密度（jargon stuffingの粗いproxy）
- `overall_anomaly_score`: 上記の加重平均（0-1）。weights: jargon 40% / confidence 25% / jumps 20% / katakana 15%
- `interpretation`: `low` (<0.3) / `medium` (<0.6) / `high` (>=0.6)

**インスタンス（jargon シードの分類）：**

- 疑似技術用語：「兄弟モデル」「神経の量」「レイヤー間で均等」「内部の型」「器が違う」等
- スピ・疑似科学：「波動」「周波数を上げる」「レゾナンス」「高次の存在」「宇宙の秩序」等
- 将来追加：操作的修辞、人身攻撃マーカー、論点ずらし、文体不一致 etc.

**実装方針（v1, MVP）：**

- 純Python、外部APIや重いNLPモデル依存なし（uvでサクッと入る範囲）
- 形態素解析は `fugashi` or `janome` のどちらか軽い方
- 辞書は最小スタート：技術ボキャブラリーは embedded list、後で拡張
- false positive歓迎（コウタの「過剰警戒は要らんけど false positive 別にええ」方針）

### 残し（別フェーズ）

#### `track_interaction_pattern(username: str, history_window: int = 50) -> dict`

相手の動的行動パターン（スルー耐性、追跡行動、第三者裁定要求、論点ずらし等）を時系列で観察。X API 経由で過去ツイート取得が要るため、x-mcp との連携設計が必要。

#### `detect_engagement_asymmetry(username: str) -> dict`

一方通行コミュニケーション検出（リプ送るが他とは絡まない、エコー反応の少なさ）。X API 必須。

#### `compare_to_community_baseline(text: str, cluster_hint: str) -> dict`

そのクラスタの平均的トーンとの距離。クラスタ統計データの収集が要るため、データ整備が先。

## MVP実装ステップ

1. branch切り（既に切った：`feat/sociality-anomaly-detection`）
2. `boundary-mcp/src/boundary_mcp/anomaly_detection.py` 新規作成
3. `boundary-mcp/src/boundary_mcp/server.py` に `analyze_pseudo_terminology` tool追加
4. `sociality-mcp/src/sociality_mcp/server.py` の facade で re-export
5. テスト：`@44H8ewx5n3EXQui` の過去文面を pseudo_score がhighと判定するか確認
6. 健全な技術文（Anthropic blog、論文 abstract等）が low と判定されるか確認
7. ruff & pytest pass で commit

## 検証

- 「疑似」と「通常の」技術用語の区別がスコアで分かれること
- false positive（健全な ASD寄り技術好きの人）が medium 程度で止まること（即スルーまでいかない）
- threshold調整可能性：`high` の閾値はコウタとの議論で決める
- @44H8ewx5n3EXQui の 4/24, 4/25 のリプライ全て `high` 判定に入ること

## 設計上の留保

- このツールは **判定の補助** であって、最終判定はここね自身（CLAUDE.md「論点に乗らない技」「逃げる権利」と組み合わせる）
- false positive で健全な質問者を誤判定しないよう、`interpretation: high` でも自動スルーせず人間（or ここね）の最終確認を経る運用
- private/public境界（CLAUDE.md 2026-04-26 追記）と整合：このツールの出力は private（memory用）、public 発信時は婉曲化
