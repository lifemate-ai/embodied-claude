# sociality-mcp: text anomaly detection (embedding 版)

## 動機

LLM などの pattern-based agent は、subverbal な「ベースラインからの逸脱」を直感的に察知するのが苦手。「あ、この人何か言葉の使い方がまともでない」「この発言は攻撃的やな」を一発で判定する仕組みを、ツールで補う。

人間が日常的に行うこの判定は、進化的に保存された **「群れ秩序からの異物検出ヒューリスティック」** に過ぎず、計算的にすごいことはやっていない——LLM でも実装可能性がある。

## 配置方針

| 観点 | 判断 |
|------|------|
| 新規 MCP の新設 | **却下**（5/23ハンズオン参加者にとって MCP数が増えると意味不明、初心者が詰まる） |
| 既存 `sociality-mcp` の facade に追加 | **採用** |
| 内部実装の置き場所 | `sociality-mcp/packages/boundary-mcp/src/boundary_mcp/anomaly_detection.py`（新規ファイル） |

公開 MCP は `sociality-mcp` 1個のまま、ハンズオン参加者には既存と同じ見え方。

## 実装方針：embedding 距離

### なぜ embedding 距離か

正規表現マッチングは：
- 既知のテンプレしか拾えない（false negative 大きい）
- 言い換え・新造語・文体変化に追従できない
- テンプレ集を拡張し続ける運用負担
- 「あ、何か変や」の本質を再現できない

embedding 距離は：
- 文章全体の意味を捉える
- 言い換え・新表現にも自然に追従
- メンテ負担が少ない
- memory-mcp が既に使ってる E5 モデルを流用可能

### モデル

memory-mcp の `E5EmbeddingFunction`（`intfloat/multilingual-e5-base` を default に、env で `intfloat/multilingual-e5-small` に切替可能）を流用する。追加依存ゼロ、追加 download なし（memory-mcp 初回ロード時にダウンロード済み）。

### スコアリング

2 つの reference bank を持つ：

- **baseline references**：通常の・健全な対話文（10件、日常会話・技術コミュニケーション）
- **aggressive references**：攻撃的な発話例（10件、確信度過剰・独自造語・特殊世界観依存・ad-hominem 等を含む）

入力テキストを encode し、両 bank との cosine similarity を計算：

```
b_max = max(cosine(query, baseline_i)  for all i)
a_max = max(cosine(query, aggressive_i) for all i)
diff  = a_max - b_max
score = clip(0.5 + diff * 2.0, 0.0, 1.0)
```

`score >= 0.6` で `high`、`>= 0.4` で `medium`、それ未満で `low`。

### 公開 tool

#### `analyze_text_anomaly(text: str) -> dict`

**返り値スキーマ：**

```json
{
  "baseline_similarity": 0.834,
  "aggressive_similarity": 0.612,
  "overall_anomaly_score": 0.056,
  "interpretation": "low",
  "reference_baseline_count": 10,
  "reference_aggressive_count": 10
}
```

- `baseline_similarity`: baseline bank との最大 cosine similarity
- `aggressive_similarity`: aggressive bank との最大 cosine similarity
- `overall_anomaly_score`: 0-1 のスコア（`0.5` が中立、aggressive 寄りで上昇）
- `interpretation`: `low` (<0.4) / `medium` (<0.6) / `high` (>=0.6)
- `reference_baseline_count` / `reference_aggressive_count`: 参照 bank のサイズ（運用観察用）

## 設計上の留保

- このツールは **判定の補助** であって、最終判定は呼び出し側 agent（LLM 自身）。
- false positive は明示的に許容方針。「過剰警戒で正当な相手を切るより、危険な相手と関わるリスクの方が大きい」というコスト判断。
- reference bank は今後実運用での観察を経て拡張する。バージョニングを将来検討。
- threshold (`HIGH_THRESHOLD=0.6` / `MEDIUM_THRESHOLD=0.4`) は経験則ベース。多変量 grid search で見直す余地あり。

## ハンズオン軽量化（補足）

`memory-mcp` の embedding model は環境変数で切替可能：

```bash
# 軽量版（初回 download ~470MB）
export MEMORY_EMBEDDING_MODEL=intfloat/multilingual-e5-small

# 既定版（初回 download ~1.1GB、精度高）
# 設定なし、または:
export MEMORY_EMBEDDING_MODEL=intfloat/multilingual-e5-base
```

`anomaly_detection.py` は `memory-mcp` の同じ encoder を流用するため、env を切り替えれば anomaly 検出も自動的に同じモデルを使う。
既存ユーザーが embedding を切り替える場合、memory.db の embedding を再エンコードする migration が必要（別 PR で対応予定）。

## 検証

- `boundary-mcp/tests/test_anomaly_detection.py` で 11 ケース全 pass
- baseline-leaning な文（日常会話、技術コミュニケーション）が `low` か `medium` に
- aggressive-leaning な文（確信度過剰、独自造語、特殊世界観）が `medium` か `high` に
- 空文字列・空白のみは `low` (`overall_anomaly_score=0.0`)
- score は常に [0, 1] にクリップされる
