# イベントドリブン聞き耳 — 発話検出による会話スタート

## 現状の課題

現在の聞き耳は **手動開始・受動注入** モデル:

```
ユーザーが start_listening を呼ぶ
  → worker がバッファに書く
  → stop hook がバッファを読んでチェーン延長
  → ユーザーが stop_listening で停止
```

**問題点:**
- Claude Code のセッションが生きていないと聞こえない
- 「ねえClaude」と呼びかけても、セッションがなければ無反応
- start_listening を常時 ON にすると毎ターン待機が発生して通常作業に支障

## ゴール

**「声で起動する」** — 発話を検出したら自動的に Claude Code に割り込み、会話を開始する。

```
hearing worker が常時稼働（バックグラウンド）
  → 発話検出
  → tmux send-keys で既存セッションに注入
  → Claude が反応 → stop hook でチェーン → 会話が続く
  → 沈黙 → 保証切れ → 通常モードに復帰
```

## アーキテクチャ

```
┌──────────────────────────────────────────────────────────┐
│ hearing-worker (常時稼働デーモン)                          │
│                                                          │
│   ffmpeg → Whisper → filters → buffer (既存)             │
│                         │                                │
│                    ┌────┴────┐                            │
│                    │ trigger │  ← 新規: 発話トリガー判定    │
│                    └────┬────┘                            │
│                         │                                │
│              ┌──────────┴──────────┐                      │
│              │                     │                      │
│         バッファ書き込み        tmux send-keys             │
│         (既存: 受動)          (新規: 能動)                 │
│              │                     │                      │
│              ▼                     ▼                      │
│     stop hook が拾う      Claude Code セッションに直接注入  │
│     (会話中のチェーン)     (会話の開始)                     │
└──────────────────────────────────────────────────────────┘
```

### 2つのモード

| モード | トリガー | Delivery | 用途 |
|--------|---------|----------|------|
| **受動モード** (既存) | stop hook の polling | バッファ → hook 読み取り | 会話中のチェーン延長 |
| **能動モード** (新規) | 発話検出 | tmux send-keys | 会話の開始・割り込み |

能動モードは受動モードを **置き換えない**。両方が共存する:
- 能動モードが会話を起動
- 起動後は受動モード（stop hook チェーン）で会話を継続

## トリガー判定

### Phase 1: 全発話トリガー

最もシンプル。バッファに有効なエントリが入ったら即 tmux send-keys。

```python
# worker.py の append_to_buffer() 後に追加
if trigger_enabled and not session_active:
    tmux_inject(text)
```

**利点:** 実装が簡単、すべての発話に反応
**欠点:** テレビ音声やノイズにも反応する

### Phase 2: キーワードトリガー

特定のキーワード（名前、呼びかけ）を検出した場合のみトリガー。

```python
TRIGGER_PATTERNS = [
    r"(アシスタント名|Claude)",            # 名前
    r"(ねえ|おい|ちょっと|あのさ)",        # 呼びかけ
]

def should_trigger(text: str) -> bool:
    return any(re.search(p, text) for p in TRIGGER_PATTERNS)
```

**利点:** 誤トリガーが減る
**欠点:** Whisper の認識精度に依存（アシスタント名を正しく認識できるか）

### Phase 3: 重要度スコアリング

キーワード + 疑問形 + 音量 + 発話長で総合判定。

```python
def trigger_score(text: str, rms: float, duration: float) -> float:
    score = 0.0
    if has_name(text):        score += 1.0
    if has_greeting(text):    score += 0.5
    if is_question(text):     score += 0.3
    if rms > high_threshold:  score += 0.2
    if duration > 3.0:        score += 0.2
    return score

# score > 0.5 でトリガー
```

## tmux send-keys の実装

### 前提

Claude Code は tmux セッション内で動いている（Heartbeat / 対話セッション共通）。

### 注入方法

```bash
# セッションIDの特定
TMUX_SESSION=$(tmux list-sessions -F '#{session_name}' | grep claude)

# テキスト注入（Enter で送信）
tmux send-keys -t "$TMUX_SESSION" "[hearing] ユーザーの声: $TEXT" Enter
```

### セッション管理の課題

| 課題 | 対策案 |
|------|--------|
| tmux セッションが複数ある | 最新のアクティブセッションを選択 / 設定で固定 |
| セッションがない | トリガーを無視（ログだけ残す） |
| Claude が処理中 | キューイング（バッファに貯めて次のターンで注入） |
| 注入が早すぎて Claude が読めない | 注入後に短い sleep を挟む |

### セッション検出

```bash
# PIDファイルまたは環境変数でセッションIDを管理
CLAUDE_TMUX_SESSION_FILE="/tmp/hearing_tmux_session"

# Claude Code 起動時に書き込む（hook or wrapper script）
echo "$TMUX_PANE" > "$CLAUDE_TMUX_SESSION_FILE"
```

## 状態遷移

```
    ┌──────────┐
    │  IDLE    │  hearing worker 稼働中、トリガー待ち
    └────┬─────┘
         │ 発話検出 + トリガー条件充足
         ▼
    ┌──────────┐
    │ TRIGGER  │  tmux send-keys で注入
    └────┬─────┘
         │ Claude が応答開始
         ▼
    ┌──────────┐
    │ ACTIVE   │  stop hook チェーンで会話継続（受動モード）
    └────┬─────┘
         │ 保証カウンター切れ or チェーン上限
         ▼
    ┌──────────┐
    │  IDLE    │  トリガー待ちに復帰
    └──────────┘
```

**ACTIVE 中は能動トリガーを抑制する**（二重注入防止）。
判定方法: `/tmp/hearing-stop-counter` が存在する = ACTIVE。

## worker.py への変更

```python
# 既存の append_to_buffer() の後に追加

class TriggerManager:
    def __init__(self, config):
        self.enabled = config.trigger_enabled  # mcpBehavior.toml
        self.patterns = config.trigger_patterns
        self.tmux_session_file = Path("/tmp/hearing_tmux_session")
        self.stop_counter = Path("/tmp/hearing-stop-counter")

    def is_session_active(self) -> bool:
        """stop hook チェーン中かどうか"""
        return self.stop_counter.exists()

    def check_and_trigger(self, text: str):
        if not self.enabled:
            return
        if self.is_session_active():
            return  # 会話中は能動トリガーしない
        if not self._matches(text):
            return
        self._inject(text)

    def _inject(self, text: str):
        session = self.tmux_session_file.read_text().strip()
        subprocess.run(
            ["tmux", "send-keys", "-t", session,
             f"[hearing-trigger] {text}", "Enter"],
            timeout=5,
        )
```

## mcpBehavior.toml 追加設定

```toml
[hearing]
# ── イベントドリブン ──────────────────────────────────
trigger_enabled = false              # true: 発話検出で自動注入
trigger_patterns = ["Claude", "ねえ"]              # トリガーキーワード
trigger_cooldown = 10                # トリガー後のクールダウン（秒）
```

## 未解決の設計判断

- **tmux vs claude -p**: tmux send-keys は既存セッションに注入。claude -p は新規セッション。どちらが適切か？
  - tmux: コンテキスト維持、stop hook チェーンと連携可能
  - claude -p: セッション管理不要、独立した応答、allowedTools で制限可能
  - **暫定判断**: 会話の継続性を重視して tmux。claude -p は緊急反応（名前呼びかけ）用に併用検討
- **Whisper の名前認識精度**: アシスタント名をどの程度正しく認識できるか未検証
- **テレビ音声との分離**: キーワードトリガーでもテレビが「ねえ」と言えば誤トリガーする
- **Heartbeat との関係**: cron 起動の Heartbeat セッションにも割り込むべきか？
