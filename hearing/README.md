# hearing-mcp

Claude Code に「耳」を与える MCP サーバー。
カメラ（RTSP）または PC マイクの音声を常時録音し、Whisper で文字起こしして Claude Code のコンテキストに注入する。

## Features

- **常時録音** — ffmpeg segment muxer でギャップなし連続録音
- **リアルタイム転写** — faster-whisper (CPU, int8) で低遅延文字起こし
- **VAD** — RMS エネルギー閾値による無音スキップ
- **ハルシネーション除去** — ルールベースフィルタ + LLM フィルタ (opt-in)
- **チェーン制御** — Stop hook でターンを自動延長し、会話を途切れさせない
- **tail_speech 検出** — セグメント末尾の音声を検知し、発話途中で切らない
- **保証カウンター** — 発話の間が空いても一定回数まで待機を保証

## Usage

### 聞き耳の開始・停止

Claude Code 内で以下のツールを呼ぶ:

- **`start_listening`** — 録音デーモンを起動。以降、音声認識結果が自動的にコンテキストに注入される
- **`stop_listening`** — 録音デーモンを停止

### 注意点

- **`start_listening` 中は毎ターン待機が発生する。**
  stop-hook がバッファをチェックするため、Claude の応答後に数秒〜十数秒の待機が入る。
  これは発話を拾うための仕様。
- **取り急ぎの待機キャンセルは ESC キー。**
  stop-hook の待機中に ESC を押すと、そのターンの待機をスキップできる。
- **`stop_listening` で完全に停止。**
  毎ターンの待機をやめたいときは `stop_listening` を呼ぶ。
  デーモンが停止すれば stop-hook は即座に pass する。

## Setup

### 1. 依存関係のインストール

```bash
cd hearing
uv sync
```

ffmpeg も必要:

```bash
# macOS
brew install ffmpeg
```

### 2. MCP サーバーの登録

`.mcp.json` に hearing MCP サーバーを追加する（`.mcp.example.json` を参照）:

```json
{
  "mcpServers": {
    "hearing": {
      "command": "uv",
      "args": ["run", "--directory", "./hearing", "hearing-mcp"],
      "env": {
        "MCP_BEHAVIOR_TOML": "/path/to/your/mcpBehavior.toml"
      }
    }
  }
}
```

### 3. Claude Code hooks の登録

`.claude/settings.json` に 2 つのフックを追加する（`.claude/settings.example.json` を参照）:

- **`hearing-hook.sh`** → `UserPromptSubmit` イベント
- **`hearing-stop-hook.sh`** → `Stop` イベント（timeout は 20 秒以上を推奨）

### 4. 設定（任意）

`mcpBehavior.toml` に `[hearing]` セクションを追加する。
設定項目の詳細は後述の [Configuration](#configuration--mcpbehaviortoml) を参照。

## Configuration — mcpBehavior.toml

```toml
[hearing]
# ── 録音・転写 ──────────────────────────────────────
whisper_model = "small"            # Whisper モデル (tiny/base/small/medium/large)
source = "camera"                  # 音声ソース: "local" (PC マイク) or "camera" (RTSP)
segment_seconds = 3                # ffmpeg セグメント長 (秒)。短いほど低遅延だが CPU 負荷増
language = "ja"                    # Whisper の言語ヒント
vad_energy_threshold = 0.003       # RMS エネルギー閾値。0 で無効。カメラマイクは音量が
                                   # 小さいため低め (0.001〜0.005) に設定する

# ── チェーン制御 ────────────────────────────────────
min_guaranteed = 10                # 保証回数。バッファ空が連続してもこの回数まで
                                   # stop-hook が block して待機する。
                                   # HIT (発話検出) するたびに 0 にリセットされる
guaranteed_sleep = 5               # 保証待機時の sleep 秒数。
                                   # 保証 1 回あたりの待機時間を決める

# ── フィルタ ────────────────────────────────────────
hallucination_blacklist = []       # 追加ブラックリスト (部分一致)。
                                   # デフォルトで「ご視聴ありがとう」等は除去済み
llm_filter = false                 # true: claude -p (haiku) でハルシネーション判定。
                                   # 精度は高いが 10〜20 秒のレイテンシが加わる
llm_filter_timeout = 20            # LLM フィルタのタイムアウト秒数
```

## Architecture

詳細は [docs/architecture.md](docs/architecture.md) を参照。

```
Claude Code
├── hearing MCP (server.py)     start/stop_listening
│   └── hearing-worker          ffmpeg → Whisper → buffer (別プロセス)
├── hearing-hook.sh             drain buffer → prompt inject
└── hearing-stop-hook.sh        check buffer → chain/pass
```

## File Structure

```
hearing/
├── README.md                  # このファイル
├── docs/
│   ├── architecture.md        # 詳細アーキテクチャ・フロー図
│   └── design-notes/          # 設計時のメモ・比較資料
├── src/hearing/
│   ├── server.py              # MCP サーバー
│   ├── worker.py              # デーモン (ffmpeg + Whisper)
│   ├── transcriber.py         # faster-whisper ラッパー
│   ├── buffer.py              # JSONL バッファ (flock 排他)
│   ├── filters.py             # フィルタパイプライン
│   ├── config.py              # HearingConfig
│   └── _behavior.py           # mcpBehavior.toml リーダー
├── tests/
├── pyproject.toml
└── uv.lock
```

## Temporary Files

すべて `/tmp/` 以下に書き出される。`stop_listening` 時に主要ファイルはクリアされる。

| ファイル | 書き手 | 用途 |
|---------|--------|------|
| `hearing_buffer.jsonl` | worker | Whisper 転写結果バッファ (JSONL) |
| `hearing-daemon.pid` | worker | デーモンの PID |
| `hearing_segments/` | worker | ffmpeg が書き出す WAV セグメント (ディレクトリ) |
| `hearing_stop_offset` | stop-hook | バッファの読み取りオフセット (行番号) |
| `hearing-stop-counter` | stop-hook | チェーンカウンター (連続ターン数) |
| `hearing-guaranteed-counter` | stop-hook | 保証カウンター (連続バッファ空回数) |
| `hearing_context.json` | stop-hook | Claude の応答コンテキスト (LLM フィルタ用) |
| `hearing_user_prompt.txt` | hearing-hook | ユーザープロンプト (LLM フィルタ用) |
| `hearing_had_speech` | hearing-hook | 発話ありフラグ |
| `hearing_buffer_drain.jsonl` | hearing-hook | ドレイン時の一時コピー |
| `hearing_timing.log` | both hooks | タイミングデバッグログ |
| `hearing_stop_last_ts` | stop-hook | 前回実行タイムスタンプ |
| `hearing_hook_last_ts` | hearing-hook | 前回実行タイムスタンプ |

## Requirements

- Python 3.11+
- ffmpeg (RTSP キャプチャ用)
- faster-whisper (CPU int8)
