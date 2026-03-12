# Hearing Module — Architecture & Flow

## Overview

Claude Code に「耳」を与えるモジュール。カメラ（RTSP）またはPCマイクの音声を常時録音し、
Whisper で文字起こしして Claude Code のコンテキストに注入する。

## Components

```
┌─────────────────────────────────────────────────────────────────┐
│ Claude Code                                                     │
│                                                                 │
│  ┌──────────────┐    ┌──────────────────┐                       │
│  │ hearing MCP  │    │ Claude Code Hooks │                       │
│  │  server.py   │    │                  │                       │
│  │              │    │  hearing-hook.sh  │  (UserPromptSubmit)   │
│  │ start/stop   │    │  hearing-stop     │  (Stop)              │
│  │ _listening() │    │   -hook.sh       │                       │
│  └──────┬───────┘    └────────┬─────────┘                       │
│         │                     │                                 │
└─────────┼─────────────────────┼─────────────────────────────────┘
          │ spawn               │ read
          ▼                     ▼
┌─────────────────┐   ┌──────────────────┐
│ hearing-worker   │   │ hearing_buffer   │
│  (subprocess)    │──▶│   .jsonl         │
│                  │   │  (shared file)   │
│  ffmpeg → WAV    │   └──────────────────┘
│  Whisper → text  │
│  filters → buf   │
└─────────────────┘
```

## Process Architecture

### 1. hearing-worker (daemon subprocess)

`start_listening()` で起動される独立プロセス。

**なぜ MCP 内で直接録音しない？** MCP サーバーは Claude Code のプロセス内で動くため、
ffmpeg や Whisper のブロッキング I/O が UI をロックしてしまう。
デーモンとして別プロセスに切り出すことで、録音・転写と Claude Code の操作を並行させている。

```
hearing-worker
├── ffmpeg (segment muxer)     # 常時録音、N秒ごとにWAVファイル分割
│   └── seg_000.wav, seg_001.wav, ...
│
└── main loop (poll)           # 新しいセグメントを監視
    ├── VAD check (_rms_energy) # 無音スキップ
    ├── Whisper transcribe      # faster-whisper (CPU, int8)
    ├── filters (should_skip)   # ハルシネーション除去
    ├── debouncer               # 重複排除
    ├── tail_speech detection   # 末尾0.5秒の音声残存チェック
    └── append_to_buffer()      # JSONL書き込み (flock排他)
```

**重要**: ffmpeg segment muxer は録音と転写を並行して行う。
転写中も次のセグメントの録音は継続されるため、音声の取りこぼしがない。

### 2. hearing-hook.sh (UserPromptSubmit hook)

ユーザーがプロンプトを送信するたびに発火。バッファをドレインして
プロンプトに音声認識結果を注入する。

**なぜ UserPromptSubmit？** ユーザーがプロンプトを送信してから以降の
音声だけが認識されてほしい、というメンタルモデルに基づいている。
送信前に溜まっていたバッファはこの時点でドレインされ、
Claude が応答している間に新たに入った音声だけが stop-hook で拾われる。

```
ユーザーがプロンプト送信
  │
  ▼
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[Bash] hearing-hook.sh  (.claude/hooks/hearing-hook.sh)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  1. stdin から JSON を読み、ユーザーのプロンプトを保存
     → /tmp/hearing_user_prompt.txt (LLM フィルタ用)

  2. バッファ (/tmp/hearing_buffer.jsonl) を atomic drain
     → flock で排他、中身を読んでからファイルを空にする

  3. 有効なエントリがあれば stdout に注入:
     [hearing] chunks=N span=HH:MM:SS~HH:MM:SS text=...

  4. had_speech フラグを記録
     → /tmp/hearing_had_speech (stop-hook の参考情報)
```

**ポイント**: ドレイン後はバッファが空になるため、直後の stop-hook は
バッファ空からスタートする。これが保証カウンターが必要な理由。

### 3. hearing-stop-hook.sh (Stop hook)

Claude の応答が終わるたびに発火。バッファに新しい発話があればチェーン
（ターン延長）する。**最も複雑なコンポーネント。**

stop-hook は **Bash + 埋め込み Python** の 2 層構造になっている。

**なぜ 2 層？** Claude Code の Stop hook は JSON (`{"decision": "block"}`) を
stdout に返す必要がある。バッファ解析・フィルタ処理は Python が適しているが、
hook のライフサイクル管理（カウンターファイル、toml 読み込み、最終判定）は
Bash が担う。Python はバッファ読み取り結果を stdout に出力し、
Bash がそれを `$RESULT` として受け取って block/pass を決定する。

## Timing Flow — Stop Hook の時系列

1ターンのstop-hook処理の流れ:

```
Claude 応答完了
  │
  ▼
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[Bash] sleep 5s (*1) ← セグメントが溜まるのを待つ
  (.claude/hooks/hearing-stop-hook.sh)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  │
  ▼
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[Python] バッファ読み取りループ  (hearing-stop-hook.sh 内 heredoc)
  最大 3 回リトライ
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  try_read() でバッファを確認
    │
    ├─ HIT, tail=false ─── そのまま stdout に出力 ──▶ [A]
    │
    ├─ HIT, tail=true ──── pending に保留
    │                       sleep 3s (*2) → 次の try_read() へ
    │                         │
    │                         ├─ HIT ── 続きが来た！出力 ─▶ [A]
    │                         └─ empty ─ pending flush ───▶ [A]
    │
    └─ empty ────────────── sleep 3s (*2) → 次の try_read() へ
                              │
                              ├─ HIT ── 出力 ────────────▶ [A]
                              └─ empty ─ 最終リトライも空
                                          何も出力しない ──▶ [B]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[Bash] $RESULT 判定  (.claude/hooks/hearing-stop-hook.sh)
  Python の stdout を受け取って block/pass を決定
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  [A] $RESULT あり (発話を検出)
        → チェーンカウンター +1
        → 保証カウンターを 0 にリセット
        → block → Claude が再応答 → stop-hook 再発火

  [B] $RESULT なし (バッファ空 or 全ハルシネーション)
        │
        ├─ 保証カウンター < min_guaranteed (*4)
        │    → sleep 5s (*3) → 保証カウンター +1
        │    → block (「聞き取り待機中...」)
        │    → stop-hook 再発火
        │
        └─ 保証カウンター >= min_guaranteed (*4)
             → pass → ターン終了

  (*1) HEARING_WAIT_SECONDS     環境変数 (デフォルト: 5)
  (*2) HEARING_RETRY_WAIT       環境変数 (デフォルト: 3)
  (*3) [hearing] guaranteed_sleep  mcpBehavior.toml (デフォルト: 5)
  (*4) [hearing] min_guaranteed    mcpBehavior.toml (デフォルト: 5)
```

## Chain Guarantee — 保証カウンターの仕組み

2つの独立したカウンターが存在する:

```
チェーンカウンター (/tmp/hearing-stop-counter)
  └─ 全体の連続ターン数。HIT でも empty でも +1。上限 MAX_HEARING_CONTINUES (20)

保証カウンター (/tmp/hearing-guaranteed-counter)
  └─ 連続してバッファが空だった回数。HIT でリセット(0)。
     上限 MIN_GUARANTEED (toml設定, デフォルト5)
```

### 例: 会話の流れ

```
ターン  バッファ  チェーン  保証    動作
─────  ──────  ──────  ─────  ──────────────
  1     empty    1/20    1/10   保証待機 (block)
  2     empty    2/20    2/10   保証待機 (block)
  3     HIT      3/20    0/10   発話あり! 保証リセット (block)
  4     HIT      4/20    0/10   発話あり! 保証リセット (block)
  5     empty    5/20    1/10   保証待機 (block)
  6     empty    6/20    2/10   保証待機 (block)
  7     HIT      7/20    0/10   発話あり! 保証リセット (block)
  8     empty    8/20    1/10   保証待機 (block)
  ...
  N     empty    N/20   10/10   保証切れ → pass (ターン終了)
```

**ポイント**: HIT するたびに保証カウンターが 0 にリセットされるため、
会話が続いている限りチェーン上限(20)まで維持できる。

## tail_speech — 発話途中検出

セグメント(3秒)の境界で発話が切れる問題への対策。

```
音声波形:
  ┌────────────────┐ ┌────────────────┐
  │  "今日は天気が"  │ │ "いいですね"    │
  │     ~~~~~~▓▓▓▓▓│ │▓▓▓~~~          │
  └────────────────┘ └────────────────┘
  seg_000.wav          seg_001.wav
                  ↑
          tail 0.5秒に音声あり
          → tail_speech: true

Worker:
  _tail_rms(seg_path, 0.5) → RMS値
  tail_has_speech = tail_rms >= vad_threshold * 0.5

Stop hook (Python):
  retry 0: HIT+tail → pending に保留、3秒待つ
  retry 1: HIT      → 2チャンク統合して出力
   or
  retry 1: empty    → pending を flush して出力
```

## Buffer Format

`/tmp/hearing_buffer.jsonl` — 1行1エントリのJSON Lines:

```json
{
  "ts": "2026-03-08T22:30:10.123456+09:00",
  "text": "はいはい、これでどうかな",
  "no_speech_prob": 0.1509,
  "seg": 15,
  "tail_speech": false
}
```

## Filter Pipeline

```
Whisper出力
  │
  ├─ 空テキスト → skip
  ├─ 2文字未満 → skip
  ├─ 記号のみ → skip
  ├─ フィラー語 (えー, あの, その...) → skip
  ├─ ハルシネーション blacklist → skip
  │   ├─ デフォルト: ご視聴ありがとう, チャンネル登録, お疲れ様...
  │   ├─ モデル別: small → 気持ちいい
  │   └─ mcpBehavior.toml: hallucination_blacklist = [...]
  ├─ 繰り返しパターン (80%同一語) → skip
  ├─ Debouncer (1.5秒以内の重複) → skip
  │
  └─ LLM filter (opt-in)
      claude -p --model haiku で会話コンテキスト付き判定
      → ハルシネーションなら除去、実発話のみ返す
```

## Configuration — mcpBehavior.toml

```toml
[hearing]
whisper_model = "small"
source = "camera"                      # "local" or "camera"
segment_seconds = 3                    # ffmpeg segment length
language = "ja"
vad_energy_threshold = 0.003           # RMS threshold (0=disable)
hallucination_blacklist = ["気持ちいい"] # extra blacklist

# Stop hook
min_guaranteed = 10                    # 保証回数 (連続空でもblock)
guaranteed_sleep = 5                   # 保証待機時のsleep秒数
llm_filter = false                     # LLM filter (opt-in)
llm_filter_timeout = 20               # LLM filter timeout
```

## File Map

```
hearing/
├── src/hearing/
│   ├── server.py        # MCP server (start/stop_listening)
│   ├── worker.py        # Daemon: ffmpeg + Whisper + buffer write
│   ├── transcriber.py   # faster-whisper wrapper
│   ├── buffer.py        # JSONL buffer (flock)
│   ├── filters.py       # Hallucination filter pipeline
│   ├── config.py        # HearingConfig (toml loader)
│   └── _behavior.py     # mcpBehavior.toml reader
│
.claude/hooks/
├── hearing-hook.sh      # UserPromptSubmit: drain buffer → prompt
└── hearing-stop-hook.sh # Stop: check buffer → chain/pass
```
