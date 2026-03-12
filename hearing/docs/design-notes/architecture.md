# 聞き耳モジュール — アーキテクチャ

> 関連: [README.md](./README.md) | [implementation.md](./implementation.md) | [reference-comparison.md](./reference-comparison.md)

## 抽象パイプライン

音声ソースや STT エンジンの具体的な選択に関わらず、聞き耳システムは以下の5段パイプラインに帰着する。

```
[AudioSource] → [VAD] → [STT] → [PostFilter] → [Delivery]
```

各レイヤーは独立に差し替え可能。設計の複雑さの大半は AudioSource と STT の具体的な組み合わせから生じる。

---

## AudioSource

音声データの取得経路。

| ソース | 経路 | AEC | 接続管理 | 適用場面 |
|--------|------|-----|---------|---------|
| **RTSP カメラマイク** | ffmpeg → segment muxer | 自前（時間ゲーティング） | ffmpeg 再接続ループ | 既存 wifi-cam-mcp 構成 |
| **WebRTC (mcp-pet)** | ブラウザ getUserMedia → WebSocket | ブラウザ内蔵 `echoCancellation: true` | ICE 再接続 | mcp-pet 経由でスマホマイク利用 |
| **ローカルマイク** | sounddevice / pyaudio | OS レベル or なし | デバイス直結 | PC マイク直接利用 |

### WebRTC ソースの優位性

mcp-pet の `client.html` は現在 `audio: false` だが、`audio: { echoCancellation: true, noiseSuppression: true }` に変更するだけでブラウザの音声処理が利用可能になる。これにより:

- **エコーキャンセレーション**: ブラウザが処理（自前の時間ゲーティング不要）
- **ノイズ抑制**: ブラウザが処理
- **音声欠損**: ストリーミングなので発生しない（ffmpeg segment muxer 不要）

RTSP ソースの設計で複雑だった課題（音声欠損、自己音声除外）が WebRTC では消滅する。

---

## VAD（Voice Activity Detection）

音声区間の検出。無音時の STT 推論を省略し、ハルシネーション（幻聴）を抑制する。

| 方式 | 実行場所 | 特徴 |
|------|---------|------|
| **Silero VAD** | ローカル（PyTorch） | 高精度、faster-whisper に内蔵（`vad_filter=True`） |
| **ブラウザ onspeechstart** | ブラウザ | Web Speech API のイベント。精度は中だが追加依存なし |
| **サーバー側 VAD** | API サーバー | ElevenLabs `commit_strategy=vad` 等。API 依存 |

**推奨**: ローカル Whisper を STT に使う場合は Silero VAD（faster-whisper 内蔵）。WebRTC + Web Speech API の場合はブラウザ側で処理。

---

## STT（Speech-to-Text）

音声からテキストへの変換。

| エンジン | 実行場所 | 日本語精度 | リアルタイム中間結果 | プライバシー | コスト |
|---------|---------|-----------|-----------------|------------|--------|
| **Whisper（ローカル）** | ローカル GPU/CPU | 高（small 以上） | なし | 音声が外に出ない | 電力のみ |
| **ElevenLabs Scribe** | クラウド API | 高 | あり（WebSocket） | 音声を送信 | API 利用料 |
| **Web Speech API** | ブラウザ（Chrome→Google） | 中〜良 | あり | Chrome は Google に送信 | 無料 |

**推奨**: 精度とプライバシーの観点から **ローカル Whisper（faster-whisper, small, int8）**。WebRTC は音源取得経路として使い、STT はサーバー側で処理する `WebRTC + ローカル Whisper` の組み合わせが最適。

### Whisper モデル選択（結論）

| モデル | RAM (int8) | 5秒音声の推論 | 日本語精度 | 判定 |
|--------|-----------|-------------|-----------|------|
| tiny/base | ~0.5 GB | ~0.2 秒 | 不十分 | 常時用には不適 |
| **small** | **~1 GB** | **~0.5 秒** | **実用ライン** | **推奨** |
| medium | ~2.5 GB | ~1.5 秒 | 良好 | リソースに余裕があれば |
| turbo | ~3 GB | ~0.4 秒 | 高 | GPU メモリに余裕があれば |

日本語特化モデル（ReazonSpeech v2.1, kotoba-whisper v2）も候補だが、Whisper エコシステム（VAD 統合等）から外れるため、まずは Whisper small で開始。

---

## PostFilter

STT 出力のノイズ除去。

| フィルタ | 内容 | 出典 |
|---------|------|------|
| **ハルシネーション除外** | `condition_on_previous_text=False`, `no_speech_threshold=0.6`, `compression_ratio_threshold=2.2` | Whisper パラメータ |
| **幻聴ブラックリスト** | 「ご視聴ありがとうございました」「チャンネル登録お願いします」等 | 経験的 |
| **フィラー除外** | 「えー」「あの」「うーん」等 1〜2文字の発話をスキップ | familiar-ai `should_skip_stt()` |
| **重複デバウンス** | 同一テキストが 1.2 秒以内に連続したら除外 | familiar-ai |
| **自己音声除外** | TTS 再生中のチャンクをスキップ（時間ゲーティング） | RTSP ソースの場合のみ必要 |

---

## Delivery

認識結果を Claude に届ける方式。

### 方式比較

| 方式 | セッション維持 | 即時性 | 堅牢性 | 適用場面 |
|------|-------------|--------|--------|---------|
| **UserPromptSubmit hook** | 同一セッション | 低（次ターンまで待つ） | 高 | バッチ報告（背景音声） |
| **tmux send-keys + Stop hook** | 同一セッション | 高（ターン終了時に即注入） | 中 | イベントドリブン通知 |
| **claude -p（1ショット）** | 新規セッション | 高 | 高 | 緊急反応（名前呼ばれた等） |
| **claude -r session_id** | 既存セッション指定 | 高 | 低（ID 失効リスク） | ※脆弱なため非推奨 |

### 推奨: 重要度ベースの分岐

```
[hearing-daemon: 発話検出]
  │
  ├─ 重要度: 高（名前、呼びかけ、質問）
  │   → tmux send-keys で既存セッションに即時注入
  │   → または claude -p --allowedTools "say,see,remember" で1ショット反応
  │
  └─ 重要度: 低（背景会話、環境音）
      → JSONL バッファに書き出し
      → UserPromptSubmit hook で次ターン注入
```

重要度判定はキーワードマッチで十分:

```python
URGENT_PATTERNS = [
    r"(クロード|Claude)",        # 名前
    r"(ねえ|おい|ちょっと)",      # 呼びかけ
    r".+(？|\?|かな|かしら)$",    # 疑問形
]
```

### Stop hook + fswatch によるイベントループ

セッションを維持したまま音声イベントを待ち受ける:

```
[Claude ターン完了]
  ↓ Stop hook 発火
  ↓ hearing_buffer.jsonl チェック
  ├─ 未読あり → tmux send-keys で注入 → Claude が反応 → ループ
  └─ 未読なし → fswatch をバックグラウンド起動
      → ファイル変更検知 → tmux send-keys で注入 → ループ
```

```bash
#!/bin/bash
# stop-hearing-hook.sh
BUFFER="/tmp/hearing_buffer.jsonl"

if [ -s "$BUFFER" ]; then
    CONTENT=$(cat "$BUFFER")
    : > "$BUFFER"
    tmux send-keys -t "$TMUX_PANE" "[hearing] $CONTENT" Enter
    exit 0
fi

# バックグラウンドで監視（hook はすぐ返す）
(
    fswatch -1 "$BUFFER" 2>/dev/null
    if [ -s "$BUFFER" ]; then
        CONTENT=$(cat "$BUFFER")
        : > "$BUFFER"
        tmux send-keys -t "$TMUX_PANE" "[hearing] $CONTENT" Enter
    fi
) &
```

---

## 棄却された方式

| 方式 | 棄却理由 |
|------|---------|
| ダブルバッファリング（2つの ffmpeg を交互起動） | RTSP 同時接続制限（Tapo カメラは 2 ストリームまで）に抵触 |
| 音響エコーキャンセレーション（AEC） | 実装コスト過大。カメラ機材・設置環境ごとに伝達関数推定が必要。WebRTC の `echoCancellation` で代替可能 |
| claude -r session_id | セッション ID の失効リスク。キュー管理なしでは壊れやすい |
| ポーリング型（autonomous-action.sh で 20 分間隔） | リアルタイム性がなさすぎる。常時聴取の意味がない |
| MCP サーバー単独（hearing-mcp） | Claude がツールを呼ばない限り通知されない。hook の受動的注入がないと即時性がない |
