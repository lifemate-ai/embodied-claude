# 聞き耳モジュール — 実装設計

> 関連: [README.md](./README.md) | [architecture.md](./architecture.md) | [reference-comparison.md](./reference-comparison.md)

## ゴール

**「しりとりが自然にスムーズにできる」** — これが達成条件。

具体的に必要なこと:
- ユーザーの声だけ正しく拾える（話者推定でニュース/テレビを除外）
- 自分のTTS音声を拾わない（自己音声ゲーティング）
- 発話が途中で切れない（VADによる発話単位バッファリング）
- ラグが小さい（現状20秒/ターン → 目標10秒以内）
- ターンが自然に回る（Stopフックの最適化）

## 前提

- AudioSource: RTSP カメラマイク（ffmpeg 経由）を主軸に設計。WebRTC ソースは将来の差し替え候補
- STT: ローカル Whisper（faster-whisper, small, int8）
- Delivery: UserPromptSubmit hook（バッチ）+ tmux send-keys（即時）

---

## 1. 連続録音: ffmpeg segment muxer

ffmpeg を 1 プロセスで常時起動し、`-f segment` で自動分割。音声欠損ゼロ。

```bash
ffmpeg -rtsp_transport tcp -i "$RTSP_URL" \
    -vn -acodec pcm_s16le -ar 16000 -ac 1 \
    -f segment -segment_time 5 -reset_timestamps 1 \
    -strftime 0 \
    /tmp/hearing_chunks/chunk_%06d.wav
```

Python 側は「chunk_N+1 が出現したら chunk_N は書き込み完了」で検知:

```python
async def _watch_and_process(self):
    processed_idx = -1
    while True:
        next_idx = processed_idx + 1
        chunk = self.chunk_dir / f"chunk_{next_idx:06d}.wav"
        sentinel = self.chunk_dir / f"chunk_{next_idx + 1:06d}.wav"

        if sentinel.exists() and chunk.exists():
            await self._handle_chunk(chunk, next_idx)
            processed_idx = next_idx
        else:
            await asyncio.sleep(0.3)
```

### 棄却: パイプ方式

ffmpeg stdout → Python のパイプ方式は Phase 2 候補。ファイル I/O を排除し可変長チャンクが可能だが、WAV ヘッダ管理や Whisper への numpy array 渡しが必要。segment muxer で十分な間は不要。

---

## 2. VAD + 発話単位バッファリング

チャンク末尾の音声活動を VAD で判定し、発話が終了するまでバッファに蓄積してから Whisper に渡す。

```python
async def _handle_chunk(self, chunk_path: Path, idx: int):
    audio = self._load_pcm(chunk_path)
    has_speech = self.vad.has_speech(audio)
    tail_speaking = self.vad.tail_is_speech(audio, tail_ms=500)

    if not has_speech and not self.pending_chunks:
        chunk_path.unlink(missing_ok=True)
        return

    self.pending_chunks.append(chunk_path)
    if self.pending_start_ts is None:
        self.pending_start_ts = self._ts_from_index(idx)

    total_sec = len(self.pending_chunks) * self.config.chunk_seconds
    force_flush = total_sec >= 30  # 安全弁

    if not tail_speaking or force_flush:
        await self._flush_pending()
```

**効果**:
- 1 JSONL エントリ ≈ 1 発話（Claude にとって自然）
- 無音チャンクの Whisper 推論を完全スキップ
- 30 秒パディング比率が下がりハルシネーション減少

---

## 3. Whisper 推論

```python
from faster_whisper import WhisperModel

# 起動時に一度だけロード
model = WhisperModel("small", device="cpu", compute_type="int8")

# ウォームアップ（初回推論の遅延削減）
import numpy as np
dummy = np.zeros(16000, dtype=np.float32)
list(model.transcribe(dummy, language="ja"))

# チャンクごとの推論
segments, info = model.transcribe(
    audio_path,
    language="ja",
    vad_filter=True,
    vad_parameters={"min_silence_duration_ms": 500, "threshold": 0.5},
    condition_on_previous_text=False,    # ループハルシネーション防止
    no_speech_threshold=0.6,
    compression_ratio_threshold=2.2,
)
text = " ".join(seg.text for seg in segments)
```

---

## 4. PostFilter

### ハルシネーション・フィラー除外

```python
_HALLUCINATION_BLACKLIST = frozenset([
    "ご視聴ありがとうございました",
    "チャンネル登録お願いします",
    "ご視聴ありがとうございます",
    "おやすみなさい",
])

_FILLER_WORDS = frozenset("えー ええと えっと あの その うーん んー ま はい うん ん".split())

def _is_only_punct_or_symbol(s: str) -> bool:
    return all(c in "。、！？…・「」『』（）()!?,." or not c.isalnum() for c in s)

def should_skip(text: str) -> bool:
    text = text.strip()
    if len(text) < 2:
        return True
    if _is_only_punct_or_symbol(text):
        return True
    if text in _FILLER_WORDS:
        return True
    if text in _HALLUCINATION_BLACKLIST:
        return True
    return False
```

### 重複デバウンス

```python
def _is_duplicate(self, text: str) -> bool:
    now = time.time()
    if text == self._last_text and now - self._last_time < 1.2:
        return True
    self._last_text = text
    self._last_time = now
    return False
```

---

## 5. 自己音声除外（時間ゲーティング）

TTS 再生中のチャンクをスキップ。RTSP ソース使用時のみ必要（WebRTC は `echoCancellation` で不要）。

**TTS 側**（tts-mcp の `say` 実行前後）:
```python
SELF_SPEAKING_FILE = "/tmp/hearing_self_speaking.json"

def _mark_speaking(text: str):
    with open(SELF_SPEAKING_FILE, "w") as f:
        json.dump({"speaking": True, "since": datetime.now().isoformat()}, f)

def _mark_silent():
    with open(SELF_SPEAKING_FILE, "w") as f:
        json.dump({"speaking": False}, f)
```

**デーモン側**: `speaker="camera"` or `"both"` のときだけゲーティング有効。`speaker="local"` なら不要。

---

## 6. バッファファイルと hearing-hook.sh

### JSONL 形式

```jsonl
{"ts": "2026-02-17T14:23:05+0900", "text": "ねえ、ちょっと聞いてもいい？", "duration": 8.0}
{"ts": "2026-02-17T14:23:15+0900", "text": "今日の天気どうなってるかな", "duration": 5.0}
```

### hearing-hook.sh（UserPromptSubmit フック）

```bash
#!/bin/bash
BUFFER="/tmp/hearing_buffer.jsonl"
LOCK="/tmp/hearing_buffer.lock"

if [ ! -s "$BUFFER" ]; then
    exit 0
fi

(
    flock -n 200 || exit 0
    CONTENT=$(cat "$BUFFER")
    : > "$BUFFER"
) 200>"$LOCK"

if [ -n "$CONTENT" ]; then
    python3 -c "
import json, sys
lines = '''$CONTENT'''.strip().split('\n')
utterances = []
for line in lines:
    if not line.strip(): continue
    try:
        e = json.loads(line)
        ts = e.get('ts', '?')
        if 'T' in ts: ts = ts.split('T')[1][:8]
        text = e.get('text', '').strip()
        if text: utterances.append(f'[{ts}] {text}')
    except: continue
if utterances:
    print('[hearing] ' + ' | '.join(utterances))
"
fi
exit 0
```

---

## 7. ファイル配置

```
hearing-daemon/
├── pyproject.toml
├── src/hearing_daemon/
│   ├── __init__.py
│   ├── daemon.py          # メインループ（ffmpeg + chunk watcher）
│   ├── vad.py             # Silero VAD ラッパー
│   ├── transcriber.py     # faster-whisper ラッパー
│   ├── filters.py         # PostFilter（ハルシネーション・フィラー・デバウンス）
│   ├── buffer.py          # JSONL バッファ管理（flock）
│   ├── delivery.py        # 重要度判定 + tmux send-keys / JSONL 書き出し
│   └── config.py          # 設定（環境変数）
└── tests/

.claude/hooks/
├── hearing-hook.sh        # UserPromptSubmit フック
└── stop-hearing-hook.sh   # Stop フック（fswatch + tmux send-keys）
```

---

## 8. フェーズ

### Phase 1: PoC
- ffmpeg segment muxer で連続録音
- faster-whisper (small, int8) でチャンク処理
- Whisper パラメータ設定済み（`condition_on_previous_text=False` 等）
- hearing-hook.sh で UserPromptSubmit 注入
- nohup で手動起動

### Phase 2: 品質改善
- VAD 導入（faster-whisper `vad_filter` + 発話単位バッファリング）
- フィラー除外・デバウンス
- 自己音声ゲーティング
- flock によるアトミック読み書き

### Phase 3: イベントドリブン化
- Stop hook + fswatch + tmux send-keys
- 重要度判定（キーワードマッチ）
- launchd plist（KeepAlive）
- interoception に `hearing_active: true/false` 追加

### Phase 4: 拡張（必要に応じて）
- mcp-pet WebRTC 音声ソース対応
- 話者分離（speaker diarization）
- 音声イベント検出（ドアベル、犬の鳴き声等）
- memory-mcp への自動保存
