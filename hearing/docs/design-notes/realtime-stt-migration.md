# Segment Muxer → リアルタイム文字起こしへの移行計画

## 現状: ffmpeg segment muxer 方式

```
RTSP/マイク → ffmpeg -f segment → seg_000.wav (3秒) → Whisper バッチ推論 → バッファ
                                  seg_001.wav (3秒) → Whisper バッチ推論 → バッファ
                                  seg_002.wav (3秒) → ...
```

### この方式の限界

| 問題 | 影響 |
|------|------|
| 固定長セグメント分割 | 発話がセグメント境界で切れる。tail_speech で緩和しているが根本解決ではない |
| レイテンシ | segment_seconds(3秒) + Whisper推論(0.5秒) + wait(5秒) = 最短8.5秒 |
| WAVファイル I/O | セグメントごとにディスク書き込み→読み込み。/tmp に大量の一時ファイル |
| バッチ推論 | セグメント完成まで推論開始できない。1文字目が出るまで最低 segment_seconds 待ち |
| 中間結果なし | 「今喋ってる途中」が見えない。発話完了まで何も返らない |

## 目標: リアルタイムストリーミング STT

```
RTSP/マイク → 音声ストリーム → リアルタイム STT → 中間結果/確定結果 → バッファ
                                     │
                                     └─ VAD 統合（発話区間を自動検出）
```

### 得られるもの

- **発話単位の自然な区切り** — VAD が発話の開始・終了を検出。セグメント境界問題が消滅
- **低レイテンシ** — 発話終了から数百ms で確定テキスト。segment wait が不要
- **中間結果** — 喋っている途中のテキストが見える（表示用途 / tail_speech 不要化）
- **ファイル I/O 削減** — WAV セグメントファイルが不要

## 候補エンジン

### A. faster-whisper + VAD パイプ方式（ローカル）

ffmpeg の出力をパイプで受け、VAD で発話区間を切り出して faster-whisper に渡す。
segment muxer を使わず、メモリ上で処理を完結させる。

```
ffmpeg -i SOURCE -f s16le -ar 16000 -ac 1 pipe:1
  → Python (stdin読み取り)
  → Silero VAD でリングバッファから発話区間を切り出し
  → faster-whisper.transcribe(numpy_array)
  → バッファ書き込み
```

| 項目 | 評価 |
|------|------|
| プライバシー | 完全ローカル |
| 日本語精度 | 高（small 以上） |
| レイテンシ | 発話終了後 0.5〜1秒 |
| 中間結果 | なし（バッチ推論のため） |
| 実装コスト | 中（VAD + リングバッファの実装が必要） |
| 依存 | faster-whisper, silero-vad (PyTorch) |

**メリット:** 現在の faster-whisper 資産をそのまま使える。segment muxer → パイプの差し替えのみ。
**デメリット:** 真のストリーミング（中間結果）はできない。VAD 発話終了を待ってバッチ推論。

### B. whisper-streaming / whisper_online（ローカル）

faster-whisper をバックエンドにしたストリーミングラッパー。
チャンクを逐次投入し、バッファリング+再推論で擬似的に中間結果を生成。

```
ffmpeg pipe → whisper_online.OnlineASRProcessor
  → 中間結果（未確定テキスト）
  → 確定結果（VAD 沈黙検出後）
```

| 項目 | 評価 |
|------|------|
| プライバシー | 完全ローカル |
| 日本語精度 | 高（faster-whisper ベース） |
| レイテンシ | 中間: リアルタイム、確定: 発話終了後 0.5〜1秒 |
| 中間結果 | あり（擬似ストリーミング） |
| 実装コスト | 低（ライブラリが VAD + バッファリングを内包） |
| 依存 | whisper-streaming, faster-whisper |

**メリット:** 中間結果が得られる。VAD 内蔵。
**デメリット:** 再推論のため CPU 負荷がやや高い。ライブラリの成熟度。

### C. ElevenLabs Scribe API（クラウド）

WebSocket でリアルタイム STT。VAD 内蔵。

```
ffmpeg pipe → WebSocket → ElevenLabs Scribe
  → 中間結果 (partial transcript)
  → 確定結果 (final transcript)
```

| 項目 | 評価 |
|------|------|
| プライバシー | 音声をクラウド送信 |
| 日本語精度 | 高 |
| レイテンシ | 中間: 数百ms、確定: 発話終了後 1秒未満 |
| 中間結果 | あり（ネイティブ） |
| 実装コスト | 低（API 呼ぶだけ） |
| 依存 | ElevenLabs API キー、ネットワーク |

**メリット:** 真のストリーミング。VAD・ノイズ除去がサーバー側。実装が最も簡単。
**デメリット:** API コスト。プライバシー。ネットワーク依存。

### D. Google Cloud Speech-to-Text Streaming（クラウド）

```
ffmpeg pipe → gRPC → Google STT
  → 中間結果 (interim_results)
  → 確定結果 (is_final=true)
```

| 項目 | 評価 |
|------|------|
| プライバシー | 音声をクラウド送信 |
| 日本語精度 | 非常に高 |
| レイテンシ | 中間: 数百ms、確定: 発話終了後 1秒未満 |
| 中間結果 | あり（ネイティブ） |
| 実装コスト | 中（gRPC セットアップ） |
| 依存 | Google Cloud アカウント、API キー |

## 推奨: 段階的移行

### Step 1: パイプ化（A 方式の前半）

segment muxer を ffmpeg パイプに差し替え。STT は faster-whisper のまま。

**変更点:**
- worker.py: ffmpeg → stdout パイプ、リングバッファで音声データ保持
- Silero VAD で発話区間検出（RMS エネルギー閾値を置き換え）
- 発話区間を numpy array で faster-whisper に渡す
- /tmp/hearing_segments/ ディレクトリが不要に

**得られるもの:**
- 発話単位の自然な区切り（tail_speech 不要化）
- WAV ファイル I/O 削減
- segment_seconds パラメータ廃止

**失うもの:**
- segment ファイルによるデバッグの容易さ（WAV を直接再生できた）

### Step 2: ストリーミング STT（B or C 方式）

Step 1 のパイプ基盤の上に、ストリーミング STT を載せる。

- ローカル重視 → B（whisper-streaming）
- 精度・レイテンシ重視 → C（ElevenLabs Scribe）
- 設定で切り替え可能にする（AudioSource と同じ思想）

```toml
[hearing]
stt_engine = "faster-whisper"     # "faster-whisper" | "whisper-streaming" | "elevenlabs-scribe"
```

## 影響範囲

| コンポーネント | 変更内容 |
|-------------|---------|
| worker.py | ffmpeg パイプ化、VAD 統合、リングバッファ |
| transcriber.py | numpy array 入力対応（現在はファイルパス） |
| config.py | segment_seconds 廃止、stt_engine 追加 |
| filters.py | 変更なし（テキストレベルのフィルタはそのまま） |
| buffer.py | 変更なし（JSONL 書き込みはそのまま） |
| stop-hook | tail_speech 判定が不要になる可能性（VAD が代替） |
| hearing-hook | 変更なし |

## 未解決の設計判断

- **Silero VAD の PyTorch 依存**: faster-whisper は CTranslate2 で PyTorch 不要だが、Silero VAD は PyTorch が必要。依存が増える
  - 対策: onnxruntime 版 Silero VAD を使う（torch 不要）
- **リングバッファのサイズ**: 何秒分保持するか。長すぎるとメモリ、短すぎると発話が切れる
  - 暫定: 30秒（安全弁と同じ）
- **デバッグ用のセグメント保存**: パイプ化後もオプションで WAV 保存する機能を残すか
