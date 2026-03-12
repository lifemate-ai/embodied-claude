# 聞き耳モジュール — 参考実装比較

> 関連: [README.md](./README.md) | [architecture.md](./architecture.md) | [implementation.md](./implementation.md)
>
> 参照コード: `references/aituber-kit/`, `references/familiar-ai/`

## 3プロジェクト構造比較

| 項目 | **embodied-claude** | **AiTuberKit** | **familiar-ai** |
|------|---|---|---|
| 実行環境 | macOS / ローカルデーモン | ブラウザ (Next.js) | Python CLI (Textual TUI) |
| 音源 | RTSP カメラマイク | PC マイク (getUserMedia) | PC マイク (sounddevice) |
| 常時リスニング | デーモン + hook 注入 | `continuousMicListeningMode` トグル | `REALTIME_STT=true` |
| STT | ローカル Whisper | Web Speech API / Whisper API / Realtime API | ElevenLabs Scribe API |
| VAD | ローカル Silero VAD | ブラウザ `onspeechstart` + 沈黙タイマー | サーバー側 (ElevenLabs) |
| 自己音声除外 | 時間ゲーティング + テキストマッチ | `echoCancellation: true` + 発話中マイク停止 | なし |
| チャンク方式 | ffmpeg segment (5秒) | MediaRecorder (100ms) | sounddevice (100ms) |
| API 依存 | なし（完全ローカル） | OpenAI（Whisper/Realtime 使用時） | ElevenLabs（必須） |

---

## AiTuberKit

### なぜ「スマートに見える」か

AiTuberKit の設計のシンプルさは、**ブラウザ API に全面的に依拠**していることに起因する:

| embodied-claude の課題 | AiTuberKit の解決方法 |
|---|---|
| 音声欠損（ffmpeg 停止中） | 存在しない（ストリーミング） |
| Whisper ハルシネーション | 該当なし（Web Speech API が内部処理） |
| 自己音声除外 | `echoCancellation: true`（1行） |
| VAD | `onspeechstart` / `onspeechend` イベント |
| チャンク分割 | MediaRecorder が自動管理 |

これらの恩恵は **ブラウザ環境でしか得られない**。RTSP カメラマイクでは getUserMedia が使えないため、同じアプローチは直接適用できない。ただし mcp-pet の WebRTC 経由でブラウザマイクを取る場合は、同等の恩恵を受けられる。

### 取り込むべき知見

| 知見 | 適用方法 |
|------|---|
| 2段階 VAD（初期音声検出 + 沈黙検出） | デーモンに初期音声検出タイムアウトを追加 |
| ノイズ判定（有意な変化のみカウント） | Whisper 出力の文字数変化を追跡、差分が小さければスキップ |
| 自動再開パターン | TTS 完了通知 → ゲーティング解除 + 300ms マージン |
| 排他制御（exclusionMiddleware） | Realtime API 有効時に他モードを無効化するパターン |

### 主要コードパス

- 常時リスニング設定: `src/hooks/useVoiceRecognition.ts` L24-26
- VAD（沈黙検出）: `src/hooks/useSilenceDetection.ts` L58-185
- Web Speech API: `src/hooks/useBrowserSpeechRecognition.ts` L377-447
- エコーキャンセル: `src/hooks/useAudioProcessing.ts` L79-85
- 自動再開: `src/hooks/useVoiceRecognition.ts` L95-107

---

## familiar-ai

### 設計判断

**外部 API に全面委託**する設計。VAD・STT を ElevenLabs に一任し、クライアント側はフィルタリングに集中。

| 層 | familiar-ai のアプローチ |
|---|---|
| AudioSource | sounddevice (100ms, 16kHz, モノラル) |
| VAD | ElevenLabs サーバー側 (`commit_strategy=vad`, 1秒沈黙で確定) |
| STT | ElevenLabs Scribe (WebSocket realtime + REST batch) |
| PostFilter | フィラー除外 + 重複デバウンス (1.2秒) + 記号除外 |

### 取り込むべき知見

| 知見 | 適用方法 |
|------|---|
| `should_skip_stt()` 三段フィルタ | 1文字・記号のみ・フィラーの3条件でスキップ |
| 重複デバウンス (1.2秒) | 同一テキスト連続を除外 |
| リサンプリング（numpy 線形補間） | 非 16kHz デバイスからの正規化 |
| `call_soon_threadsafe` パターン | sounddevice スレッド → asyncio イベントループの安全な移譲 |

### 弱点

- **エコーキャンセレーションが未実装**: 物理的分離を仮定しているが、スピーカーとマイクが近い環境では問題になる
- **API 依存**: ElevenLabs の可用性・コストに縛られる

### 主要コードパス

- マイクキャプチャ: `src/familiar_agent/tools/mic.py` L47-82
- Realtime STT: `src/familiar_agent/tools/realtime_stt.py` L21-72
- フィルタリング: `src/familiar_agent/realtime_stt_session.py` L32-49
- デバウンス: `src/familiar_agent/realtime_stt_session.py` L126-141

---

## PersonaPlex（NVIDIA）

Moshi ベースの全二重音声対話システム。エンドツーエンドモデル（音声→音声）なので本プロジェクトとはアーキテクチャが根本的に異なるが、以下の知見が参考になる:

| 知見 | 適用 |
|------|------|
| **適応的バッファ管理** | チャンクサイズの動的調整（Whisper の処理速度に応じて）。Phase 1 では固定で十分 |
| **遅延メトリクスの分離追跡** | 録音→推論→注入の各段階で遅延を計測。品質改善の判断材料 |
| **デコーダーのプリウォーム** | Whisper モデルの起動時ダミー推論で初回遅延を 30-50% 削減 |
| **入出力の完全分離** | 聞く（デーモン）と喋る（TTS）を独立パイプラインとして扱い、薄いシグナリングで同期 |

---

## 核心的な結論

3プロジェクトの設計を抽象化すると**同一のパイプライン**に帰着する:

```
[AudioSource] → [VAD] → [STT] → [PostFilter] → [Delivery]
```

設計の複雑さの違いは、各レイヤーの具体的な実装選択から生じる。特に AudioSource の制約（RTSP vs ブラウザ vs ローカルマイク）が複雑さの最大の決定要因であり、mcp-pet の WebRTC 音声対応によって embodied-claude の制約を AiTuberKit 相当に緩和できる。
