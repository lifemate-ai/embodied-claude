# Design Notes — 目次

hearing モジュールの設計時に書かれたドキュメント群。
実装前・実装中の検討メモであり、現在の実装とは差異がある箇所も多い。

## 計画済みの課題

設計書が起こされており、実装に着手できる状態。

| 課題 | 設計書 | 概要 |
|------|--------|------|
| Segment muxer → リアルタイム STT | [realtime-stt-migration.md](./realtime-stt-migration.md) | ffmpeg パイプ化 + Silero VAD + ストリーミング STT への段階的移行 |
| イベントドリブン発話トリガー | [event-driven-hearing.md](./event-driven-hearing.md) | 発話検出で tmux send-keys → 会話自動開始 |

## 要検討の課題

設計ドキュメントで言及されているが、具体的な計画は未策定。

| 課題 | 検討箇所 | 状況 |
|------|---------|------|
| 自己音声ゲーティング | [implementation.md](./implementation.md) §5 | 設計あり、未実装。TTS 再生中のセグメントスキップ |
| Silero VAD への移行 | [architecture.md](./architecture.md) §VAD | 現状 RMS で運用中。精度向上には Silero が有力。realtime-stt-migration の Step 1 に包含 |
| 重要度ベースの Delivery 分岐 | [architecture.md](./architecture.md) §Delivery 推奨 | 名前呼びかけ等に即時反応する仕組み。event-driven-hearing の Phase 2〜3 に包含 |
| ラグ短縮 | [field-notes.md](./field-notes.md) §残課題 | segment 3 秒 + wait 5 秒でかなり改善。根本解決は realtime-stt-migration で |
| WebRTC 音声ソース | [architecture.md](./architecture.md) §AudioSource | mcp-pet 経由でブラウザマイクを利用する構想 |
| 話者分離 | [implementation.md](./implementation.md) §Phase4 | テレビ等との音声分離。未着手 |
| 音声イベント検出 | [implementation.md](./implementation.md) §Phase4 | ドアベル、犬の鳴き声等。未着手 |

## 解決済みの課題

設計ドキュメントで検討され、現在の実装で解決済み。

| 課題 | 設計時の検討 | 現在の実装 |
|------|------------|-----------|
| 連続録音の音声欠損 | ffmpeg segment muxer vs パイプ vs ダブルバッファ ([implementation.md](./implementation.md) §1) | ffmpeg segment muxer 採用 |
| Whisper ハルシネーション | ブラックリスト + フィラー除外 ([implementation.md](./implementation.md) §4) | 3層フィルタ (デフォルト + モデル別 + toml) + LLM フィルタ (opt-in) |
| バッファ管理方式 | drain (atomic rename) → 行番号ベース ([field-notes.md](./field-notes.md) §Phase2) | 行番号オフセット + truncate |
| Delivery 方式 | hook / tmux / claude -p / claude -r の比較 ([architecture.md](./architecture.md) §Delivery) | UserPromptSubmit hook + Stop hook のチェーン |
| VAD | Silero VAD / ブラウザ / サーバー側 ([architecture.md](./architecture.md) §VAD) | RMS エネルギー閾値 (軽量版) |
| 発話途中の切断 | segment 境界問題 ([field-notes.md](./field-notes.md) §残課題) | tail_speech 検出 (末尾 0.5 秒の RMS) |
| チェーン持続性 | チェーン上限のみ ([field-notes.md](./field-notes.md)) | 保証カウンター分離 (HIT でリセット) |
| STT エンジン選定 | Whisper / ElevenLabs / Web Speech API ([architecture.md](./architecture.md) §STT) | faster-whisper (small, CPU, int8) |
| 重複デバウンス | familiar-ai 参考 ([reference-comparison.md](./reference-comparison.md)) | Debouncer (1.5 秒) 実装済み |

## 参考資料

| ファイル | 内容 |
|---------|------|
| [reference-comparison.md](./reference-comparison.md) | AiTuberKit / familiar-ai / PersonaPlex との比較。設計判断の参考資料。取り込み済みの知見多数 |

## ファイル一覧

| ファイル | 概要 |
|---------|------|
| [architecture.md](./architecture.md) | 抽象パイプライン設計、AudioSource / VAD / STT / Delivery の選択肢比較 |
| [implementation.md](./implementation.md) | 実装設計、コード例、フェーズ計画 |
| [field-notes.md](./field-notes.md) | 2026-03-02 初日の実録、しりとりテスト、Phase 2 品質改善 |
| [reference-comparison.md](./reference-comparison.md) | AiTuberKit / familiar-ai / PersonaPlex との比較分析 |
| [event-driven-hearing.md](./event-driven-hearing.md) | イベントドリブン発話トリガーの設計書（tmux send-keys による会話自動開始） |
| [realtime-stt-migration.md](./realtime-stt-migration.md) | Segment muxer → リアルタイム STT への移行計画 |
