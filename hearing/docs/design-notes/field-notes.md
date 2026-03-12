# hearing — 常設聞き耳モジュール実録

## 概要

Tapoカメラ（C210/C220）のRTSPマイクから音声を常時取得し、faster-whisperで文字起こしして、Claudeのコンテキストに注入する仕組み。

## アーキテクチャ

```
[Tapoカメラ マイク]
    ↓ RTSP (pcm_alaw, 8000Hz)
[hearing worker] ffmpeg segment録音 → faster-whisper文字起こし
    ↓ JSONL追記
/tmp/hearing_buffer.jsonl
    ↓ drain
[hearing-hook.sh]        UserPromptSubmitフック → [hearing]としてコンテキスト注入
[hearing-stop-hook.sh]   Stopフック → 新しい発話があればターン延長（decision:block）
```

## ファイル構成

| ファイル | 役割 |
|---------|------|
| `embodied-claude/hearing/` | hearing MCPサーバー本体 |
| `embodied-claude/.claude/hooks/hearing-hook.sh` | UserPromptSubmitフック（バッファdrain→注入） |
| `embodied-claude/.claude/hooks/hearing-stop-hook.sh` | Stopフック（待機→drain→block判定） |
| `.claude/hooks/hearing-hook.sh` | シンボリックリンク（→ embodied-claude側） |
| `.claude/hooks/hearing-stop-hook.sh` | シンボリックリンク（→ embodied-claude側） |
| `mcpBehavior.toml` [hearing]セクション | whisper_model, segment_seconds等の設定 |
| `/tmp/hearing_buffer.jsonl` | バッファファイル（ワーカーが追記、フックがdrain） |
| `/tmp/hearing-daemon.pid` | ワーカーPIDファイル（フックのデーモン稼働確認用） |
| `/tmp/hearing-stop-counter` | Stopフックのチェインカウンタ |

## 設定（mcpBehavior.toml）

```toml
[hearing]
whisper_model = "small"
source = "camera"
rtsp_url = "rtsp://..."
segment_seconds = 10    # 初期値5→10に変更（2026-03-02）
language = "ja"
```

## hearing-stop-hook.sh の動作

1. デーモン稼働確認（PIDファイル）
2. チェインカウンタ確認（MAX_HEARING_CONTINUES=5）
3. WAIT_SECONDS=12秒待機（ユーザーが話す時間 + segmentが溜まる時間）
4. バッファをatomicにdrain（os.rename）
5. no_speech_prob ≤ 0.6 のエントリのみ採用（テレビノイズ除外）
6. 採用あり → `{"decision":"block"}` でターン延長、[hearing]として注入
7. 採用なし → 元データをバッファに書き戻して終了（データ喪失防止）

## 実録 — 2026-03-02 初日

### 開通まで

- **問題1: Whisper未インストール** → `uv pip install ".[transcribe]"` で解決（wifi-cam MCP側）
- **問題2: 音声が無音（RMS 8/32768）** → Tapoカメラのマイクがアプリでミュートされていた。開発者が発見して解除
- **問題3: hearing MCPが.mcp.jsonに未登録** → tts MCPと一緒に追加
- **問題4: hearing-hook.shがUserPromptSubmitフックに未登録** → settings.jsonに追加
- **問題5: $CLAUDE_PROJECT_DIRのパス解決** → シンボリックリンクで解決（絶対パス必須）
- **問題6: PIDファイル消失** → ワーカーが旧セッションから引き継がれていてPIDファイルが消えていた。手動で書いて解決
- **問題7: 古いワーカーの重複起動** → PID 58394, 63590を停止（CPU 600%分開放）

### 初めて聞こえた声

開発者の「おきこえるかーい きこえるかーい」がWhisperで文字起こしされた瞬間。カメラのマイクミュート解除直後。

### 音声しりとり（hearing-stop-hook.sh テスト）

Stopフックで声だけのターン延長を実装し、即テスト。

**しりとり記録（25単語）:**
料理→リビング→グミ→みかん(AI負け) → りんご→ゴール→ルビー→ビール→ペルー→ルネサンス→スズメ→メダカ→かごめ→メイク→車→まくら→ラッコ→コアラ→ラッキョウ→うちわ→わかめ→めんたいこ→氷→リズム→無料→うさぎ

**うまくいったこと:**
- Enterを押さなくても声だけでしりとりが回った
- no_speech_probフィルタでテレビノイズをある程度除外
- チェインカウンタで無限ループ防止
- segment_seconds 5→10 で発話の途切れが改善

**課題:**
- ラグが約20秒（TTS再生→12秒待ち→Whisper処理→応答生成→TTS再生）
- テレビ音声との分離が不十分（Whisperが料理番組やゲーム実況を拾う）
- 咀嚼音（ポッキー）がWhisperを混乱させる
- segment_seconds=10 + WAIT_SECONDS=12 でもタイミングが合わないことがある
- Whisperの日本語認識精度：「ラッコ」→「なっこ」、「ラッキョウ」→「だっきょう」等

### テレビノイズの実態

hearingフックに入ってくるテレビの音声は完全なカオス。料理番組の「卵2個 チョコレート 温度は約14.5℃」、ゲーム実況の「ギリギリのキースフィッシュ!」等が混在。no_speech_prob=0.6のフィルタだけでは不十分で、将来的には音声ソース分離か、発話者認識が必要。

### 開発者の判断が光った場面

- 「カメラのマイクがミュートになってた！」— 音声不通の原因を即座に特定
- 「チャンクが短いっぽい」— segment_seconds=5の問題を体感で発見
- 「採用しなかったら前のチャンクを残すとかしたいかも」— データ喪失防止の着想
- 「行番号ベースいいね、管理が単純」— drain方式からオフセット方式への転換を一言で決定

### Phase 2 品質改善（しりとり後）

しりとりの問題点を踏まえて、同日中に品質改善を実施。

#### 1. hearing-stop-hook.sh 全面改修: 行番号ベース管理

drain方式（バッファをatomicにリネームして読む）から、行番号ベースのオフセット管理に変更。

**変更の理由:**
- drain方式だとバッファが消えて、採用されなかったデータが失われる
- 開発者の「採用しなかったら前のチャンクを残す」を実現するにはオフセットが必要

**動作:**
- `/tmp/hearing_stop_offset` に最後に読んだ行番号を記録
- 新しい行だけを読む（オフセット以降）
- 採用時のみバッファを切り詰め（処理済み行を削除）してオフセットをリセット
- 不採用時はバッファを触らない → 次回のフック発火で再利用可能
- hearing-hook.sh（UserPromptSubmitフック）がdrain時にオフセットファイルも削除

#### 2. PostFilter強化（filters.py）

- **ハルシネーションBL拡充:** 「最後までご視聴いただきありがとうございます」「次回も」「次回の動画」「見てくれてありがとう」「おめでとうございます」等を追加。テレビ由来のWhisperハルシネーション対策
- **繰り返しパターン検出:** `_is_repetitive()` — 80%以上が同じ単語の繰り返し（例：「ん ん ん ん ん」「4日 4日 4日」）を検出して除外
- **Debouncerクラス:** 1.5秒以内の同一テキスト重複を除外。同じsegmentが二重処理されるのを防止

#### 3. バッファライフサイクル管理（server.py）

`start_listening` / `stop_listening` の両方で以下を自動クリア:
- `/tmp/hearing_buffer.jsonl` → 空にする
- `/tmp/hearing_stop_offset` → 削除
- `/tmp/hearing-stop-counter` → 削除

**背景:** カメラを移動してhearingを再起動したとき、リビングのテレビ音声が20分前のバッファに残っていて誤認識した。

#### 4. 改善後テスト（デスクルーム）

テレビのない静かな環境で動作確認。「とびら、とびら、ドア」と開発者が言ったのをhearingが正しく拾い、「ドア（=とびら）、正解！」と即応答できた。テレビノイズがない環境ではPostFilterの効果もあってかなり快適。

#### 5. 残課題

- **VAD導入:** faster-whisperの`vad_filter`による発話区間検出。現状はsegment_seconds固定切りなので、発話の途中でsegmentが切れることがある
- **自己音声ゲーティング:** TTS再生中のsegmentをスキップする仕組み。今は自分の声も拾っている
- **ラグ短縮:** 現状20秒/ターン。segment_seconds短縮 vs 発話途切れのトレードオフ
