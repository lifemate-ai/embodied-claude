# Known Issues & Future Improvements

## 一時ファイル管理

### `/tmp/` 固定パスの問題
- すべての一時ファイルが `/tmp/` にハードコードされている
- 複数インスタンスの同時実行で競合する
- デバッグ時に読み出し権限がないことがある（別ユーザーが作成した場合）
- **改善案**: `$XDG_RUNTIME_DIR` や設定可能なベースディレクトリに変更

### 一時ファイルの掃除ロジックがない
- `stop_listening` は `hearing_buffer.jsonl`, `hearing_stop_offset`, `hearing-stop-counter` のみクリア
- タイミングログ (`hearing_timing.log`) やタイムスタンプファイルは残り続ける
- `hearing_segments/` 内の古い WAV ファイルも蓄積する
- **改善案**: `stop_listening` 時に全一時ファイルを掃除する / 起動時に古いファイルを削除

## Hook

### settings.json への手動登録が必要
- hearing-hook.sh と hearing-stop-hook.sh を `.claude/settings.json` に手動で追加する必要がある
- 登録忘れると hearing MCP は動くがフック注入されず、無言になる

### stop-hook の Bash + Python 2層構造
- hearing-stop-hook.sh に埋め込み Python heredoc がある
- デバッグしづらい、テストしづらい
- **改善案**: Python 部分を独立スクリプトに切り出す

## 転写精度

### Whisper のハルシネーション
- 無音〜小音量でも「ご視聴ありがとうございました」等の定型句を生成する
- ルールベースフィルタ + LLM フィルタで対処しているが、完全ではない
- 短い発話（1〜2語）の認識精度が低い

### カメラマイクの音質
- RTSP 経由のカメラマイクは音質が悪く、誤認識が多い
- 「今度」→「コンブ」、「実装してみよう」→「ディスウォース」等
- **改善案**: PC マイク (`source = "local"`) の方が精度は高い
