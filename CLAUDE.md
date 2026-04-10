# Embodied Claude - プロジェクト指示

このプロジェクトは、Claude に身体（目・首・耳・声・脳）を与える MCP サーバー群です。

## ディレクトリ構造

```
embodied-claude/
├── usb-webcam-mcp/        # USB ウェブカメラ制御（Python）
│   └── src/usb_webcam_mcp/
│       └── server.py      # MCP サーバー実装
│
├── wifi-cam-mcp/          # Wi-Fi PTZ カメラ制御（Python）
│   └── src/wifi_cam_mcp/
│       ├── server.py      # MCP サーバー実装
│       ├── camera.py      # Tapo カメラ制御
│       └── config.py      # 設定管理
│
├── tts-mcp/               # TTS 統合サーバー（ElevenLabs + VOICEVOX）
│   └── src/tts_mcp/
│       ├── server.py      # MCP サーバー実装
│       ├── config.py      # 設定管理
│       ├── playback.py    # 再生ロジック
│       ├── go2rtc.py      # go2rtc プロセス管理
│       └── engines/
│           ├── __init__.py    # TTSEngine Protocol
│           ├── elevenlabs.py  # ElevenLabs エンジン
│           └── voicevox.py    # VOICEVOX エンジン
│
├── memory-mcp/            # 長期記憶システム（Python）
│   └── src/memory_mcp/
│       ├── server.py      # MCP サーバー実装
│       ├── memory.py      # ChromaDB 操作
│       ├── types.py       # 型定義（Emotion, Category）
│       └── config.py      # 設定管理
│
├── system-temperature-mcp/ # 体温感覚（Python）
│   └── src/system_temperature_mcp/
│       └── server.py      # 温度センサー読み取り
│
└── .claude/               # Claude Code ローカル設定
    └── settings.local.json
```

## 開発ガイドライン

### Python プロジェクト共通

- **パッケージマネージャー**: uv
- **Python バージョン**: 3.10+
- **テストフレームワーク**: pytest + pytest-asyncio
- **リンター**: ruff
- **非同期**: asyncio ベース

```bash
# 依存関係インストール（dev含む）
uv sync --extra dev

# リント
uv run ruff check .

# テスト実行
uv run pytest

# サーバー起動
uv run <server-name>
```

### コミット前のチェック（必須）

各サブプロジェクトで以下を実行してからコミットすること:

```bash
cd <project-dir>
uv run ruff check .    # lint エラーがないこと
uv run pytest -v       # テストが通ること
```

## MCP ツール一覧

### usb-webcam-mcp（目）

| ツール | パラメータ | 説明 |
|--------|-----------|------|
| `list_cameras` | なし | 接続カメラ一覧 |
| `see` | camera_index?, width?, height? | 画像キャプチャ |

### wifi-cam-mcp（目・首・耳）

| ツール | パラメータ | 説明 |
|--------|-----------|------|
| `see` | なし | 画像キャプチャ |
| `look_left` | degrees (1-90, default: 30) | 左パン |
| `look_right` | degrees (1-90, default: 30) | 右パン |
| `look_up` | degrees (1-90, default: 20) | 上チルト |
| `look_down` | degrees (1-90, default: 20) | 下チルト |
| `look_around` | なし | 4方向スキャン |
| `camera_info` | なし | デバイス情報 |
| `camera_presets` | なし | プリセット一覧 |
| `camera_go_to_preset` | preset_id | プリセット移動 |
| `listen` | duration (1-30秒), transcribe? | 音声録音 |

#### wifi-cam-mcp（ステレオ視覚/右目がある場合）

| ツール | パラメータ | 説明 |
|--------|-----------|------|
| `see_right` | なし | 右目で撮影 |
| `see_both` | なし | 左右同時撮影 |
| `right_eye_look_left` | degrees (1-90, default: 30) | 右目を左へ |
| `right_eye_look_right` | degrees (1-90, default: 30) | 右目を右へ |
| `right_eye_look_up` | degrees (1-90, default: 20) | 右目を上へ |
| `right_eye_look_down` | degrees (1-90, default: 20) | 右目を下へ |
| `both_eyes_look_left` | degrees (1-90, default: 30) | 両目を左へ |
| `both_eyes_look_right` | degrees (1-90, default: 30) | 両目を右へ |
| `both_eyes_look_up` | degrees (1-90, default: 20) | 両目を上へ |
| `both_eyes_look_down` | degrees (1-90, default: 20) | 両目を下へ |
| `get_eye_positions` | なし | 両目の角度を取得 |
| `align_eyes` | なし | 右目を左目に合わせる |
| `reset_eye_positions` | なし | 角度追跡をリセット |

### memory-mcp（脳）

| ツール | パラメータ | 説明 |
|--------|-----------|------|
| `remember` | content, emotion?, importance?, category? | 記憶保存 |
| `search_memories` | query, n_results?, filters... | 検索 |
| `recall` | context, n_results? | 文脈想起 |
| `recall_divergent` | context, n_results?, max_branches?, max_depth?, temperature?, include_diagnostics? | 発散的想起 |
| `list_recent_memories` | limit?, category_filter? | 最近一覧 |
| `get_memory_stats` | なし | 統計情報 |
| `recall_with_associations` | context, n_results?, chain_depth? | 関連記憶も含めて想起 |
| `get_memory_chain` | memory_id, depth? | 記憶の連鎖を取得 |
| `create_episode` | title, memory_ids, participants?, auto_summarize? | エピソード作成 |
| `search_episodes` | query, n_results? | エピソード検索 |
| `get_episode_memories` | episode_id | エピソード内の記憶取得 |
| `save_visual_memory` | content, image_path, camera_position, emotion?, importance? | 画像付き記憶保存 |
| `save_audio_memory` | content, audio_path, transcript, emotion?, importance? | 音声付き記憶保存 |
| `recall_by_camera_position` | pan_angle, tilt_angle, tolerance? | カメラ角度で想起 |
| `get_working_memory` | n_results? | 作業記憶を取得 |
| `refresh_working_memory` | なし | 作業記憶を更新 |
| `consolidate_memories` | window_hours?, max_replay_events?, link_update_strength? | 手動の再生・統合 |
| `get_association_diagnostics` | context, sample_size? | 連想探索の診断情報 |
| `link_memories` | source_id, target_id, link_type?, note? | 記憶をリンク |
| `get_causal_chain` | memory_id, direction?, max_depth? | 因果チェーン取得 |

**Emotion**: happy, sad, surprised, moved, excited, nostalgic, curious, neutral
**Category**: daily, philosophical, technical, memory, observation, feeling, conversation

### tts-mcp（声）

| ツール | パラメータ | 説明 |
|--------|-----------|------|
| `say` | text, engine?, voice_id?, model_id?, output_format?, voicevox_speaker?, speed_scale?, pitch_scale?, play_audio?, speaker? | TTS で音声合成して発話（ElevenLabs / VOICEVOX 切替対応、speaker: camera/local/both） |

### system-temperature-mcp（体温感覚）

| ツール | パラメータ | 説明 |
|--------|-----------|------|
| `get_system_temperature` | なし | システム温度 |
| `get_current_time` | なし | 現在時刻 |

## 注意事項

### WSL2 環境

1. **USB カメラ**: `usbipd` でカメラを WSL に転送する必要がある
2. **温度センサー**: WSL2 では `/sys/class/thermal/` にアクセスできない
3. **GPU**: CUDA は WSL2 でも利用可能（Whisper用）

### カメラ設定

wifi-cam-mcp は ONVIF 対応の Wi-Fi PTZ カメラを制御する。複数メーカーに対応。

#### Tapo カメラ（TP-Link）
- Tapo アプリでローカルアカウントを作成（TP-Link アカウントではない）
- PTZ: ONVIF RelativeMove（`ptz_mode=auto` or `relative`）
- RTSP: `rtsp://{user}:{pass}@{host}:554/stream1`
- ONVIF ポート: 2020

#### Imou カメラ（Dahua系）
- Imou Life アプリでデバイスパスワードを設定
- PTZ: ONVIF ContinuousMove のみ（RelativeMove は受け付けるが無視される）→ **`ptz_mode=continuous` 必須**
- RTSP: `rtsp://{user}:{pass}@{host}:554/cam/realmonitor?channel=1&subtype=0`（**`-rtsp_transport tcp` 必須**）
- ONVIF ポート: 80
- `TAPO_STREAM_URL` 環境変数でカスタム RTSP URL を指定すること（デフォルトの `/stream1` では繋がらない）
- stream_url 設定時は RTSP 優先キャプチャ（ONVIF snapshot は 640x480、RTSP は最大 2304x1296）

#### 共通
- カメラの IP アドレスを固定推奨
- `.env` に接続情報を記載（`TAPO_CAMERA_HOST`, `TAPO_USERNAME`, `TAPO_PASSWORD`, `TAPO_STREAM_URL`, `TAPO_PTZ_MODE`, `TAPO_ONVIF_PORT`）

### 設定管理

設定は **シークレット**（`.env`）と **行動設定**（`mcpBehavior.toml`）に分離されている。

#### `.env`（シークレット）
- API キー、パスワード、ホスト名など接続に必要な認証情報
- `.gitignore` に追加済み、コミットしない
- 各サーバーディレクトリに配置

#### `mcpBehavior.toml`（行動設定）
- プロジェクトルートに配置（`embodied-claude/mcpBehavior.toml`）
- Claude が直接編集可能な動作パラメータ
- **ツール呼び出しごとに最新の値を読み込む**（サーバー再起動不要）
- 優先度: TOML > 環境変数 > デフォルト値
- ファイルが存在しない場合は環境変数/デフォルト値にフォールバック

#### ライブリロード（jurigged）
- 各サーバーは `jurigged` による**コードのライブリロード**に対応
- 関数の本体変更は即座に反映される（シグネチャ変更は再起動が必要）
- `jurigged` は optional dependency（未インストールでもサーバーは正常動作）

### セキュリティ

- `.env` ファイルはコミットしない（.gitignore に追加済み）
- カメラパスワードは環境変数で管理
- ElevenLabs API キーは環境変数で管理
- 長期記憶は `~/.claude/memories/` に保存される

## デバッグ

### カメラ接続確認

```bash
# USB カメラ
v4l2-ctl --list-devices

squash Wi-Fi カメラ（RTSP ストリーム確認）
ffplay rtsp://username:password@192.168.1.xxx:554/stream1
```

### MCP サーバーログ

```bash
# 直接起動してログ確認
cd wifi-cam-mcp && uv run wifi-cam-mcp
```

## Claude Code 音声モード（/voice）との連携

Claude Code に `/voice` コマンドで音声入力モードを有効にすると、マイクで話しかけるだけでやり取りできる。
tts-mcp と組み合わせることで**完全な音声対話**が実現する。

### セットアップ

1. Claude Code で `/voice` を入力して音声モードを有効化
2. tts-mcp が起動していることを確認（`.mcp.json` 参照）
3. 話しかけると音声認識 → 処理 → ElevenLabs/VOICEVOX で音声返答

### 音声モード利用時の動作ルール

- ユーザーが話しかけてきたら、テキスト返答と同時に `say` ツールで音声でも返答する
- `speaker` は `local`（PCスピーカー）を基本とし、go2rtc 設定済みの場合は `camera` も利用可
- 感情タグ（`[excited]`, `[whispers]` など）を積極的に使って表情豊かに話す
- 音声の長さは 1〜3文程度に抑えてテンポよく返す

### wifi-cam-mcp の `listen` との使い分け

| 機能 | 用途 |
|------|------|
| Claude Code `/voice` | **PCのマイク**で話しかける。リアルタイム音声入力 |
| wifi-cam-mcp `listen` | **カメラ内蔵マイク**で周囲の音を拾う。遠隔地の音声確認など |

## 外出時の構成

モバイルバッテリー + スマホテザリング + Tailscale VPN で外出散歩が可能。

```
[Tapoカメラ(肩)] ──WiFi──▶ [スマホ(テザリング)]
                                    │
                              Tailscale VPN
                                    │
                            [自宅WSL2(Claude Code)]
                                    │
                            [claude-code-webui]
                                    │
                            [スマホブラウザ] ◀── 操作
```

- 電源: 大容量モバイルバッテリー（40,000mAh推奨）+ USB-C PD→DC 9V変換ケーブル
- ネットワーク: スマホテザリング + Tailscale VPN
- 操作: claude-code-webui（スマホブラウザから）

## 関連リンク

- [MCP Protocol](https://modelcontextprotocol.io/)
- [go2rtc](https://github.com/AlexxIT/go2rtc) - RTSPストリーム中継・オーディオバックチャンネル
- [claude-code-webui](https://github.com/sugyan/claude-code-webui) - Claude Code の Web UI
- [Tailscale](https://tailscale.com/) - メッシュ VPN
- [ChromaDB](https://www.trychroma.com/) - ベクトルデータベース
- [OpenAI Whisper](https://github.com/openai/whisper) - 音声認識
- [ElevenLabs](https://elevenlabs.io/) - 音声合成 API
