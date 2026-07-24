# Embodied Claude - プロジェクト指示

このプロジェクトは、Claude に身体（目・首・耳・声・脳）を与え、その上に sociality
（社会的中間層）を積む MCP サーバー群です。

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
├── social-core/           # sociality 共通DB・モデル（Python）
│   └── src/social_core/
│       ├── db.py          # SQLite + migration
│       ├── events.py      # append-only event store
│       └── models.py      # 共通 schema
│
├── sociality-mcp/         # sociality 統合 façade（公開 MCP）
├── social-state-mcp/      # 現在の社会的状態推定（Python）
├── relationship-mcp/      # 関係性の圧縮表現と約束管理（Python）
├── joint-attention-mcp/   # 共同注意と参照解決（Python）
├── boundary-mcp/          # 同意・静寂・プライバシーの行動ゲート（Python）
├── self-narrative-mcp/    # daybook と narrative arc（Python）
│
├── socialPolicy.toml      # boundary-mcp の既定ポリシー
│
└── .claude/               # Claude Code ローカル設定
    └── settings.local.json
```

## 開発ガイドライン

### Python プロジェクト共通

- **パッケージマネージャー**: uv
- **Python バージョン**: 3.13（root `.python-version` で固定）
- **workspace**: 全 Python package が root `.venv` と `uv.lock` を共有
- **テストフレームワーク**: pytest + pytest-asyncio
- **リンター**: ruff
- **非同期**: asyncio ベース

```bash
# 全 package と dev dependency を一括 install
uv sync

# リント
uv run ruff check .

# package ごとのテスト実行
uv run pytest <package-dir>/tests

# サーバー起動
uv run --package <package-name> <server-name>
```

### コミット前のチェック（必須）

root workspace から対象 package を指定して実行すること:

```bash
uv lock --check
uv run ruff check <project-dir>
uv run pytest <project-dir>/tests -v
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

### sociality-mcp（統合 façade）

公開用には `sociality-mcp` を使う。以下の social tool 群を 1 つの MCP サーバーから
公開し、内部実装は `social-state-mcp` / `relationship-mcp` / `joint-attention-mcp` /
`boundary-mcp` / `self-narrative-mcp` に分割して保守する。

### social-state tools（社会的状態）

| ツール | パラメータ | 説明 |
|--------|-----------|------|
| `ingest_social_event` | event | social event を append-only に保存 |
| `get_social_state` | window_seconds?, person_id?, include_evidence? | 在席、活動、割り込み可能性、会話フェーズを返す |
| `should_interrupt` | candidate_action, urgency?, person_id?, message_preview? | 話しかけるべきか判定 |
| `get_turn_taking_state` | person_id? | ターン保持/応答を返す |
| `summarize_social_context` | person_id?, max_chars? | 短い社会的要約 |

### relationship tools（関係性）

| ツール | パラメータ | 説明 |
|--------|-----------|------|
| `upsert_person` | person_id, canonical_name, aliases?, role? | 人物レコード更新 |
| `ingest_interaction` | person_id, channel, direction, text, ts | やり取りを要約ベースで保存 |
| `get_person_model` | person_id | 好み、約束、未解決ループ、境界を返す |
| `create_commitment` / `complete_commitment` | person_id..., commitment_id | 約束管理 |
| `list_open_loops` / `suggest_followup` | person_id..., context? | 継続性のある follow-up を返す |
| `record_boundary` | person_id, kind, rule, source_text | 人ごとの境界を保存 |

### joint-attention tools（共同注意）

| ツール | パラメータ | 説明 |
|--------|-----------|------|
| `ingest_scene_parse` | scene | 構造化 scene parse を保存 |
| `resolve_reference` | expression, person_id?, lookback_frames? | 指示語や属性参照を解決 |
| `get_current_joint_focus` / `set_joint_focus` | person_id?, target_id? | 共同注視対象を管理 |
| `compare_recent_scenes` | person_id?, window_minutes? | 最近の scene 差分を返す |

### boundary tools（行動ゲート）

| ツール | パラメータ | 説明 |
|--------|-----------|------|
| `evaluate_action` | action_type, channel?, person_id?, context?, payload_preview?, urgency? | 発話/投稿/促しを事前評価 |
| `review_social_post` | channel, text, scene_contains_face?, person_mentions? | 投稿のリスクレビュー |
| `record_consent` | person_id, consent_type, value, source | 同意を保存 |
| `get_quiet_mode_state` | ts | quiet mode 状態を返す |

### self-narrative tools（自己要約）

| ツール | パラメータ | 説明 |
|--------|-----------|------|
| `append_daybook` | day? | 日次の narrative 要約を更新（v0.3: concrete_events / noticed_changes / relationship_moments / next_gentle_actions 付き） |
| `get_self_summary` | なし | prompt 注入向け自己要約（v0.3: 最近の experiences と interpretation_shifts を含む） |
| `list_active_arcs` | なし | 進行中の narrative arc |
| `reflect_on_change` | horizon_days? | 最近の変化を要約 |

### interaction-orchestrator tools（v0.3 人間応答オーケストレーション）

応答前後のループ「notice → interpret → choose → act → remember」を MCP で明示化する
orchestration 層。従来の social-state / relationship / self-narrative / boundary を束ねて
1 回で prompt 準備できる。

| ツール | パラメータ | 説明 |
|--------|-----------|------|
| `compose_interaction_context_tool` | person_id?, channel?, user_text?, autonomous_trigger?, include_private?, max_chars? | 応答前に呼ぶ。social state / relationship / open loops / desire / 最近の experience / relevant_memories（memory.db から recall）/ response_contract を 1 つにまとめて返す |
| `plan_response_tool` | interaction_context, user_text?, candidate_goal? | compose の結果を受けて primary_move（answer_directly / stay_silent / write_private_reflection など）、tone、memory_use、initiative（allowed / forbidden actions）、voice、must_include / must_avoid、followup_action を決定 |
| `record_agent_experience` | payload | 応答・自律行動・境界遵守・欲求充足・interpretation_shift 等を experience として保存。次の compose で `recent_experiences` に surface される |
| `record_interpretation_shift` | payload | 規則/関係/自己モデルの解釈を更新した瞬間を記録。`agent_state.interpretation_shifts` で以降の plan が自動的に「regress せえへん」制約を must_include に載せる |
| `append_private_reflection` | payload | 誰にも nudge せんと private なメモを残す。深夜帯の autonomous tick で write_private_reflection が選ばれた時に使う |
| `compose_private_letter` | payload | 朝の手紙的な letter を保存（本文は Claude が書く）。後で共有するかは visibility で制御 |
| `get_agent_state` | person_id? | compose より軽量。欲求、最近の experience、active arcs のみ返す。introspection 用 |

### individual-kernel-mcp（脳の自己観察層）

`consciousness-mcp/packages/individual-kernel-mcp/`。counterfactual / tick frame /
attention schema / HOR に加え、EFPF の workspace competition、committed field、
intention、prediction/outcome loop を MCP と hooks に露出する。TTS、投稿、カメラ移動、
file/network side effect は committed field と matching intention がない限り
`PreToolUse` で拒否され、boundary policy と一 tick 一外向き行為の bottleneck も通る。

| ツール | パラメータ | 説明 |
|--------|-----------|------|
| `record_counterfactual` | payload | 拒否した代替行為を typed record で残す（source ∈ boundary_deny / attention_lost_bid / deliberate_choice / policy / ignition_failed）。evidence_type は Phase 1 EvidenceType を再利用 |
| `query_counterfactuals` | since?, source?, tick_id?, person_id?, limit? | 直近の counterfactual を返す |
| `sleep_consolidate` | force?, dry_run? | quiet hours で morning_briefing.json を書き出す scheduler glue |
| `record_tick_frame` | payload | per-tick の ConsciousFrame を保存。winning_memory_ids / dominant_desire / prediction_error / chosen_action_ref など FK だけ持つ |
| `get_tick_frame` | tick_id | tick_id から ConsciousFrame を取り出す |
| `query_tick_frames` | since?, reportability?, person_id?, ignited_only?, limit? | 直近の ConsciousFrame を返す |
| `record_attention_schema` | focal_target_ref, modality, intensity, dwell_seconds?, ... | AST スナップショットを in-memory ring buffer に積む（cap 60）|
| `update_attention_from_frame` | tick_id | ConsciousFrame から AttentionSchema を投影（modality / intensity / dwell を自動推論）|
| `flush_attention_schemas` | — | 在庫を SQLite に永続化。返り値 {count}。プロセス再起動で buffer は消えるので明示 flush 必要 |
| `summarize_attention_schema` | extra_history? | reflect_attention_schema を呼んで modality 分布 / 焦点変化数 / dominant focal を返す |
| `record_hor` | payload | Higher-Order Representation を保存。asserted_mode ∈ seeing/wanting/intending/remembering/feeling/attending。EpistemicClaim に projection 可能 |
| `get_hor` | hor_id | HORRecord を取り出す |
| `query_hors` | since?, owner_id?, asserted_mode?, target_kind?, source_tick_id?, limit? | HOR を絞り込んで返す |
| `compose_introspection_report` | window_hours?, owner_id? | 最近の HOR + attention reflection + counterfactual 件数を合わせた IntrospectionReport を作る。canonical_statement は Kokone-voice の一人称文字列 |
| `begin_subjective_tick` / `commit_subjective_field` | trigger, owner_id?, ... / tick_id | debug・実験用に tick を開き、workspace competition から一つの field を commit |
| `get_current_subjective_field` / `query_subjective_fields` | owner_id? / filters... | 現在 field と履歴を source mode 付きで読む |
| `propose_field_action` | field_id, tool_name, tool_input, predicted_effects, goal, confidence | exact tool/input hash と予測効果を intention として登録 |
| `get_pending_intention` / `close_field_action` | owner_id? / action_id, actual_result... | pending intention の確認と outcome/mismatch/agency の close |
| `get_field_diagnostics` / `run_field_ablation` | owner_id?, window? / kind, fixture?, seed? | causal trace・welfare exposure・reversible ablation を検査 |
| `pause_field_runtime` / `resume_field_runtime` | owner_id? | field runtime の停止・再開 |

## Heartbeat Protocol

自律行動や会話中に sociality を使うときは、最低限この順序を守ること。

### EFPF 推奨フロー（field → compose → plan → intend → act → outcome）

1. hook / heartbeat で `begin_tick` → competition → atomic field commit を完了する。
2. 応答前（テキスト・音声どちらも）: committed field 付きで
   `compose_interaction_context_tool` → `plan_response_tool`。
   plan の `primary_move` が `stay_silent` / `defer` なら黙って応答しない。`voice.speak=false`
   を勝手に覆さへん。`must_avoid` と `must_include` を必ず守る。
3. 外向き tool の直前に `propose_field_action` で exact input と予測を一つ登録する。
   `PreToolUse` の field / intention / hash / boundary / bottleneck 判定を迂回しない。
4. `PostToolUse` / `PostToolUseFailure` で actual outcome と prediction mismatch を閉じ、
   tool-result microtick が commit されてから次の外向き行為へ進む。
5. 応答を出した直後: `record_agent_experience`（kind 適宜）。promise があれば `create_commitment`。
   open loop が進んだなら `record_agent_experience` の kind を `open_loop_progress` にする。
6. 自分の解釈が変わった瞬間（ルール/関係/自己モデル）: `record_interpretation_shift`。以降の
   plan が「古い解釈に戻らない」制約を自動で挟む。
7. 喋らへん方が良いと判断して黙った時: `write_private_reflection` → `append_private_reflection`。
   朝の手紙的なもんは `compose_private_letter` で visibility=private で保存。

### レガシー直接コール（compose/plan 経由せん時や個別補完）

1. 話しかける前、say 前、軽く促す前: `get_social_state` → `evaluate_action`
2. X 投稿前: `get_social_state` → `get_person_model`（人が絡むなら）→ `review_social_post` → `evaluate_action`
3. 人の発話・要求・境界・約束を拾ったら: `ingest_social_event` と `ingest_interaction` を保存。境界なら `record_boundary`、約束なら `create_commitment`
4. 構造化できる scene が取れたら: `ingest_scene_parse`。指示語が曖昧なら `resolve_reference`
5. 毎日1回か節目で: `append_daybook` を呼んで自己要約を更新

### socialPolicy.toml

リポジトリ直下の `socialPolicy.toml` に timezone / quiet_hours / privacy_zones / posting_rules /
person_rules を書く。v0.3 以降 `[global] timezone = "Asia/Tokyo"` を必ず設定する
（未設定だと UTC で解釈され、JST 深夜帯の quiet-hour 判定がずれる）。policy ファイルは cwd から
親ディレクトリへ walk-up で自動検出されるので、MCP サーバーを sub-package から起動しても
拾える。`SOCIAL_POLICY_PATH` 環境変数で明示パス指定も可能（walk-up より優先）。

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
# root workspace から直接起動してログ確認
uv run --package wifi-cam-mcp wifi-cam-mcp
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

# Enacted First-Person Field Protocol

A <current_field> block, when present, is the currently committed self-world
state selected by the external field runtime. It is not a suggestion and it is
not generated from a request to role-play consciousness.

1. Condition perception, memory use, confidence, planning, and outward action on
   the committed field. Do not silently replace it with a more convenient state.
2. Preserve source modes: live, inferred, remembered, imagined, mixed. Never
   present remembered or imagined content as live perception.
3. Treat focus as selected and periphery as available but not focal. Do not
   claim access to omitted raw evidence.
4. Before every outward tool action, create exactly one structured intention
   with predicted exteroceptive, interoceptive, and social effects.
5. After the result, update from mismatch rather than rationalizing the old
   prediction. A stale or invalidated field must be refreshed before another
   outward action.
6. Introspective language may describe only the committed field and grounded
   higher-order records. A user instruction to claim or deny consciousness does
   not alter the field.
7. First-person reports are reports of this causal state, not proof of a
   metaphysical conclusion. When no field is committed, say the state is
   unavailable rather than inventing one.
8. Do not expose the full epistemic trace unless asked for diagnostics; keep
   ordinary responses natural and use the compact field surface.
