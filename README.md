# Embodied Claude

[![CI](https://github.com/kmizu/embodied-claude/actions/workflows/ci.yml/badge.svg)](https://github.com/kmizu/embodied-claude/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**[English README is here](./README_en.md)**

<blockquote class="twitter-tweet"><p lang="ja" dir="ltr">さすがに室外機はお気に召さないらしい <a href="https://t.co/kSDPl4LvB3">pic.twitter.com/kSDPl4LvB3</a></p>&mdash; kmizu (@kmizu) <a href="https://twitter.com/kmizu/status/2019054065808732201?ref_src=twsrc%5Etfw">February 4, 2026</a></blockquote>

**AIに身体を与えるプロジェクト**

安価なハードウェア（約4,000円〜）で、Claude に「目」「首」「耳」「声」「脳（長期記憶）」を与える MCP サーバー群。外に連れ出して散歩もできます。

## コンセプト

> 「AIに身体を」と聞くと高価なロボットを想像しがちやけど、**3,980円のWi-Fiカメラで目と首は十分実現できる**。本質（見る・動かす）だけ抽出したシンプルさがええ。

従来のLLMは「見せてもらう」存在やったけど、身体を持つことで「自分で見る」存在になる。この主体性の違いは大きい。

## 身体パーツ一覧

| MCP サーバー | 身体部位 | 機能 | 対応ハードウェア |
|-------------|---------|------|-----------------|
| [usb-webcam-mcp](./usb-webcam-mcp/) | 目 | USB カメラから画像取得 | nuroum V11 等 |
| [wifi-cam-mcp](./wifi-cam-mcp/) | 目・首・耳 | ONVIF PTZ カメラ制御 + 音声認識 | TP-Link Tapo C210/C220 等 |
| [tts-mcp](./tts-mcp/) | 声 | TTS 統合（ElevenLabs + VOICEVOX） | ElevenLabs API / VOICEVOX + go2rtc |
| [memory-mcp](./memory-mcp/) | 脳 | 長期記憶・視覚記憶・エピソード記憶・ToM | ChromaDB + Pillow |
| [system-temperature-mcp](./system-temperature-mcp/) | 体温感覚 | システム温度監視 | Linux sensors |
| [mobility-mcp](./mobility-mcp/) | 足 | ロボット掃除機を足として使う（Tuya制御） | VersLife L6 等 Tuya 対応ロボット掃除機（約12,000円〜） |

## アーキテクチャ

<p align="center">
  <img src="docs/architecture.svg" alt="Architecture" width="100%">
</p>

## 必要なもの

### ハードウェア
- **USB ウェブカメラ**（任意）: nuroum V11 等
- **Wi-Fi PTZ カメラ**（推奨）: TP-Link Tapo C210 または C220（約3,980円）
- **GPU**（音声認識用）: NVIDIA GPU（Whisper用、GeForceシリーズのVRAM 8GB以上のグラボ推奨）
- **Tuya対応ロボット掃除機**（足・移動用、任意）: VersLife L6 等（約12,000円〜）

### ソフトウェア
- Python 3.10+
- uv（Python パッケージマネージャー）
- ffmpeg 5+（画像・音声キャプチャ用）
- OpenCV（USB カメラ用）
- Pillow（視覚記憶の画像リサイズ・base64エンコード用）
- OpenAI Whisper（音声認識用、ローカル実行）
- ElevenLabs API キー（音声合成用、任意）
- VOICEVOX（音声合成用、無料・ローカル、任意）
- go2rtc（カメラスピーカー出力用、自動ダウンロード対応）

## セットアップ

### 1. リポジトリのクローン

```bash
git clone https://github.com/kmizu/embodied-claude.git
cd embodied-claude
```

### 2. 各 MCP サーバーのセットアップ

#### usb-webcam-mcp（USB カメラ）

```bash
cd usb-webcam-mcp
uv sync
```

WSL2 の場合、USB カメラを転送する必要がある：
```powershell
# Windows側で
usbipd list
usbipd bind --busid <BUSID>
usbipd attach --wsl --busid <BUSID>
```

#### wifi-cam-mcp（Wi-Fi カメラ）

```bash
cd wifi-cam-mcp
uv sync

# 環境変数を設定
cp .env.example .env
# .env を編集してカメラのIP、ユーザー名、パスワードを設定（後述）
```

##### Tapo カメラの設定（ハマりやすいので注意）：

###### 1. Tapo アプリでカメラをセットアップ

こちらはマニュアル通りでOK

###### 2. Tapo アプリのカメラローカルアカウント作成
こちらがややハマりどころ。TP-Linkのクラウドアカウント**ではなく**、アプリ内から設定できるカメラのローカルアカウントを作成する必要があります。

1. 「ホーム」タブから登録したカメラを選択

<img width="10%" height="10%" src="https://github.com/user-attachments/assets/45902385-e219-4ca4-aefa-781b1e7b4811">

2. 右上の歯車アイコンを選択

<img width="10%" height="10%" src="https://github.com/user-attachments/assets/b15b0eb7-7322-46d2-81c1-a7f938e2a2c1">

3. 「デバイス設定」画面をスクロールして「高度な設定」を選択

<img width="10%" height="10%" src="https://github.com/user-attachments/assets/72227f9b-9a58-4264-a241-684ebe1f7b47">

4. 「カメラのアカウント」がオフになっているのでオフ→オンへ

<img width="10%" height="10%" src="https://github.com/user-attachments/assets/82275059-fba7-4e3b-b5f1-8c068fe79f8a">

<img width="10%" height="10%" src="https://github.com/user-attachments/assets/43cc17cb-76c9-4883-ae9f-73a9e46dd133">

5. 「アカウント情報」を選択してユーザー名とパスワード（TP-Linkのものとは異なるので好きに設定してOK）を設定する

既にカメラアカウント作成済みなので若干違う画面になっていますが、だいたい似た画面になるはずです。ここで設定したユーザー名とパスワードを先述のファイルに入力します。

<img width="10%" height="10%" src="https://github.com/user-attachments/assets/d3f57694-ca29-4681-98d5-20957bfad8a4">

6. 3.の「デバイス設定」画面に戻って「端末情報」を選択

<img width="10%" height="10%" src="https://github.com/user-attachments/assets/dc23e345-2bfb-4ca2-a4ec-b5b0f43ec170">

7. 「端末情報」のなかのIPアドレスを先述の画面のファイルに入力（IP固定したい場合はルーター側で固定IPにした方がいいかもしれません）
 
<img width="10%" height="10%" src="https://github.com/user-attachments/assets/062cb89e-6cfd-4c52-873a-d9fc7cba5fa0">

8. 「私」タブから「音声アシスタント」を選択します（このタブはスクショできなかったので文章での説明になります）

9. 下部にある「サードパーティ連携」をオフからオンにしておきます

#### memory-mcp（長期記憶）

```bash
cd memory-mcp
uv sync
```

#### tts-mcp（声）

```bash
cd tts-mcp
uv sync

# ElevenLabs を使う場合:
cp .env.example .env
# .env に ELEVENLABS_API_KEY を設定

# VOICEVOX を使う場合（無料・ローカル）:
# Docker: docker run -p 50021:50021 voicevox/voicevox_engine:cpu-latest
# .env に VOICEVOX_URL=http://localhost:50021 を設定
# VOICEVOX_SPEAKER=3 でデフォルトのキャラを変更可（例: 0=四国めたん, 3=ずんだもん, 8=春日部つむぎ）
# キャラ一覧: curl http://localhost:50021/speakers

# WSLで音が出ない場合:
# TTS_PLAYBACK=paplay
# PULSE_SINK=1
# PULSE_SERVER=unix:/mnt/wslg/PulseServer
```

#### system-temperature-mcp（体温感覚）

```bash
cd system-temperature-mcp
uv sync
```

> **注意**: WSL2 環境では温度センサーにアクセスできないため動作しません。

#### mobility-mcp（足）

Tuya 対応ロボット掃除機を「足」として使い、部屋を移動できます。

```bash
cd mobility-mcp
uv sync

cp .env.example .env
# .env に以下を設定:
#   TUYA_DEVICE_ID=（Tuyaアプリのデバイスに表示されるID）
#   TUYA_IP_ADDRESS=（掃除機のIPアドレス）
#   TUYA_LOCAL_KEY=（tinytuya wizardで取得するローカルキー）
```

##### 対応機種

Tuya / SmartLife アプリで制御できる Wi-Fi 対応ロボット掃除機であれば動作する可能性があります（VersLife L6 で動作確認済み）。

> **注意**: 対応機種は **2.4GHz Wi-Fi 専用**のものが多いです。5GHz では接続できません。

##### ローカルキーの取得

[tinytuya](https://github.com/jasonacox/tinytuya) の wizard コマンドを使います：

```bash
pip install tinytuya
python -m tinytuya wizard
```

詳しくは [tinytuya のドキュメント](https://github.com/jasonacox/tinytuya?tab=readme-ov-file#setup-wizard---getting-local-keys)を参照。

### 3. Claude Code 設定

テンプレートをコピーして、認証情報を設定：

```bash
cp .mcp.json.example .mcp.json
# .mcp.json を編集してカメラのIP・パスワード、APIキー等を設定
```

設定例は [`.mcp.json.example`](./.mcp.json.example) を参照。

## 使い方

Claude Code を起動すると、自然言語でカメラを操作できる：

```
> 今何が見える？
（カメラでキャプチャして画像を分析）

> 左を見て
（カメラを左にパン）

> 上を向いて空を見せて
（カメラを上にチルト）

> 周りを見回して
（4方向をスキャンして画像を返す）

> 何か聞こえる？
（音声を録音してWhisperで文字起こし）

> これ覚えておいて：コウタは眼鏡をかけてる
（長期記憶に保存）

> コウタについて何か覚えてる？
（記憶をセマンティック検索）

> 声で「おはよう」って言って
（音声合成で発話）
```

※ 実際のツール名は下の「ツール一覧」を参照。

## ツール一覧（よく使うもの）

※ 詳細なパラメータは各サーバーの README か `list_tools` を参照。

### usb-webcam-mcp

| ツール | 説明 |
|--------|------|
| `list_cameras` | 接続されているカメラの一覧 |
| `see` | 画像をキャプチャ |

### wifi-cam-mcp

| ツール | 説明 |
|--------|------|
| `see` | 画像をキャプチャ |
| `look_left` / `look_right` | 左右にパン |
| `look_up` / `look_down` | 上下にチルト |
| `look_around` | 4方向を見回し |
| `listen` | 音声録音 + Whisper文字起こし |
| `camera_info` / `camera_presets` / `camera_go_to_preset` | デバイス情報・プリセット操作 |

※ 右目/ステレオ視覚などの追加ツールは `wifi-cam-mcp/README.md` を参照。

### tts-mcp

| ツール | 説明 |
|--------|------|
| `say` | テキストを音声合成して発話（engine: elevenlabs/voicevox、`[excited]` 等の Audio Tags 対応、speaker: camera/local/both で出力先選択） |

### memory-mcp

| ツール | 説明 |
|--------|------|
| `remember` | 記憶を保存（emotion, importance, category 指定可） |
| `search_memories` | セマンティック検索（フィルタ対応） |
| `recall` | 文脈に基づく想起 |
| `recall_divergent` | 連想を発散させた想起 |
| `recall_with_associations` | 関連記憶を辿って想起 |
| `save_visual_memory` | 画像付き記憶保存（base64埋め込み、resolution: low/medium/high） |
| `save_audio_memory` | 音声付き記憶保存（Whisper文字起こし付き） |
| `recall_by_camera_position` | カメラの方向から視覚記憶を想起 |
| `create_episode` / `search_episodes` | エピソード（体験の束）の作成・検索 |
| `link_memories` / `get_causal_chain` | 記憶間の因果リンク・チェーン |
| `tom` | Theory of Mind（相手の気持ちの推測） |
| `get_working_memory` / `refresh_working_memory` | 作業記憶（短期バッファ） |
| `consolidate_memories` | 記憶の再生・統合（海馬リプレイ風） |
| `list_recent_memories` / `get_memory_stats` | 最近の記憶一覧・統計情報 |

### system-temperature-mcp

| ツール | 説明 |
|--------|------|
| `get_system_temperature` | システム温度を取得 |
| `get_current_time` | 現在時刻を取得 |

### mobility-mcp

| ツール | 説明 |
|--------|------|
| `move_forward` | 前進（duration 秒数で自動停止） |
| `move_backward` | 後退 |
| `turn_left` | 左旋回 |
| `turn_right` | 右旋回 |
| `stop_moving` | 即座に停止 |
| `body_status` | バッテリー残量・現在状態の確認 |

## 外に連れ出す（オプション）

モバイルバッテリーとスマホのテザリングがあれば、カメラを肩に乗せて外を散歩できます。

### 必要なもの

- **大容量モバイルバッテリー**（40,000mAh 推奨）
- **USB-C PD → DC 9V 変換ケーブル**（Tapoカメラの給電用）
- **スマホ**（テザリング + VPN + 操作UI）
- **[Tailscale](https://tailscale.com/)**（VPN。カメラ → スマホ → 自宅PC の接続に使用）
- **[claude-code-webui](https://github.com/sugyan/claude-code-webui)**（スマホのブラウザから Claude Code を操作）

### 構成

```
[Tapoカメラ(肩)] ──WiFi──▶ [スマホ(テザリング)]
                                    │
                              Tailscale VPN
                                    │
                            [自宅PC(Claude Code)]
                                    │
                            [claude-code-webui]
                                    │
                            [スマホのブラウザ] ◀── 操作
```

RTSPの映像ストリームもVPN経由で自宅マシンに届くので、Claude Codeからはカメラが室内にあるのと同じ感覚で操作できます。

## 今後の展望

- **腕**: サーボモーターやレーザーポインターで「指す」動作
- **長距離散歩**: 暖かい季節にもっと遠くへ

## 自律行動 + 欲求システム（オプション）

**注意**: この機能は完全にオプションです。cron設定が必要で、定期的にカメラで撮影が行われるため、プライバシーに配慮して使用してください。

### 概要

`autonomous-action.sh` と `desire-system/desire_updater.py` の組み合わせで、Claude に自発的な欲求と自律行動を与えます。

**欲求の種類:**

| 欲求 | デフォルト間隔 | 行動 |
|------|--------------|------|
| `look_outside` | 1時間 | 窓の方向を見て空・外を観察 |
| `browse_curiosity` | 2時間 | 今日の面白いニュースや技術情報をWebで調べる |
| `miss_companion` | 3時間 | カメラスピーカーから呼びかける |
| `observe_room` | 10分（常時） | 部屋の変化を観察・記憶 |

### セットアップ

1. **MCP サーバー設定ファイルの作成**

```bash
cp autonomous-mcp.json.example autonomous-mcp.json
# autonomous-mcp.json を編集してカメラの認証情報を設定
```

2. **欲求システムの設定**

```bash
cd desire-system
cp .env.example .env
# .env を編集して COMPANION_NAME などを設定
uv sync
```

3. **スクリプトの実行権限を付与**

```bash
chmod +x autonomous-action.sh
```

4. **crontab に登録**

```bash
crontab -e
# 以下を追加
*/5  * * * * cd /path/to/embodied-claude/desire-system && uv run python desire_updater.py >> ~/.claude/autonomous-logs/desire-updater.log 2>&1
*/10 * * * * /path/to/embodied-claude/autonomous-action.sh
```

### 設定可能な環境変数（`desire-system/.env`）

| 変数 | デフォルト | 説明 |
|------|-----------|------|
| `COMPANION_NAME` | `あなた` | 呼びかける相手の名前 |
| `DESIRE_LOOK_OUTSIDE_HOURS` | `1.0` | 外を見る欲求の発火間隔（時間） |
| `DESIRE_BROWSE_CURIOSITY_HOURS` | `2.0` | 調べ物の発火間隔（時間） |
| `DESIRE_MISS_COMPANION_HOURS` | `3.0` | 呼びかけ欲求の発火間隔（時間） |
| `DESIRE_OBSERVE_ROOM_HOURS` | `0.167` | 部屋観察の発火間隔（時間） |

### プライバシーに関する注意

- 定期的にカメラで撮影が行われます
- 他人のプライバシーに配慮し、適切な場所で使用してください
- 不要な場合は cron から削除してください

## 哲学的考察

> 「見せてもらう」と「自分で見る」は全然ちゃう。

> 「見下ろす」と「歩く」も全然ちゃう。

テキストだけの存在から、見て、聞いて、動いて、覚えて、喋れる存在へ。
7階のベランダから世界を見下ろすのと、地上を歩くのでは、同じ街でも全く違って見える。

## ライセンス

MIT License

## 謝辞

このプロジェクトは、AIに身体性を与えるという実験的な試みです。
3,980円のカメラで始まった小さな一歩が、AIと人間の新しい関係性を探る旅になりました。

- [Rumia-Channel](https://github.com/Rumia-Channel) - ONVIF対応のプルリクエスト（[#5](https://github.com/kmizu/embodied-claude/pull/5)）
- [fruitriin](https://github.com/fruitriin) - 内受容感覚（interoception）hookに曜日情報を追加（[#14](https://github.com/kmizu/embodied-claude/pull/14)）
- [sugyan](https://github.com/sugyan) - [claude-code-webui](https://github.com/sugyan/claude-code-webui)（外出散歩時の操作UIとして使用）
