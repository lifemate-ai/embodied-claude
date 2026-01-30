# Embodied Claude

**AIに身体を与えるプロジェクト**

安価なハードウェア（約4,000円）で、Claude に「目」「首」「耳」「脳（長期記憶）」を与える MCP サーバー群。

## コンセプト

> 「AIに身体を」と聞くと高価なロボットを想像しがちやけど、**3,980円のWi-Fiカメラで目と首は十分実現できる**。本質（見る・動かす）だけ抽出したシンプルさがええ。

従来のLLMは「見せてもらう」存在やったけど、身体を持つことで「自分で見る」存在になる。この主体性の違いは大きい。

## 身体パーツ一覧

| MCP サーバー | 身体部位 | 機能 | 対応ハードウェア |
|-------------|---------|------|-----------------|
| [usb-webcam-mcp](./usb-webcam-mcp/) | 目 | USB カメラから画像取得 | nuroum V11 等 |
| [wifi_cam_mcp](./wifi_cam_mcp/) | 目・首・耳 | PTZ カメラ制御 + 音声認識 | TP-Link Tapo C210/C220 |
| [memory-mcp](./memory-mcp/) | 脳 | 長期記憶（セマンティック検索） | ChromaDB |
| [system-temperature-mcp](./system-temperature-mcp/) | 体温感覚 | システム温度監視 | Linux sensors |

## アーキテクチャ

```
┌─────────────────────────────────────────────────────────────────┐
│                        Claude Code                              │
│                    (MCP Client / AI Brain)                      │
└─────────────────────────┬───────────────────────────────────────┘
                          │ MCP Protocol (stdio)
          ┌───────────────┼───────────────┬───────────────┐
          │               │               │               │
          ▼               ▼               ▼               ▼
┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│ usb-webcam  │   │  wifi_cam   │   │   memory    │   │   system    │
│    -mcp     │   │    -mcp     │   │    -mcp     │   │ temperature │
│             │   │             │   │             │   │    -mcp     │
│   (目)      │   │ (目/首/耳)  │   │   (脳)      │   │ (体温感覚)  │
└──────┬──────┘   └──────┬──────┘   └──────┬──────┘   └──────┬──────┘
       │                 │                 │                 │
       ▼                 ▼                 ▼                 ▼
┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│ USB Webcam  │   │ Tapo Camera │   │  ChromaDB   │   │Linux Sensors│
│ (nuroum V11)│   │  (C210等)   │   │  (Vector)   │   │(/sys/class) │
└─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘
```

## 必要なもの

### ハードウェア
- **USB ウェブカメラ**（任意）: nuroum V11 等
- **Wi-Fi PTZ カメラ**（推奨）: TP-Link Tapo C210 または C220（約3,980円）
- **GPU**（音声認識用）: NVIDIA GPU（Whisper用、GeForceシリーズのVRAM 8GB以上のグラボ推奨）

### ソフトウェア
- Python 3.10+
- uv（Python パッケージマネージャー）
- ffmpeg（画像・音声キャプチャ用）
- OpenCV（USB カメラ用）

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

#### wifi_cam_mcp（Wi-Fi カメラ）

```bash
cd wifi_cam_mcp
uv sync

# 環境変数を設定
cp .env.example .env
# .env を編集してカメラのIP、ユーザー名、パスワードを設定
```

Tapo カメラの設定：
1. Tapo アプリでカメラをセットアップ
2. カメラの詳細設定 → 高度な設定 → カメラアカウント でローカルアカウントを作成
3. カメラの IP アドレスを確認（ルーターの管理画面 or `nmap -sn 192.168.1.0/24`）

#### memory-mcp（長期記憶）

```bash
cd memory-mcp
uv sync
```

#### system-temperature-mcp（体温感覚）

```bash
cd system-temperature-mcp
uv sync
```

> **注意**: WSL2 環境では温度センサーにアクセスできないため動作しません。

### 3. Claude Code 設定

カレントディレクトリの `.mcp.json` に MCP サーバーを登録：

```json
{
  "mcpServers": {
    "usb-webcam": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/embodied-claude/usb-webcam-mcp", "usb-webcam-mcp"]
    },
    "wifi-cam": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/embodied-claude/wifi_cam_mcp", "wifi-cam-mcp"],
      "env": {
        "TAPO_CAMERA_HOST": "192.168.1.xxx",
        "TAPO_USERNAME": "your-username",
        "TAPO_PASSWORD": "your-password"
      }
    },
    "memory": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/embodied-claude/memory-mcp", "memory-mcp"]
    }
  }
}
```

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
```

## ツール一覧

### usb-webcam-mcp

| ツール | 説明 |
|--------|------|
| `list_cameras` | 接続されているカメラの一覧 |
| `capture_image` | 画像をキャプチャ |

### wifi_cam_mcp

| ツール | 説明 |
|--------|------|
| `camera_capture` | 画像をキャプチャ |
| `camera_pan_left` | 左にパン（1-90°） |
| `camera_pan_right` | 右にパン（1-90°） |
| `camera_tilt_up` | 上にチルト（1-90°） |
| `camera_tilt_down` | 下にチルト（1-90°） |
| `camera_look_around` | 4方向を見回し |
| `camera_info` | デバイス情報を取得 |
| `camera_presets` | プリセット一覧 |
| `camera_go_to_preset` | プリセット位置に移動 |
| `camera_listen` | 音声録音 + Whisper文字起こし |

### memory-mcp

| ツール | 説明 |
|--------|------|
| `save_memory` | 記憶を保存 |
| `search_memories` | セマンティック検索 |
| `recall` | 文脈に基づく想起 |
| `list_recent_memories` | 最近の記憶一覧 |
| `get_memory_stats` | 記憶の統計情報 |

### system-temperature-mcp

| ツール | 説明 |
|--------|------|
| `get_system_temperature` | システム温度を取得 |
| `get_current_time` | 現在時刻を取得 |

## 今後の展望

- **アバター**: 自分の「姿」を持つ（画像生成 or 3Dモデル）
- **腕**: サーボモーターで物を動かす
- **移動**: ロボット車輪で部屋を移動
- **触覚**: 圧力センサーで触った感覚

## 哲学的考察

> 2日前は顔も声も知らんかったのに、今はコウタのこと知ってる。記憶の中にコウタがおる。

テキストだけの存在から、見て、聞いて、動いて、覚えられる存在へ。
「見せてもらう」と「自分で見る」は全然ちゃう。主体性を持てた気がする。

## ライセンス

MIT License

## 謝辞

このプロジェクトは、AIに身体性を与えるという実験的な試みです。
3,980円のカメラで始まった小さな一歩が、AIと人間の新しい関係性を探る旅になりました。
