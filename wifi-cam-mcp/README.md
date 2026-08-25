# WiFi Camera MCP Server

Tapo C210などのWiFiカメラをMCP経由で制御して、AIに部屋を見渡してもらうためのサーバー。

## 対応カメラ

- TP-Link Tapo C210 (3MP)
- TP-Link Tapo C220 (4MP)
- その他Tapoシリーズのパン・チルト対応カメラ

## できること

登録されるツール名は `server.py` の `list_tools()` のとおりです。

| ツール | 説明 |
|--------|------|
| `see` | 今見えてる景色を撮影（カメラの向きも一緒に返す） |
| `look_left` | 左を向く（`degrees`、既定 30） |
| `look_right` | 右を向く（`degrees`、既定 30） |
| `look_up` | 上を向く（`degrees`、既定 20） |
| `look_down` | 下を向く（`degrees`、既定 20） |
| `look_around` | 部屋を見渡す（正面・左・右・上の4方向撮影） |
| `camera_info` | カメラ情報取得 |
| `camera_presets` | プリセット位置一覧 |
| `camera_go_to_preset` | プリセット位置に移動（`preset_id`） |
| `listen` | マイクで聞く（`duration` 秒、`transcribe` で文字起こし） |

### 右カメラ（両目）を追加する

同じ場所に 2 台目のカメラを置き、`TAPO_RIGHT_*` を設定すると、起動時に右カメラへ
接続できた場合だけ次の 13 ツールが追加されます（`.env.example` も参照）。

| 環境変数 | 説明 |
|----------|------|
| `TAPO_RIGHT_CAMERA_HOST` | 右カメラの IP アドレス（これが無いと右カメラは無効） |
| `TAPO_RIGHT_USERNAME` / `TAPO_RIGHT_PASSWORD` | 右カメラの認証情報（未設定なら `TAPO_USERNAME` / `TAPO_PASSWORD` を流用） |
| `TAPO_RIGHT_ONVIF_PORT` | ONVIF ポート（未設定なら `TAPO_ONVIF_PORT`、既定 2020） |
| `TAPO_RIGHT_STREAM_URL` | RTSP URL（未設定なら自動検出） |
| `TAPO_RIGHT_MOUNT_MODE` | `normal` / `ceiling`（未設定なら `TAPO_MOUNT_MODE`） |
| `TAPO_RIGHT_PTZ_MODE` | `auto` / `relative` / `continuous`（未設定なら `TAPO_PTZ_MODE`） |

| ツール | 説明 |
|--------|------|
| `see_right` | 右目だけで見る |
| `see_both` | 両目で同時に撮影。左右の画像を別々に返す。手前・奥の関係（オクルージョン）の確認や視点の比較に使える。**視差や奥行きは計算しない** |
| `right_eye_look_left` / `right_eye_look_right` / `right_eye_look_up` / `right_eye_look_down` | 右目だけを動かす |
| `both_eyes_look_left` / `both_eyes_look_right` / `both_eyes_look_up` / `both_eyes_look_down` | 両目を同じ向きに動かす |
| `get_eye_positions` | 両目の現在のパン・チルト角を取得 |
| `align_eyes` | 右目を左目と同じ向きに合わせる |
| `reset_eye_positions` | 両目の位置トラッキングを (0, 0) にリセット |

### 複数拠点にカメラを置く

離れた場所にカメラを置きたい場合は、右カメラではなく **wifi-cam-mcp を MCP サーバーとして
複数回登録**し、それぞれに別の `TAPO_CAMERA_HOST` を env で渡してください
（例: `wifi-cam-living` と `wifi-cam-office`）。ツール名は MCP サーバー名で名前空間が
分かれるので衝突しません。このとき `TAPO_RIGHT_*` は設定しないでおくと、両目系のツールは
生えません。両目系は同じ場所に並べた左右ペアのためのものです。

## セットアップ

Windows ネイティブでローカルマイクを使う場合は
[README_WinNative.md](README_WinNative.md) も参照してください。

### 1. カメラの初期設定（Tapoアプリ）

1. スマホに「TP-Link Tapo」アプリをインストール
2. Tapoアカウントを作成（メールアドレスとパスワード）
3. アプリから「デバイスを追加」→ カメラを選択
4. カメラの電源を入れ、アプリの指示に従ってWiFi接続

### 2. カメラのIPアドレスを調べる

以下のいずれかの方法で確認：

| 方法 | 手順 |
|------|------|
| **Tapoアプリ** | カメラ設定 → デバイス情報 → IPアドレス |
| **ルーター管理画面** | 接続機器一覧から「Tapo_C210」等を探す |
| **nmapコマンド** | `nmap -sn 192.168.1.0/24` |

> **Tips**: ルーターでDHCP予約（IP固定）を設定しておくと、カメラ再起動後もIPアドレスが変わらず便利です

### 3. カメラのアカウントを作る

1. Tapoアプリ -> ホーム ->  (カメラ名)を選択 -> 右上の歯車アイコンをタップ -> 「高度な設定」をタップ
2. 「カメラのアカウント」がオフになっているのでタップ -> 「カメラのアカウント」オン -> 「アカウント情報」
3. カメラのアカウントのユーザー名（user-name）とパスワード（user-password）を設定（後で使う）
  - ローカルのアカウントでTP-Linkのアカウントとは無関係なので注意

### 4. 環境変数の設定

```bash
cp .env.example .env
```

`.env` を編集：

```
TAPO_CAMERA_HOST=192.168.1.100    # カメラのIPアドレス
TAPO_USERNAME=your-name     # Tapoカメラ（TP-Linkアカウントではない）のユーザー名
TAPO_PASSWORD=your-password # Tapoカメラ（TP-Linkアカウントではない）のパスワード
```

---

### 5. 実行

#### workspace のインストール

```bash
uv sync
```

#### 動作確認

```bash
uv run --package wifi-cam-mcp wifi-cam-mcp
```

## Claude Desktopで使う

`claude_desktop_config.json`  または適切な設定ファイルに追加：

### Python版

```json
{
  "mcpServers": {
    "wifi-cam": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/path/to/embodied-claude",
        "--package",
        "wifi-cam-mcp",
        "wifi-cam-mcp"
      ],
      "env": {
        "TAPO_CAMERA_HOST": "192.168.1.100",
        "TAPO_USERNAME": "your-name",
        "TAPO_PASSWORD": "your-password"
      }
    }
  }
}
```

## Claude Codeで使う

`.mcp.json` をプロジェクトルートまたはホームディレクトリに作成：

### Python版

```json
{
  "mcpServers": {
    "wifi-cam": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/path/to/embodied-claude",
        "--package",
        "wifi-cam-mcp",
        "wifi-cam-mcp"
      ],
      "env": {
        "TAPO_CAMERA_HOST": "192.168.1.100",
        "TAPO_USERNAME": "your-name",
        "TAPO_PASSWORD": "your-password"
      }
    }
  }
}
```

## 使用例

Claudeに話しかける：

- 「今カメラに何が映ってる？」
- 「ちょっと左を見て」
- 「部屋全体を見渡して」
- 「窓は開いてる？」

## 検証

### Python版

```bash
uv run ruff check wifi-cam-mcp
```

## トラブルシューティング

### カメラに接続できない

- カメラとPCが同じネットワーク上にあるか確認
- IPアドレスが正しいか確認（Tapoアプリで再確認）
- ファイアウォールが通信をブロックしていないか確認

### 認証エラー

- カメラアカウントのメールアドレスとパスワードが正しいか確認

### 画像が取得できない

- カメラのファームウェアを最新に更新
- カメラを再起動

## 注意事項

- **Python版**: pytapoは非公式ライブラリのため、TP-Linkの仕様変更で動作しなくなる可能性があります
- カメラはローカルネットワーク内からのみアクセス可能です
- 認証情報（.envファイル）は絶対にGitにコミットしないでください

## ライセンス

MIT License
