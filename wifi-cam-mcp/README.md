# WiFi Camera MCP Server

Tapo C210などのWiFiカメラをMCP経由で制御して、AIに部屋を見渡してもらうためのサーバー。

## 対応カメラ

- TP-Link Tapo C210 (3MP)
- TP-Link Tapo C220 (4MP)
- その他Tapoシリーズのパン・チルト対応カメラ

## できること

| ツール | 説明 |
|--------|------|
| `camera_capture` | 今見えてる景色を撮影 |
| `camera_pan_left` | 左を向く |
| `camera_pan_right` | 右を向く |
| `camera_tilt_up` | 上を向く |
| `camera_tilt_down` | 下を向く |
| `camera_look_around` | 部屋を見渡す（4方向撮影） |
| `camera_info` | カメラ情報取得 |
| `camera_presets` | プリセット位置一覧 |
| `camera_go_to_preset` | プリセット位置に移動 |

## セットアップ

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
