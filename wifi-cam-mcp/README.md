# WiFi Camera MCP Server

Tapo C210などのWiFiカメラをMCP経由で制御して、AIに部屋を見渡してもらうためのサーバー。

## 対応カメラ

- TP-Link Tapo C200 (2MP, 1080p)
- TP-Link Tapo C210 (3MP)
- TP-Link Tapo C220 (4MP)
- TP-Link Tapo C230 (5MP, 3K)
- その他Tapoシリーズのパン・チルト対応カメラ

## できること

| ツール | 説明 |
|--------|------|
| `see` | 今見えてる景色を撮影 |
| `look_left` | 左を向く |
| `look_right` | 右を向く |
| `look_up` | 上を向く |
| `look_down` | 下を向く |
| `look_around` | 部屋を見渡す（4方向撮影） |
| `camera_info` | カメラ情報取得 |
| `camera_presets` | プリセット位置一覧 |
| `camera_go_to_preset` | プリセット位置に移動 |

### 互換エイリアス（段階廃止）

旧名のツールは互換のため当面利用可能です。呼び出すと非推奨メッセージが返ります。

| 旧ツール名 | 新ツール名 |
|-----------|------------|
| `camera_capture` | `see` |
| `camera_pan_left` | `look_left` |
| `camera_pan_right` | `look_right` |
| `camera_tilt_up` | `look_up` |
| `camera_tilt_down` | `look_down` |
| `camera_look_around` | `look_around` |

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

#### 依存関係のインストール

```bash
uv sync
```

#### 動作確認

```bash
uv run wifi-cam-mcp
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
        "--directory",
        "/path/to/wifi-cam-mcp",
        "run",
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
        "--directory",
        "/path/to/wifi-cam-mcp",
        "run",
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

## テスト

### Python版

```bash
uv run pytest
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

### Windows での注意事項

- `ffmpeg` が PATH に入っていること（RTSP フォールバックとオーディオ録音に必要）
- キャプチャ画像のデフォルト保存先: `%LOCALAPPDATA%\wifi-cam-mcp`
- `CAPTURE_DIR` を設定する場合はバックスラッシュ使用（例: `C:\Users\you\captures`）

### C200 固有の注意事項

- ONVIF スナップショットに対応していない場合があります（RTSP フォールバックが自動で動作します）
- モーターが C210/C220 より遅いため、`TAPO_MOTOR_DELAY=0.8` を推奨
- 赤外線LEDは 850nm（赤く光るタイプ）
- microSD は最大128GBまで対応

## 注意事項

- **Python版**: pytapoは非公式ライブラリのため、TP-Linkの仕様変更で動作しなくなる可能性があります
- カメラはローカルネットワーク内からのみアクセス可能です
- 認証情報（.envファイル）は絶対にGitにコミットしないでください

## ライセンス

MIT License
