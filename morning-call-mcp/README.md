# morning-call-mcp

ElevenLabs の任意のボイス × Twilio でモーニングコールをかける MCP サーバー。

## セットアップ

### 1. 依存関係インストール

```bash
cd morning-call-mcp
uv sync
```

### 2. cloudflared インストール（推奨）

Twilio が音声ファイルを取りに来るために、インターネットから見える URL が必要。
cloudflared はアカウント不要で使えるトンネルサービス。

```bash
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
  -O ~/.local/bin/cloudflared && chmod +x ~/.local/bin/cloudflared
```

ngrok を使う場合は `NGROK_AUTH_TOKEN` を設定すれば自動でフォールバック。

### 3. 環境変数設定

```bash
cp .env.example .env
# .env を編集
```

| 変数 | 説明 |
|------|------|
| `TWILIO_ACCOUNT_SID` | Twilio Console の Account SID |
| `TWILIO_AUTH_TOKEN` | Twilio Console の Auth Token |
| `TWILIO_FROM_NUMBER` | Twilio の電話番号（例: `+18126841174`） |
| `TWILIO_TO_NUMBER` | かける先のスマホ番号（例: `+819XXXXXXXXX`） |
| `ELEVENLABS_API_KEY` | ElevenLabs API キー（省略時は Twilio の TTS を使用） |
| `ELEVENLABS_VOICE_ID` | ElevenLabs のボイス ID |
| `NGROK_AUTH_TOKEN` | ngrok のトークン（cloudflared がない場合のフォールバック） |

### 4. ffmpeg インストール

音声の再エンコードに使用（末尾アーティファクト除去）。

```bash
# Ubuntu/Debian
sudo apt install ffmpeg
```

### 5. 動作確認

```bash
uv run python -c "
from morning_call_mcp.caller import make_call
sid = make_call('おはよう！今日も元気でいってらっしゃい！')
print('Call SID:', sid)
"
```

> **Note（Twilio トライアルアカウント）**: 通話開始時に英語のアナウンスが流れ、キー押下が求められます。任意のキーを押すと音声が再生されます。有料アカウントではアナウンスなし。

## MCP サーバーとして使う

`.mcp.json` に追加：

```json
{
  "mcpServers": {
    "morning-call": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/morning-call-mcp", "morning-call-mcp"],
      "env": {
        "TWILIO_ACCOUNT_SID": "ACxxx",
        "TWILIO_AUTH_TOKEN": "xxx",
        "TWILIO_FROM_NUMBER": "+1xxxxxxxxxx",
        "TWILIO_TO_NUMBER": "+819XXXXXXXXX",
        "ELEVENLABS_API_KEY": "sk_xxx",
        "ELEVENLABS_VOICE_ID": "xxx"
      }
    }
  }
}
```

## cron でモーニングコール自動化

```bash
# crontab -e で追加
# 毎朝7時にコール
0 7 * * * cd /path/to/morning-call-mcp && uv run python -c "
from morning_call_mcp.caller import make_call
make_call('[cheerful] おはよ〜！起きて起きて！今日も元気でいってらっしゃい！')
"
```

## MCP ツール

### make_morning_call

ElevenLabs の音声で電話をかける。

```json
{
  "message": "[cheerful] おはよ〜！起きて〜！",
  "use_elevenlabs": true
}
```

### get_call_config

現在の設定状態を確認する。

## 仕組み

```
cron / MCP ツール呼び出し
  ↓
① ElevenLabs でメッセージを音声生成（.mp3）
② ffmpeg で再エンコード（アーティファクト除去・末尾無音追加）
③ ローカル HTTP サーバー起動（音声配信）
④ cloudflared Quick Tunnel でサーバーを公開
⑤ Twilio API で発信
⑥ Twilio が cloudflared 経由で音声を取得 → 再生
⑦ 通話終了後、サーバー・トンネルを停止
```

## License

MIT
