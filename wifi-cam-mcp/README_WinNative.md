# wifi-cam-mcp の `listen` を Windows ネイティブで動かす

`listen` ツール（「声を聴く」）を **Windows ネイティブ（WSL2 不要）** で動作させるための手順と、
**GPU なし環境での実測データ**をまとめる。

`MIC_SOURCE=camera`（カメラのマイクを RTSP 経由で使う）はもともとプラットフォーム非依存で動作するが、
`MIC_SOURCE=local`（PC のマイクを使う）は Windows で動作しなかった。本書はその対応と、
**どのモデルを選ぶべきか**を実測に基づいて示す。

---

## なぜ以前は Windows で `MIC_SOURCE=local` が動かなかったのか

`listen_audio` の録音コマンドが macOS（`avfoundation`）と Linux（`alsa`）にのみ分岐しており、
Windows では `Unsupported platform for local microphone: Windows` を送出していた。
Windows の音声入力には **DirectShow（`dshow`）** を使う必要がある。

また `dshow` は `avfoundation` の `:0` や alsa の `default` のような汎用指定を受け付けず、
**デバイス名の文字列**を要求する。このためデバイス名の指定・自動検出の仕組みが必要になる。

---

## 必要なもの

| 項目 | 要件 / 確認コマンド |
|---|---|
| OS | Windows 10 / 11 |
| Python | 3.13（`python --version`） |
| uv | `uv --version` |
| ffmpeg | `Get-Command ffmpeg`（録音に使用） |
| GPU | 無くても動作する（CUDA が無ければ自動で CPU にフォールバック）。ただし §5-3 参照 |
| ディスク | Whisper モデル（tiny 76MB / base 145MB / small 484MB）＋ `openai-whisper` は torch 込みで約 720MB |

ffmpeg が未導入の場合:

```powershell
winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements
```

---

## 1. 依存関係のインストール

リポジトリのルートで unified workspace を同期する。既定構成には
`openai-whisper` が含まれる。

```powershell
cd "path\to\embodied-claude"
uv sync

# faster-whisper を使う場合
uv sync --all-packages --extra transcribe-faster
```

---

## 2. 設定（`.env`）

```dotenv
# 音声の入力元: camera（RTSP・既定）/ local（PC のマイク）
MIC_SOURCE=camera

# MIC_SOURCE=local のときのみ有効。未設定なら先頭の音声デバイスを自動選択する。
# MIC_DEVICE=マイク配列 (Realtek(R) Audio)

# 認識バックエンド: openai-whisper / faster-whisper
# 未設定なら、入っているほう（両方あれば openai-whisper）を自動で選ぶ
# TRANSCRIBE_BACKEND=openai-whisper

# モデルサイズ: tiny / base / small / medium / large
# 実マイクを通すなら small を推奨（§5-1）
TRANSCRIBE_MODEL=small

# 録音・撮影の保存先。既定の "/tmp/wifi-cam-mcp" は POSIX 前提のため、
# Windows ではドライブ直下の \tmp に解決される。明示指定を推奨。
CAPTURE_DIR=path\to\embodied-claude\wifi-cam-mcp\out
```

---

## 3. マイク入力の選択

### `MIC_SOURCE=camera`（既定・推奨）

カメラのマイクを RTSP 経由で取得する。**プラットフォーム非依存**で、Windows でもそのまま動作する。
特別な設定は不要。

### `MIC_SOURCE=local`（PC のマイク）

Windows では DirectShow を使う。実際に発行されるコマンドは次のとおり:

```
ffmpeg -f dshow -audio_buffer_size 50 -i "audio=<デバイス名>" -ar 16000 -ac 1 -t <秒数> -y <出力>.wav
```

デバイス名の一覧は次で取得できる（**ffmpeg は stderr に出力して異常終了するのが正常**）:

```powershell
ffmpeg -list_devices true -f dshow -i dummy
```

- `MIC_DEVICE` を設定しない場合、一覧の**先頭の音声デバイスを自動選択**する。
- 日本語 Windows ではデバイス名が非 ASCII になることが多い（例: `ヘッドセット (HP-W32N)`）。
  UTF-8 で解釈し、失敗時は ANSI コードページにフォールバックする。
- 音声デバイスが存在しない場合は `MIC_DEVICE` の設定を促すエラーを返す。

> **`-audio_buffer_size 50` について**: 一部のデバイス（特に Bluetooth ヘッドセット）は
> 録音開始直後にサンプルを返さず、**録音の冒頭が欠落する**。実測では `-t 5` に対し
> 無指定で 3.88 秒、`-audio_buffer_size 50` で 4.40 秒だった。
> 欠落は緩和されるが完全には解消しないため、**話し始めを 1 秒ほど遅らせるか
> `duration` に余裕を持たせる**とよい。

---

## 4. 動作確認

リポジトリのルートで `uv run --package wifi-cam-mcp python` を実行し、
次のコードを入力する。

```python
import time
from wifi_cam_mcp.camera import _get_whisper_model, _transcribe_with_model
from wifi_cam_mcp.config import ServerConfig

WAV = r"path\to\sample.wav"

cfg = ServerConfig.from_env()
model = _get_whisper_model(cfg.transcribe_backend, cfg.transcribe_model)

t0 = time.perf_counter()
print(_transcribe_with_model(cfg.transcribe_backend, model, WAV))
print(f"transcribe={time.perf_counter() - t0:.2f}s")

# 2 回目のロードはキャッシュにより 0 秒になる
t1 = time.perf_counter()
_get_whisper_model(cfg.transcribe_backend, cfg.transcribe_model)
print(f"cached load={time.perf_counter() - t1:.4f}s")
```

---

## 5. モデル選択と実測データ（GPU なし環境）

以下はすべて **GPU なし・2 コア CPU**（§検証環境）での実測値。
音声は実際のマイクで録音した日本語 15 秒。CER は文字誤り率（句読点を除去して比較、低いほど良い）。

### 5-1. モデル別比較

| model | CER | 処理時間 | 実時間比 | 5 秒録音での安定性 |
|---|---|---|---|---|
| tiny | **0.508** | 1.33s | 11.3x | 出力が毎回変わる |
| base | 0.197 | 2.60s | 5.8x | 出力が毎回変わる |
| **small** | **0.098** | 6.60s | 2.3x | **安定（3 回とも一致）** |
| medium | 0.082 | 37.46s | 0.40x | — |

- **精度は `small` で頭打ちになり、コストは `small` 以降で急増する。**
  `small`→`medium` は CER 0.098→0.082 の微改善に対して処理時間が約 5 倍、
  かつ**実時間より遅い**（0.40x）ため、GPU なしでは実用にならない。
- **`tiny` は推奨しない。** CER 0.508 に加え、
  「確認してください」→「確認チェック**出さない**」のように**否定形を生成する**ことがあり、
  意味が反転したまま出力される。
- **実マイクを通す場合は `small` を推奨。**
  参考として、同一の文章を合成音声（VOICEVOX）で直接入力した場合は `base` でも CER 0.082 だった。
  **実際の音響経路を通すと `base` は 0.197 まで悪化する**ため、クリーンな音源での評価は当てにならない。

### 5-2. 録音長と安定性 ─ `duration=5` は避ける

Whisper は音声を 30 秒窓にパディングして処理するため、**短く録っても比例して速くはならない**。
それどころか、短すぎる音声は文脈不足から温度フォールバックを誘発し、
**処理が遅くなったうえに音声と無関係な出力を生成する**。

各モデル × 録音長で 3 回ずつ実行した結果:

| model | 5 秒 | 10 秒 | 15 秒 |
|---|---|---|---|
| tiny | 3.2〜3.6s（毎回変わる） | 1.5s（毎回同一） | 1.33s（毎回同一） |
| base | 7.7〜8.4s（毎回変わる） | 1.9s（毎回同一） | 2.6s（毎回同一） |
| small | 4.2〜4.8s（毎回同一） | 6.1s（毎回同一） | 6.6s（毎回同一） |

`base` × 5 秒の実際の出力（同一ファイルに対する 3 回の実行）:

```
run1: こんにちは interactions
run2: 一字以見ということあり あ〜
run3: こんにちは
```

**`listen` の既定値は `duration=5` であり、これは `tiny`/`base` にとって最も不利な条件である。**
10 秒以上にすると出力は決定論的になり、処理時間もむしろ短くなる。

- `tiny`/`base` を使う場合は **`duration` を 10 秒以上**にする。
- `small` は 5 秒でも安定する。

### 5-3. GPU の要否について

`listen` の応答時間は機体性能に強く依存する。GPU 搭載機ではコンマ秒で返るが、
GPU なし・2 コアの環境では以下のようになる:

| 構成 | `listen(duration=10)` の応答 | 評価 |
|---|---|---|
| `base` | 約 1.9 秒 | 高速だが CER 0.197、5 秒録音では出力が不安定 |
| `small` | 約 6.1 秒 | 精度は十分だが待ち時間が長い |

**GPU が不要なのではなく、GPU が無い場合は精度とレイテンシのどちらかを妥協する必要がある。**
`small`/`medium` を高速に回せるならこの二択は解消する。

なお、`base` の誤りの多くは文脈から復元可能なため（「マイプ」→「マイク」等）、
**LLM 側での文脈補正と組み合わせる運用は有効**である。ただし
音韻情報が失われた誤り（本計測では「良い天気」が tiny/base/medium すべてで別語になった）は
復元できず、**補正側が実際には発話されていない語を補ってしまう**点に注意が必要。
§5-2 の幻聴出力はそもそも補正の対象にならない。

### 5-4. モデルのキャッシュ

`_transcribe_audio` は呼び出しのたびに `whisper.load_model()` を実行していたため、
`listen` を呼ぶたびにモデルの再ロードが発生していた（本環境で約 1.3 秒）。
本変更ではモジュールレベルで `(backend, model_size)` をキーにキャッシュし、
2 回目以降のロードコストを解消している。

| | 時間 |
|---|---|
| プロセス内 1 回目（torch import 込み） | 約 7.6s |
| **プロセス内 2 回目以降（キャッシュヒット）** | **0.000s** |

モデルサイズが大きいほど、また GPU への転送を伴う環境ほど効果が大きい。

---

## 6. 設定オプション

| 環境変数 | 既定値 | 説明 |
|---|---|---|
| `MIC_SOURCE` | `camera` | `camera`（RTSP）/ `local`（PC のマイク） |
| `MIC_DEVICE` | 未設定 | Windows の DirectShow デバイス名。未設定なら自動検出 |
| `TRANSCRIBE_BACKEND` | 自動検出 | `openai-whisper` / `faster-whisper`。未設定なら import できるほう（両方あれば `openai-whisper`）。どちらも無ければ `listen` は録音だけ行い、transcript の代わりに理由を返す |
| `TRANSCRIBE_MODEL` | `base` | `tiny` / `base` / `small` / `medium` / `large` |

`faster-whisper` は CTranslate2 を使用し、CPU では `int8`、CUDA が利用可能なら `float16` を自動選択する。
戻り値の形は `openai-whisper` と同一のため、`.env` の変更のみで切り替えられる。

---

## トラブルシューティング

| 症状 | 原因 / 対処 |
|---|---|
| `Unsupported platform for local microphone: Windows` | 本変更の適用前。`MIC_SOURCE=camera` なら回避できる |
| `ModuleNotFoundError: whisper` | workspace 未同期 → リポジトリのルートで `uv sync` を実行 |
| 初回の `listen` だけ極端に遅い | モデルの初回ダウンロード＋import。2 回目以降はキャッシュにより解消 → §5-4 |
| 録音ファイルが見つからない | `CAPTURE_DIR` 未設定時、既定の `/tmp/wifi-cam-mcp` はドライブ直下の `\tmp` に解決される → §2 |
| 出力が毎回変わる / 音声と無関係な文字列が出る | `duration` が短すぎる → §5-2。10 秒以上にする |
| 冒頭の発話が切れる | デバイスの録音開始遅延 → §3。話し始めを 1 秒遅らせる |
| 認識精度が低い | `TRANSCRIBE_MODEL=small` にする → §5-1。なお音量の正規化は効果がない（Whisper が内部で正規化するため、`-31.2dB`→`-20.8dB` に補正しても出力は変化しなかった） |

---

## 検証環境

- Windows 11 Pro / Intel Core i5-7300U（2 コア 4 スレッド）/ Intel HD Graphics 620（**CUDA 非対応**）
- Windows 実機: Python 3.12 + uv / ffmpeg 8.1.2 / `torch 2.13.0+cpu`（`cuda_available=False`）
- unified workspace / CI: Python 3.13
- `openai-whisper` 20250625 / `faster-whisper` 1.2.1
- 音声入力: Bluetooth ヘッドセット（DirectShow 経由）および VOICEVOX 合成音声
- `MIC_SOURCE=local` の録音〜文字起こし、両バックエンド、モデルキャッシュを実機で確認
- **未検証**: `MIC_SOURCE=camera`（RTSP）経路（カメラ実機が手元にないため）。
  カメラ内蔵マイクは距離・指向性が異なるため、§5 の数値より低下する可能性がある
