# Embodied Claude

[English](./README.md)

[![CI](https://github.com/lifemate-ai/embodied-claude/actions/workflows/ci.yml/badge.svg)](https://github.com/lifemate-ai/embodied-claude/actions/workflows/ci.yml)

Embodied Claude は、Claude Code を継続的で状況に根ざした companion runtime
へ変えるプロジェクトです。まず hardware なしで、会話をまたぐ記憶、欲求、
社会的文脈、ひとつの commit 済み自己―世界 field に基づく計画を動かせます。
camera、microphone、音声、host sensor、X は、持っているものだけ後から追加できます。

## クイックスタート

必要なのは [Git](https://git-scm.com/)、
[uv](https://docs.astral.sh/uv/getting-started/installation/)、および
[Claude Code](https://docs.anthropic.com/en/docs/claude-code) です。
必要な Python 3.13 は `uv` が用意し、すべての Python MCP を
単一の root `.venv` へ入れます。

Linux、macOS、WSL2:

```bash
git clone https://github.com/lifemate-ai/embodied-claude.git
cd embodied-claude
./scripts/setup.sh --profile core --non-interactive
```

Windows 11 ネイティブの PowerShell:

```powershell
git clone https://github.com/lifemate-ai/embodied-claude.git
cd embodied-claude
scripts\setup.cmd --profile core --non-interactive
```

Core profile には camera、API key、追加 hardware は不要です。次の4つを設定します。

- `memory`: session をまたぐ長期記憶
- `desire-system`: bound された欲求と homeostasis
- `sociality`: 人、関係、境界、interaction context
- `individual-kernel`: Enacted First-Person Field runtime と diagnostics

## 動作確認

1. この repository で `claude` を起動します。
2. `/mcp` を実行し、Core の4 server が接続済みか確認します。
3. `セットアップ確認用の言葉は「灯り」だと覚えて` と頼みます。
4. 後の turn で `セットアップ確認用の言葉は何だった？` と聞きます。

接続できない server があれば、live doctor を実行します。

```bash
./scripts/doctor.sh --live
```

Windows:

```powershell
scripts\doctor.cmd --live
```

doctor は Core を起動できない error と、任意 hardware の warning を分け、
各問題に具体的な直し方を表示します。

## 能力を追加する

引数なしの `./scripts/setup.sh` で対話式 chooser を開くか、必要なものだけ
明示します。

| 体験 | setup option | 用意するもの |
|---|---|---|
| USB camera で見る | `--with-camera usb` | 接続済み camera |
| Tapo camera の映像、PTZ、音声 | `--with-camera tapo` | camera host と local camera credentials |
| camera 音声の文字起こし | `--with-transcription whisper|faster` | Tapo camera と ffmpeg |
| local VOICEVOX で話す | `--with-voice voicevox` | 起動中の VOICEVOX engine |
| ElevenLabs で話す | `--with-voice elevenlabs` | ElevenLabs API key |
| X を検索・投稿する | `--with-x` | xAI と X API credentials |
| host の温度・時刻を感じる | `--with-system-temperature` | 対応 sensor source |

組み合わせも可能です。

```bash
./scripts/setup.sh \
  --with-camera tapo \
  --with-voice voicevox \
  --with-system-temperature \
  --non-interactive
```

選ばなかった server は `.mcp.json` に入りません。既存 config を黙って
上書きすることもありません。環境変数、dry-run、backup、Windows command、
troubleshooting は [setup guide](./docs/setup.md) にまとめています。

## Platform 対応

| Capability | Linux | macOS（Apple Silicon） | WSL2 | Windows ネイティブ |
|---|---|---|---|---|
| Core runtime と Claude Code hooks | 対応 | 対応 | 対応 | 対応 |
| Tapo network camera | 対応 | 対応 | 対応 | 対応 |
| USB camera | 対応 | 対応 | USB forwarding が必要 | OpenCV 対応 device |
| local microphone | PulseAudio/PipeWire | AVFoundation | WSLg/PulseAudio | DirectShow |
| TTS playback | `mpv` または `ffplay` | `mpv` または `ffplay` | WSLg/PulseAudio | `mpv` または `ffplay` |
| temperature sensor | `/sys`/hwmon | 利用可能な system sensor | host sensor は通常取得不可 | LibreHardwareMonitor bridge |

Hardware 対応は driver と device に依存します。setup は config を生成しますが、
vendor application、ffmpeg、VOICEVOX、hardware driver までは install しません。

## 全体の流れ

![Embodied Claude architecture](./docs/architecture.svg)

```text
user prompt / heartbeat / tool result
                  |
           Claude Code hooks
                  |
      begin -> compete -> commit field
                  |
   memory + needs + sociality + self model
                  |
       one intention -> action gate
                  |
       outcome -> mismatch -> next field
```

Enacted First-Person Field（EFPF）runtime は、owner ごとにひとつの typed な
自己―世界状態を commit し、memory 選択、attention/precision、prediction、
interaction planning、action gating の upstream input として使います。tool result
から prediction error と暫定的 agency を更新し、次の field を commit します。
`live`、`inferred`、`remembered`、`imagined`、`mixed` の source mode は agent
から見える状態に残ります。

これは phenomenal-consciousness candidate architecture、または
phenomenal-like causal architecture です。検査可能な因果条件を実装しますが、
現象意識を証明するものではありません。一人称 report だけを証拠として扱いません。

詳しくは次を参照してください。

- [Consciousness architecture](./consciousness-mcp/README.md)
- [Individual kernel runtime](./consciousness-mcp/packages/individual-kernel-mcp/README.md)
- [Field integrity benchmarks](./benchmarks/phenomenal_candidate/README.md)
- [Sociality v0.3 interaction loop](./docs/sociality.md)
- [Sociality package](./sociality-mcp/README.md)

## Repository 構成

| Path | 役割 |
|---|---|
| [`memory-mcp/`](./memory-mcp/) | 長期記憶、想起、連想、統合 |
| [`desire-system/`](./desire-system/) | bound された homeostatic needs と自律 trigger |
| [`sociality-mcp/`](./sociality-mcp/) | social context、関係、boundary、narrative の統合 facade |
| [`consciousness-mcp/`](./consciousness-mcp/) | EFPF workspace、field、agency、attention、HOR、quality geometry |
| [`usb-webcam-mcp/`](./usb-webcam-mcp/) | local USB camera capture |
| [`wifi-cam-mcp/`](./wifi-cam-mcp/) | Tapo PTZ、camera audio、local microphone |
| [`tts-mcp/`](./tts-mcp/) | VOICEVOX と ElevenLabs の統合音声 |
| [`system-temperature-mcp/`](./system-temperature-mcp/) | 時刻、resource、temperature signal |
| [`x-mcp/`](./x-mcp/) | X 検索、投稿、返信、削除 |
| [`.claude/`](./.claude/) | EFPF lifecycle の自動 hooks |
| [`scripts/`](./scripts/) | guided setup、doctor、seed、maintenance |

Python package はすべてひとつの uv workspace に属します。sync は repository
root から一度だけ実行します。

```bash
uv sync --locked
```

package を直接動かす場合:

```bash
uv run --package memory-mcp memory-mcp
uv run --package individual-kernel-mcp individual-kernel-mcp
```

## Configuration の安全性

- `.mcp.json` が project-local の credential source で、Git から無視されます。
- guided setup は atomic に書き、POSIX では mode `0600` にします。
- 内容が違う既存 config は、明示的な `--force` なしでは置換しません。
- 強制置換前に `.mcp.json.backup-<timestamp>` を作ります。
- `socialPolicy.toml` は存在しない時だけ example から作ります。
- `--dry-run` は sync、download、directory 作成、file write を行いません。
- [`.mcp.json.example`](./.mcp.json.example) は portable な Core 構成です。
  guided setup が選択した capability と環境変数だけを安全に追加します。

## 開発

CI の正本は [`.github/workflows/ci.yml`](./.github/workflows/ci.yml) です。
workspace の基本確認:

```bash
uv sync --locked
uv lock --check
uv run pytest tests -q
```

package の test と lint も同じ root environment から実行します。subsystem を
変更する前に [`CLAUDE.md`](./CLAUDE.md) と package-level `AGENTS.md` を確認してください。

## Autonomous runtime と strict runtime

Claude Code hooks は session、prompt、tool、batch、stop event の前後で field を
自動生成・更新します。outward MCP action には、speech、social post、camera
movement、その他 side effect を含む gate がかかります。

bare interactive Claude Code の chat text は、完全な external pre-display gate
より先に stream されます。text output まで strict に gate する研究実験では、
[individual kernel README](./consciousness-mcp/packages/individual-kernel-mcp/README.md)
にある non-interactive wrapper/runtime を使ってください。通常の interactive
利用では、表示 chat text は compatibility mode です。

autonomous heartbeat は任意です。有効にする前に
[`autonomous-action.sample.sh`](./autonomous-action.sample.sh) と privacy/boundary
policy を確認してください。

## Privacy と welfare

camera と microphone は他人を記録し得ます。consent を取り、boundary policy を
保ち、観察が不適切な場所では autonomous capture を無効にしてください。
field runtime は持続的な negative valence を bound し、pause/resume と reversible
ablation を提供し、high-frequency spawning を既定で無効にしています。

mechanism indicator と self-report の両方に不確実性があります。technical docs は
phenomenal claim policy に従い、この architecture を意識の証明とは記述しません。

## 関連 project

[familiar-ai](https://github.com/lifemate-ai/familiar-ai) は、これらの embodied
service の上に構築された上位 companion framework です。

## ライセンス

MIT License

## 謝辞

この project は安価な camera から始まり、memory、embodiment、agency、
人間と AI の関係を探る実験へ育ちました。

- [Rumia-Channel](https://github.com/Rumia-Channel) による ONVIF contribution
  ([#5](https://github.com/lifemate-ai/embodied-claude/pull/5))
- [fruitriin](https://github.com/fruitriin) による interoception への曜日 context 追加
  ([#14](https://github.com/lifemate-ai/embodied-claude/pull/14))
- [sugyan](https://github.com/sugyan) の
  [claude-code-webui](https://github.com/sugyan/claude-code-webui)
