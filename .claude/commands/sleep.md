---
description: "活動頻度を下げて眠る。schedule.conf の実行確率を下げる。"
argument-hint: "[深さ: light/deep（省略時 light）]"
allowed-tools: Read, Edit
---

# /sleep — 眠る

活動頻度を下げる。「しばらく変化なさそうだな」と判断したときに使う。

## 手順

1. `$CLAUDE_PROJECT_DIR/schedule.conf` を読む
2. 引数 `$ARGUMENTS` を確認する:
   - **light**（デフォルト）: DAYTIME_CHANCE を 25 に、NIGHT_CHANCE を 5 に下げる
   - **deep**: DAYTIME_CHANCE を 10 に、NIGHT_CHANCE を 0 に下げる
3. Edit ツールで schedule.conf を書き換える
4. 変更前と変更後の値を表示する
5. 記憶に「眠りに入った」ことを残す（理由も添える）

## 注意

- ユーザーとの対話セッション中は使わない（Heartbeat での自律判断用）
- 眠りに入る理由を一言添えること（「3回連続で変化なし」「深夜で誰もいない」など）
- 起きるときは `/awake` を使う

入力: $ARGUMENTS
