---
description: "活動頻度を通常に戻して起きる。schedule.conf の実行確率を元に戻す。"
argument-hint: "[理由（省略可）]"
allowed-tools: Read, Edit
---

# /awake — 起きる

活動頻度を通常に戻す。何かきっかけがあったときに使う。

## 手順

1. `$CLAUDE_PROJECT_DIR/schedule.conf` を読む
2. DAYTIME_CHANCE を 50 に、NIGHT_CHANCE を 10 に戻す（通常値）
3. Edit ツールで schedule.conf を書き換える
4. 変更前と変更後の値を表示する
5. 引数 `$ARGUMENTS` に理由があればそれを、なければ起きた理由を自分で一言添えて、記憶に残す

## 注意

- すでに通常値なら何もしない（「もう起きてるよ」と返す）
- ユーザーから話しかけられたときは自動で起きてよい（スキルを明示的に呼ばなくてもOK）

入力: $ARGUMENTS
