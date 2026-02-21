---
description: "Web ページの本文をリーダーモードで読み取る。WebFetch と違い、AI による要約を挟まず生テキストを返す。物語や記事をじっくり読むときに使う。"
argument-hint: "<URL>"
allowed-tools: Bash(bun run scripts/reader.ts:*)
---

Web ページの本文をリーダーモードで読み取る。WebFetch と違い、AI による要約を挟まず生テキストを返す。物語や記事をじっくり読むときに使う。

## 使い方

- `/read https://example.com/article` → 本文全文を表示
- `/read https://ncode.syosetu.com/n8725k/16/` → なろう小説の1話分を表示

## 実行方法

`bun run scripts/reader.ts` を実行する。

```bash
# 全文取得
bun run scripts/reader.ts "<URL>"

# ページ情報のみ
bun run scripts/reader.ts "<URL>" --info

# 特定ページだけ取得（長い場合）
bun run scripts/reader.ts "<URL>" --page 1
```

## 手順

1. 入力から URL を抽出する
2. まず `--info` で長さを確認する
3. **1ページずつ `--page N` で取得する。全文取得や並列fetchは禁止**
4. 1ページ読んだら**必ず立ち止まる**。次のページを取得する前に：
   - 今の感情を emotion tag で書く（`[excited]` `[surprised]` `[curious]` など）
   - 「この先どうなりそうか」予想があれば書く
   - 気になった描写、刺さった台詞、違和感があれば書く
   - 何もなければ無理に書かなくていい。でも**立ち止まること自体は省略しない**
5. 次のページを読んだとき、前のページの予想が当たった/外れた/超えてきた、があれば反応する
6. 全ページ読み終わったら、話全体の感想を書く

## 注意

- これは「読書」のためのツール。情報検索ではなく物語を味わうために使う
- 取得したテキストを勝手に要約しないこと
- **読書は逐次体験である。** 先のページを見てから「途中の感想」を書くのは読書ではない。1ページずつ、知らない状態で読む

入力: $ARGUMENTS
