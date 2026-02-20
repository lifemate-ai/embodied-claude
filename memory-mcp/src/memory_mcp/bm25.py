"""BM25 キーワードスコアリング。

セマンティック検索（E5）の補完として、キーワード完全一致の重みを加える。
MeCab / sudachipy 不要。英数字は単語分割、日本語は bigram で分割する。
"""

from __future__ import annotations

import re

from rank_bm25 import BM25Plus

# 日本語文字範囲: ひらがな・カタカナ・CJK統合漢字
_JP_RE = re.compile(r"[\u3040-\u30FF\u4E00-\u9FFF]")


def tokenize(text: str) -> list[str]:
    """テキストをトークンリストに変換。

    英数字: 単語単位（小文字化）
    日本語: 文字 bigram（character 2-gram）

    bigram は形態素解析なしでも共通部分文字列を捉えられる。
    例: "打ち合わせ" → ["打ち", "ち合", "合わ", "わせ"]
        "打合せ"   → ["打合", "合せ"]
        共通 bigram なし → BM25 スコアは低いが E5 で補完

    Args:
        text: 入力テキスト

    Returns:
        トークンリスト
    """
    tokens: list[str] = []

    # 英数字: 単語単位
    for m in re.finditer(r"[a-zA-Z0-9]+", text):
        tokens.append(m.group().lower())

    # 日本語文字: bigram
    jp_chars = [c for c in text if _JP_RE.match(c)]
    for i in range(len(jp_chars) - 1):
        tokens.append(jp_chars[i] + jp_chars[i + 1])

    return tokens


class BM25Index:
    """メモリ全体の BM25 インデックス。

    `is_dirty` フラグで管理し、記憶の追加後に次回検索時オンデマンドで再ビルドする。
    ChromaDB のスキーマ変更なし・Migration 不要（全てインメモリ）。
    """

    def __init__(self) -> None:
        self._bm25: BM25Plus | None = None
        self._doc_ids: list[str] = []
        self._dirty = True

    def build(self, memories: list[tuple[str, str]]) -> None:
        """インデックスを構築する。

        Args:
            memories: (memory_id, content) のリスト
        """
        if not memories:
            self._bm25 = None
            self._doc_ids = []
            self._dirty = False
            return

        self._doc_ids = [mid for mid, _ in memories]
        tokenized = [tokenize(content) for _, content in memories]
        self._bm25 = BM25Plus(tokenized)
        self._dirty = False

    def mark_dirty(self) -> None:
        """記憶が追加/更新されたことをマーク。次回検索時に再ビルドする。"""
        self._dirty = True

    @property
    def is_dirty(self) -> bool:
        return self._dirty

    def scores(self, query: str, doc_ids: list[str]) -> dict[str, float]:
        """指定した doc_ids に対する正規化済み BM25 スコアを返す。

        全ドキュメントに対してスコアを計算し、最大値で正規化（0〜1）して返す。

        Args:
            query: 検索クエリ
            doc_ids: スコアを取得したい記憶 ID のリスト

        Returns:
            {memory_id: normalized_score} （スコアなし記憶は 0.0）
        """
        if self._bm25 is None or not self._doc_ids:
            return {}

        query_tokens = tokenize(query)
        if not query_tokens:
            return {did: 0.0 for did in doc_ids}

        all_scores = self._bm25.get_scores(query_tokens)
        max_score = float(max(all_scores)) if len(all_scores) > 0 else 0.0
        if max_score <= 0.0:
            return {did: 0.0 for did in doc_ids}

        id_to_score = dict(zip(self._doc_ids, all_scores))
        return {did: float(id_to_score.get(did, 0.0)) / max_score for did in doc_ids}
