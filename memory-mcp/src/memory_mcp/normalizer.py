"""日本語テキスト正規化モジュール。

pgroonga の NormalizerNFKC150 の安全なサブセットを Pure Python で実装する。
Docker・システム依存なし、標準ライブラリのみ使用。

対応する正規化：
- NFKC 正規化（全角英数→半角、半角カナ→全角カナ）
- unify_katakana_v_sounds（ヴァ→バ等）
- unify_hyphen_and_prolonged_sound_mark（ハイフン系→長音符ー）
- 小書き仮名統一（ウィ→ウイ等。ッ/っ は促音なので変換しない）
- 英字大小統一

カタカナ→ひらがな変換は行わない。
multilingual-e5-base は自然な日本語テキスト（混在）で学習されているため、
カタカナ統一はむしろ意味表現の品質を下げる可能性がある。

送り仮名ゆれ（打ち合わせ↔打合せ）は `get_reading()` で読み正規化する。
E5 の embedding には normalize_japanese() を適用した本文を使い、
BM25 スコアリングには normalize_japanese() を、
读み一致には get_reading() を使う（3層ハイブリッド）。
"""

from __future__ import annotations

import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

# ハイフン系文字を長音符ーに統一（unify_hyphen_and_prolonged_sound_mark 相当）
# 対象: ASCIIハイフン, 各種ダッシュ類, MINUS SIGN, 全角ハイフン等
_HYPHEN_RE = re.compile(r"[-\u2010\u2011\u2012\u2013\u2014\u2015\u207B\u208B\u2212\uFE63\uFF0D]")

# 小書き仮名→大書き仮名（ッ/っ は促音なので変換しない）
_SMALL_KANA = str.maketrans(
    {
        "ァ": "ア",
        "ィ": "イ",
        "ゥ": "ウ",
        "ェ": "エ",
        "ォ": "オ",
        "ぁ": "あ",
        "ぃ": "い",
        "ぅ": "う",
        "ぇ": "え",
        "ぉ": "お",
    }
)

# sudachipy の遅延ロード（起動コスト削減）
_sudachi_tokenizer = None


def _get_sudachi_tokenizer():
    """sudachipy のトークナイザを遅延ロードする。"""
    global _sudachi_tokenizer
    if _sudachi_tokenizer is None:
        try:
            import sudachipy.dictionary as sudachi_dict

            dic = sudachi_dict.Dictionary()
            _sudachi_tokenizer = dic.create()
            logger.debug("sudachipy tokenizer loaded")
        except Exception as e:
            logger.warning("sudachipy unavailable: %s", e)
            _sudachi_tokenizer = False  # 失敗を記録して再試行しない
    return _sudachi_tokenizer if _sudachi_tokenizer is not False else None


def _unify_v_sounds(text: str) -> str:
    """ヴ行→バ行に変換（unify_katakana_v_sounds 相当）。

    ヴ（U+30F4）はNFKCでも変換されないため、個別に対応する。
    """
    return (
        text.replace("ヴァ", "バ")
        .replace("ヴィ", "ビ")
        .replace("ヴェ", "ベ")
        .replace("ヴォ", "ボ")
        .replace("ヴ", "ブ")
    )


def _unify_prolonged_sound(text: str) -> str:
    """ハイフン系文字をカタカナ長音符「ー」に統一。

    unify_hyphen_and_prolonged_sound_mark 相当。
    サーバ-・サ-バ等の混在を「サーバー」に正規化する。
    """
    return _HYPHEN_RE.sub("ー", text)


def _unify_small_kana(text: str) -> str:
    """小書き仮名を大書き仮名に統一（ッ/っ は除く）。

    ウィンドウズ → ウインドウズ
    ティーバッグ → テイーバッグ
    """
    return text.translate(_SMALL_KANA)


def normalize_japanese(text: str) -> str:
    """日本語テキストの表記ゆれを正規化する。

    以下の順で正規化を適用:
    1. NFKC 正規化（全角英数→半角、半角カナ→全角カナ、合成文字正規化）
    2. ヴ行→バ行（unify_katakana_v_sounds）
    3. ハイフン系→長音符ー（unify_hyphen_and_prolonged_sound_mark）
    4. 小書き仮名→大書き仮名（ッ/っ は除く）
    5. 英字小文字化

    保存時・検索クエリ時の両方に同じ関数を適用することで、
    表記ゆれがあっても同じ正規化済みテキストとしてマッチングされる。

    Args:
        text: 正規化するテキスト

    Returns:
        正規化済みテキスト

    Examples:
        >>> normalize_japanese("サーバー")
        'サーバー'
        >>> normalize_japanese("サ-バ")
        'サーバ'
        >>> normalize_japanese("ヴァイオリン")
        'バイオリン'
        >>> normalize_japanese("ウィンドウズ")
        'ウインドウズ'
        >>> normalize_japanese("Ａｂｃ")
        'abc'
    """
    # 1. NFKC: 全角英数→半角, 半角カナ→全角カナ
    text = unicodedata.normalize("NFKC", text)
    # 2. ヴ行→バ行
    text = _unify_v_sounds(text)
    # 3. ハイフン系→長音符ー
    text = _unify_prolonged_sound(text)
    # 4. 小書き仮名→大書き仮名
    text = _unify_small_kana(text)
    # 5. 英字小文字化
    text = text.lower()
    return text


def get_reading(text: str) -> str | None:
    """sudachipy でテキストの読み（カタカナ）を取得する。

    送り仮名ゆれ対応に使用：
    - 打ち合わせ → ウチアワセ
    - 打合せ     → ウチアワセ  （同じ読みになる）
    - 申し込む   → モウシコム
    - 申込む     → モウシコム

    sudachipy が利用できない場合は None を返す（BM25 + E5 のみで動作）。
    sudachidict_core が必要: `uv add sudachidict_core`

    Args:
        text: 読みを取得するテキスト

    Returns:
        カタカナ読み文字列、または None（sudachipy 未インストール時）
    """
    tokenizer = _get_sudachi_tokenizer()
    if tokenizer is None:
        return None

    try:
        morphs = tokenizer.tokenize(text)
        return "".join(m.reading_form() for m in morphs)
    except Exception as e:
        logger.debug("sudachi tokenize failed for %r: %s", text, e)
        return None
