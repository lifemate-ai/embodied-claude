"""Tests for BM25 index and tokenizer (Phase 9)."""

from memory_mcp.bm25 import BM25Index, tokenize


class TestTokenize:
    """tokenize() の動作テスト."""

    def test_english_words(self) -> None:
        """英単語は小文字化されて分割される。"""
        tokens = tokenize("Hello World")
        assert "hello" in tokens
        assert "world" in tokens

    def test_alphanumeric(self) -> None:
        """英数字混在も分割される。"""
        tokens = tokenize("Python3 GPT4")
        assert "python3" in tokens
        assert "gpt4" in tokens

    def test_japanese_bigram(self) -> None:
        """日本語は bigram で分割される。"""
        tokens = tokenize("打ち合わせ")
        assert "打ち" in tokens
        assert "ち合" in tokens

    def test_mixed_jp_en(self) -> None:
        """日英混在テキストも分割される。"""
        tokens = tokenize("サーバーserver")
        assert "サー" in tokens
        assert "server" in tokens

    def test_empty_string(self) -> None:
        """空文字列はトークンなし。"""
        assert tokenize("") == []

    def test_hiragana_bigram(self) -> None:
        """ひらがなも bigram になる。"""
        tokens = tokenize("うちあわせ")
        assert "うち" in tokens
        assert "ちあ" in tokens


class TestBM25Index:
    """BM25Index の動作テスト."""

    def test_build_and_score(self) -> None:
        """ビルド後にスコアが取得できる。"""
        index = BM25Index()
        memories = [
            ("id1", "打ち合わせの内容を記録した"),
            ("id2", "今日の天気は晴れです"),
        ]
        index.build(memories)

        scores = index.scores("打ち合わせ", ["id1", "id2"])
        # 打ち合わせ bigram が含まれる id1 のスコアが高い
        assert scores["id1"] > scores["id2"]

    def test_dirty_flag_on_init(self) -> None:
        """初期状態は dirty。"""
        index = BM25Index()
        assert index.is_dirty is True

    def test_dirty_cleared_after_build(self) -> None:
        """build() 後は dirty が解除される。"""
        index = BM25Index()
        index.build([("id1", "test content")])
        assert index.is_dirty is False

    def test_mark_dirty(self) -> None:
        """mark_dirty() で dirty フラグが立つ。"""
        index = BM25Index()
        index.build([("id1", "content")])
        index.mark_dirty()
        assert index.is_dirty is True

    def test_score_normalized_0_to_1(self) -> None:
        """スコアは 0〜1 に正規化される。"""
        index = BM25Index()
        index.build([
            ("id1", "Python programming language"),
            ("id2", "日本語テキスト処理"),
        ])
        scores = index.scores("Python", ["id1", "id2"])
        for score in scores.values():
            assert 0.0 <= score <= 1.0

    def test_empty_index_returns_empty(self) -> None:
        """記憶が0件の場合はスコアなし。"""
        index = BM25Index()
        index.build([])
        scores = index.scores("query", ["id1"])
        assert scores == {}

    def test_unknown_doc_id_score_zero(self) -> None:
        """存在しない doc_id のスコアは 0。"""
        index = BM25Index()
        index.build([("id1", "some content")])
        scores = index.scores("content", ["id1", "unknown_id"])
        assert scores["unknown_id"] == 0.0

    def test_empty_query_returns_zeros(self) -> None:
        """空クエリは全スコア 0。"""
        index = BM25Index()
        index.build([("id1", "some content")])
        scores = index.scores("", ["id1"])
        assert scores["id1"] == 0.0

    def test_exact_match_higher_score(self) -> None:
        """完全一致する単語のスコアが高い。"""
        index = BM25Index()
        index.build([
            ("id1", "machine learning deep learning"),
            ("id2", "cooking recipe ingredients"),
        ])
        scores = index.scores("machine learning", ["id1", "id2"])
        assert scores["id1"] > scores["id2"]
