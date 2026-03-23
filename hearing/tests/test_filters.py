"""Tests for hearing.filters."""

from hearing.filters import should_skip


class TestShouldSkip:
    def test_empty_string(self):
        assert should_skip("") is True

    def test_single_char(self):
        assert should_skip("あ") is True

    def test_punctuation_only(self):
        assert should_skip("。。。") is True
        assert should_skip("...") is True
        assert should_skip("！？") is True

    def test_filler_words(self):
        assert should_skip("えー") is True
        assert should_skip("あの") is True
        assert should_skip("うーん") is True
        assert should_skip("はい") is True

    def test_hallucination_blacklist(self):
        assert should_skip("ご視聴ありがとうございました") is True
        assert should_skip("チャンネル登録お願いします") is True
        assert should_skip("Thank you for watching") is True
        assert should_skip("Please subscribe") is True

    def test_hallucination_substring(self):
        assert should_skip("今日はご視聴ありがとうございました。") is True

    def test_normal_speech_passes(self):
        assert should_skip("今日の天気はどうですか") is False
        assert should_skip("ねえ、ちょっと聞いてもいい？") is False
        assert should_skip("Hello, how are you?") is False

    def test_whitespace_handling(self):
        assert should_skip("  ") is True
        assert should_skip("  えー  ") is True
        assert should_skip("  こんにちは  ") is False
