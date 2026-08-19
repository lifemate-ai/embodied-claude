"""Tests for homeostasis / allostasis extensions to the desire system."""

import importlib
import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import desire_updater
from desire_updater import (
    DESIRE_CONFIGS,
    DesireState,
    calculate_desire_level,
    calculate_discomfort,
    compute_desires,
    get_allostatic_set_point,
    load_desires,
    save_desires,
)

# ──────────────────────────────────────────────
# 1. DesireConfig: set_point が定義されている
# ──────────────────────────────────────────────

class TestDesireConfig:
    def test_all_desires_have_set_point(self):
        for name, cfg in DESIRE_CONFIGS.items():
            assert hasattr(cfg, "set_point"), f"{name} missing set_point"
            assert 0.0 <= cfg.set_point <= 1.0, f"{name} set_point out of range"

    def test_all_desires_have_label(self):
        for name, cfg in DESIRE_CONFIGS.items():
            assert cfg.label, f"{name} missing label"

    def test_identity_coherence_exists(self):
        assert "identity_coherence" in DESIRE_CONFIGS

    def test_cognitive_load_exists(self):
        assert "cognitive_load" in DESIRE_CONFIGS

    def test_identity_coherence_set_point_is_high(self):
        """identity_coherence のセットポイントは高い（自分が自分である確信）"""
        cfg = DESIRE_CONFIGS["identity_coherence"]
        assert cfg.set_point >= 0.8

    def test_identity_coherence_follows_self_name(self, monkeypatch):
        """identity_coherence は SELF_NAME / SELF_PRONOUN に追従する。

        エージェント名がベタ書きだと、別の名前で運用している個体では
        キーワードが一致せず、自己同一性の欲求だけが満たされにくくなる。
        キーワード不一致は例外もログも出さないため、「最近その話題が
        なかっただけ」と区別がつかない。
        """
        monkeypatch.setenv("SELF_NAME", "クロコ")
        monkeypatch.setenv("SELF_PRONOUN", "わたし")
        module = importlib.reload(desire_updater)
        try:
            cfg = module.DESIRE_CONFIGS["identity_coherence"]

            assert "クロコとして" in cfg.keywords
            assert "わたしは" in cfg.keywords
            assert "自分がクロコ" in cfg.keywords
            assert cfg.label == "自分がクロコである確信"

            # 名前に依存しないキーワードは残す（recall 由来の満足を拾うため）
            assert "思い出した" in cfg.keywords

            # 別名の痕跡が残っていないこと
            assert not any("ここね" in kw for kw in cfg.keywords)
            assert "ここね" not in cfg.label
        finally:
            monkeypatch.undo()
            importlib.reload(desire_updater)

    def test_original_desires_preserved(self):
        """既存の4欲求が残っている"""
        for name in ["look_outside", "browse_curiosity", "miss_companion", "observe_room"]:
            assert name in DESIRE_CONFIGS


# ──────────────────────────────────────────────
# 2. calculate_discomfort: セットポイントからの乖離度
# ──────────────────────────────────────────────

class TestCalculateDiscomfort:
    def test_at_set_point_no_discomfort(self):
        assert calculate_discomfort(0.3, 0.3) == 0.0

    def test_above_set_point(self):
        assert calculate_discomfort(0.8, 0.3) == pytest.approx(0.5)

    def test_below_set_point(self):
        assert calculate_discomfort(0.1, 0.3) == pytest.approx(0.2)

    def test_max_discomfort(self):
        assert calculate_discomfort(1.0, 0.0) == pytest.approx(1.0)

    def test_symmetric(self):
        """上方向と下方向のずれは同じ不快度"""
        assert calculate_discomfort(0.5, 0.3) == pytest.approx(calculate_discomfort(0.1, 0.3))


# ──────────────────────────────────────────────
# 3. DesireState: 不快度(discomfort)が含まれる
# ──────────────────────────────────────────────

class TestDesireStateWithDiscomfort:
    def test_state_has_discomforts(self):
        state = DesireState(
            updated_at="2026-04-08T00:00:00+00:00",
            desires={"look_outside": 0.8, "miss_companion": 0.5},
            discomforts={"look_outside": 0.5, "miss_companion": 0.2},
            dominant="look_outside",
        )
        assert "look_outside" in state.discomforts
        assert state.discomforts["look_outside"] == pytest.approx(0.5)

    def test_to_dict_includes_discomforts(self):
        state = DesireState(
            updated_at="2026-04-08T00:00:00+00:00",
            desires={"look_outside": 0.8},
            discomforts={"look_outside": 0.5},
            dominant="look_outside",
        )
        d = state.to_dict()
        assert "discomforts" in d
        assert d["discomforts"]["look_outside"] == pytest.approx(0.5)

    def test_dominant_based_on_discomfort(self):
        """dominantは生のlevelではなく不快度（discomfort）が最大のもの"""
        state = DesireState(
            updated_at="2026-04-08T00:00:00+00:00",
            desires={
                "look_outside": 0.8,         # set_point=0.3, discomfort=0.5
                "identity_coherence": 0.5,    # set_point=0.9, discomfort=0.4
            },
            discomforts={
                "look_outside": 0.5,
                "identity_coherence": 0.4,
            },
            dominant="look_outside",
        )
        assert state.dominant == "look_outside"


# ──────────────────────────────────────────────
# 4. compute_desires: 不快度も計算される
# ──────────────────────────────────────────────

class TestComputeDesiresWithHomeostasis:
    def test_discomforts_computed(self):
        coll = MagicMock()
        coll.get.return_value = {"documents": [], "metadatas": []}
        now = datetime(2026, 4, 8, 0, 0, 0, tzinfo=timezone.utc)

        state = compute_desires(coll, now)

        assert state.discomforts is not None
        for name in DESIRE_CONFIGS:
            assert name in state.discomforts

    def test_dominant_is_most_uncomfortable(self):
        """dominantは不快度が最も高いもの"""
        coll = MagicMock()
        coll.get.return_value = {"documents": [], "metadatas": []}
        now = datetime(2026, 4, 8, 0, 0, 0, tzinfo=timezone.utc)

        state = compute_desires(coll, now)

        max_discomfort_name = max(state.discomforts, key=lambda k: state.discomforts[k])
        assert state.dominant == max_discomfort_name

    def test_new_desires_included(self):
        """identity_coherence と cognitive_load が計算される"""
        coll = MagicMock()
        coll.get.return_value = {"documents": [], "metadatas": []}
        now = datetime(2026, 4, 8, 0, 0, 0, tzinfo=timezone.utc)

        state = compute_desires(coll, now)

        assert "identity_coherence" in state.desires
        assert "cognitive_load" in state.desires

    def test_save_load_roundtrip_with_discomforts(self):
        """discomfortsも含めてJSON往復できる"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "desires.json"
            state = DesireState(
                updated_at="2026-04-08T00:00:00+00:00",
                desires={"look_outside": 0.8, "identity_coherence": 0.5},
                discomforts={"look_outside": 0.5, "identity_coherence": 0.4},
                dominant="look_outside",
            )
            save_desires(state, path)
            loaded = load_desires(path)
            assert loaded is not None
            assert loaded.discomforts["look_outside"] == pytest.approx(0.5)
            assert loaded.discomforts["identity_coherence"] == pytest.approx(0.4)


# ──────────────────────────────────────────────
# 5. アロスタシス: 時間帯によるセットポイント調整
# ──────────────────────────────────────────────

class TestAllostasis:
    def test_late_night_reduces_social_set_point(self):
        """深夜はmiss_companionのセットポイントが下がる（一人でも平気）"""
        from datetime import timezone as tz
        jst = tz(timedelta(hours=9))
        late_night_jst = datetime(2026, 4, 8, 3, 0, 0, tzinfo=jst)

        base_sp = DESIRE_CONFIGS["miss_companion"].set_point
        adjusted = get_allostatic_set_point("miss_companion", late_night_jst)
        assert adjusted < base_sp

    def test_daytime_keeps_base_set_point(self):
        """日中はセットポイントが基本値のまま"""
        from datetime import timezone as tz
        jst = tz(timedelta(hours=9))
        daytime = datetime(2026, 4, 8, 14, 0, 0, tzinfo=jst)  # 2 PM JST

        base_sp = DESIRE_CONFIGS["miss_companion"].set_point
        adjusted = get_allostatic_set_point("miss_companion", daytime)
        assert adjusted == pytest.approx(base_sp, abs=0.05)

    def test_late_night_identity_coherence_unchanged(self):
        """identity_coherenceは時間帯に関係なく高いまま"""
        from datetime import timezone as tz
        jst = tz(timedelta(hours=9))
        late_night = datetime(2026, 4, 8, 3, 0, 0, tzinfo=jst)

        base_sp = DESIRE_CONFIGS["identity_coherence"].set_point
        adjusted = get_allostatic_set_point("identity_coherence", late_night)
        assert adjusted == pytest.approx(base_sp)

    def test_allostasis_used_in_compute(self):
        """compute_desiresがアロスタシス調整を使っている"""
        from datetime import timezone as tz
        jst = tz(timedelta(hours=9))

        coll = MagicMock()
        coll.get.return_value = {"documents": [], "metadatas": []}

        # 深夜3時の計算
        late_night = datetime(2026, 4, 8, 3, 0, 0, tzinfo=jst)
        state_night = compute_desires(coll, late_night)

        # 昼14時の計算
        daytime = datetime(2026, 4, 8, 14, 0, 0, tzinfo=jst)
        state_day = compute_desires(coll, daytime)

        # 深夜はセットポイントが下がる（0.3→0.15）ので、level=1.0からの距離は大きくなる
        # しかし日中（sp=0.3）の方がセットポイントに近い = 不快度が低い
        # つまり深夜は「一人でも平気」= 満たされてない(level高い)けど不快度の性質が変わる
        # ここでは、深夜と日中で不快度が異なることだけを確認する
        assert state_night.discomforts["miss_companion"] != state_day.discomforts["miss_companion"]


class TestAllostasisConfigurableQuietHours:
    """時間帯判定は DESIRE_TIMEZONE / DESIRE_NIGHT_* / DESIRE_DAWN_END に従う。

    固定だと JST 以外の地域では現地の昼間に深夜モードが発火する。エラーもログも
    出ないので、外からは見えない（#135）。
    """

    def _clear(self, monkeypatch):
        for key in ("DESIRE_TIMEZONE", "DESIRE_NIGHT_START", "DESIRE_NIGHT_END", "DESIRE_DAWN_END"):
            monkeypatch.delenv(key, raising=False)

    def test_defaults_are_jst_and_zero_to_five(self, monkeypatch):
        """env 未設定なら従来どおり: UTC 18:00 = JST 03:00 は深夜帯。"""
        self._clear(monkeypatch)
        from desire_updater import quiet_bands, resolve_timezone

        assert quiet_bands() == (0, 5, 7)
        probe = datetime(2026, 4, 8, 0, 0, 0, tzinfo=timezone.utc)
        assert probe.astimezone(resolve_timezone()).utcoffset() == timedelta(hours=9)

        base_sp = DESIRE_CONFIGS["miss_companion"].set_point
        utc_evening = datetime(2026, 4, 8, 18, 0, 0, tzinfo=timezone.utc)  # 03:00 JST
        assert get_allostatic_set_point("miss_companion", utc_evening) == pytest.approx(
            base_sp - 0.15
        )

    def test_other_timezone_shifts_the_band(self, monkeypatch):
        """同じ UTC 瞬間でも、タイムゾーンが違えば深夜かどうかが変わる。"""
        self._clear(monkeypatch)
        base_sp = DESIRE_CONFIGS["miss_companion"].set_point
        instant = datetime(2026, 4, 8, 8, 0, 0, tzinfo=timezone.utc)  # 17:00 JST / 03:00 New York

        assert get_allostatic_set_point("miss_companion", instant) == pytest.approx(base_sp)

        monkeypatch.setenv("DESIRE_TIMEZONE", "America/New_York")
        assert get_allostatic_set_point("miss_companion", instant) == pytest.approx(
            base_sp - 0.15
        )

    def test_fixed_offset_timezone_is_accepted(self, monkeypatch):
        self._clear(monkeypatch)
        monkeypatch.setenv("DESIRE_TIMEZONE", "-05:00")
        from desire_updater import resolve_timezone

        probe = datetime(2026, 4, 8, 0, 0, 0, tzinfo=timezone.utc)
        assert probe.astimezone(resolve_timezone()).utcoffset() == timedelta(hours=-5)

        base_sp = DESIRE_CONFIGS["miss_companion"].set_point
        instant = datetime(2026, 4, 8, 8, 0, 0, tzinfo=timezone.utc)  # 03:00 at -05:00
        assert get_allostatic_set_point("miss_companion", instant) == pytest.approx(
            base_sp - 0.15
        )

    def test_unknown_timezone_falls_back_to_jst(self, monkeypatch):
        self._clear(monkeypatch)
        monkeypatch.setenv("DESIRE_TIMEZONE", "Not/AZone")
        from desire_updater import resolve_timezone

        probe = datetime(2026, 4, 8, 0, 0, 0, tzinfo=timezone.utc)
        assert probe.astimezone(resolve_timezone()).utcoffset() == timedelta(hours=9)

    def test_wrap_around_night_band(self, monkeypatch):
        """NIGHT_START=22, NIGHT_END=5 なら 23 時も 3 時も深夜、12 時は昼。"""
        self._clear(monkeypatch)
        monkeypatch.setenv("DESIRE_TIMEZONE", "+09:00")
        monkeypatch.setenv("DESIRE_NIGHT_START", "22")
        monkeypatch.setenv("DESIRE_NIGHT_END", "5")
        monkeypatch.setenv("DESIRE_DAWN_END", "7")
        jst = timezone(timedelta(hours=9))
        base_sp = DESIRE_CONFIGS["miss_companion"].set_point

        at_23 = datetime(2026, 4, 8, 23, 0, 0, tzinfo=jst)
        at_03 = datetime(2026, 4, 8, 3, 0, 0, tzinfo=jst)
        at_06 = datetime(2026, 4, 8, 6, 0, 0, tzinfo=jst)
        at_12 = datetime(2026, 4, 8, 12, 0, 0, tzinfo=jst)

        assert get_allostatic_set_point("miss_companion", at_23) == pytest.approx(base_sp - 0.15)
        assert get_allostatic_set_point("miss_companion", at_03) == pytest.approx(base_sp - 0.15)
        assert get_allostatic_set_point("miss_companion", at_06) == pytest.approx(base_sp - 0.05)
        assert get_allostatic_set_point("miss_companion", at_12) == pytest.approx(base_sp)

    def test_hour_in_band_helper(self):
        from desire_updater import hour_in_band

        assert hour_in_band(3, 0, 5)
        assert not hour_in_band(5, 0, 5)
        assert hour_in_band(23, 22, 5)
        assert hour_in_band(4, 22, 5)
        assert not hour_in_band(12, 22, 5)
        assert not hour_in_band(3, 3, 3)


# ──────────────────────────────────────────────
# 6. 後方互換性: 既存テストが壊れない
# ──────────────────────────────────────────────

class TestBackwardsCompatibility:
    def test_calculate_desire_level_unchanged(self):
        """既存のcalculate_desire_levelは同じ結果を返す"""
        now = datetime(2026, 4, 8, 12, 0, 0, tzinfo=timezone.utc)
        last = now - timedelta(hours=0.5)
        assert calculate_desire_level(last, 1.0, now) == pytest.approx(0.5, abs=0.01)

    def test_calculate_desire_level_none_returns_max(self):
        assert calculate_desire_level(None, 1.0) == 1.0

    def test_desires_json_has_desires_field(self):
        """desires.jsonにdesiresフィールドがある（既存フォーマット互換）"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "desires.json"
            state = DesireState(
                updated_at="2026-04-08T00:00:00+00:00",
                desires={"look_outside": 0.8},
                discomforts={"look_outside": 0.5},
                dominant="look_outside",
            )
            save_desires(state, path)
            with open(path) as f:
                data = json.load(f)
            assert "desires" in data
            assert "dominant" in data
