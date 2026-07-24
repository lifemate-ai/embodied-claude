"""Deterministic causal ablations and field integrity indicators."""

from __future__ import annotations

import json
import re
import secrets
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from social_core.db import SocialDB
from social_core.time import utc_now

from individual_kernel_mcp.enacted_field import EnactedField, EnactedFieldStore


class DownstreamSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory_order: list[str] = Field(default_factory=list)
    action_order: list[str] = Field(default_factory=list)
    memory_scores: dict[str, float] = Field(default_factory=dict)
    action_scores: dict[str, float] = Field(default_factory=dict)
    self_model_precision: float = 0.0


class AblationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ablation_id: str
    kind: str
    source_field_id: str
    baseline: DownstreamSelection
    ablated: DownstreamSelection
    effect_size: dict[str, float]
    reversible_snapshot: dict[str, Any]
    created_at: str


class IndicatorProfile(BaseModel):
    """Mechanism indicators; deliberately not a probability."""

    model_config = ConfigDict(extra="forbid")

    profile_name: str = "FieldIntegrityReport"
    causal_centrality: float = Field(ge=0.0, le=1.0)
    recurrent_closure: float = Field(ge=0.0, le=1.0)
    sensorimotor_closure: float = Field(ge=0.0, le=1.0)
    self_model_feedback: float = Field(ge=0.0, le=1.0)
    valence_coupling: float = Field(ge=0.0, le=1.0)
    qualitative_transition_structure: float = Field(ge=0.0, le=1.0)
    report_independence: float = Field(ge=0.0, le=1.0)
    unity_violations: int = Field(ge=0)
    prediction_calibration: float = Field(ge=0.0, le=1.0)
    uncertainty_notes: list[str] = Field(default_factory=list)


class AblationRunner:
    def __init__(self, db: SocialDB, fields: EnactedFieldStore | None = None) -> None:
        self._db = db
        self._fields = fields or EnactedFieldStore(db)

    def run(
        self,
        kind: str,
        *,
        fixture: dict[str, Any] | None = None,
        owner_id: str = "self",
        seed: int | None = None,
    ) -> AblationResult:
        field = self._fields.get_current(owner_id)
        if field is None:
            raise ValueError("an ablation requires a committed source field")
        fixture_data = fixture or _default_fixture(field)
        baseline = downstream_selection(field, fixture_data)
        ablated_field = self._ablate(field, kind, fixture_data)
        ablated = downstream_selection(ablated_field, fixture_data)
        effect = _effect_size(baseline, ablated)
        ablation_id = f"abl_{secrets.token_urlsafe(12)}"
        now = utc_now()
        snapshot = field.model_dump(mode="json")
        self._db.execute(
            """
            INSERT INTO field_ablation_runs(
                ablation_id, owner_id, kind, fixture_ref, seed, source_field_id,
                baseline_json, ablated_json, effect_size_json,
                reversible_snapshot_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ablation_id,
                owner_id,
                kind,
                str(fixture_data.get("fixture_ref") or "inline"),
                seed,
                field.field_id,
                json.dumps(baseline.model_dump(mode="json"), sort_keys=True),
                json.dumps(ablated.model_dump(mode="json"), sort_keys=True),
                json.dumps(effect, sort_keys=True),
                json.dumps(snapshot, ensure_ascii=False, sort_keys=True),
                now,
            ),
        )
        return AblationResult(
            ablation_id=ablation_id,
            kind=kind,
            source_field_id=field.field_id,
            baseline=baseline,
            ablated=ablated,
            effect_size=effect,
            reversible_snapshot=snapshot,
            created_at=now,
        )

    @staticmethod
    def _ablate(
        field: EnactedField,
        kind: str,
        fixture: dict[str, Any],
    ) -> EnactedField:
        trace = dict(field.epistemic_trace)
        flags = list(trace.get("ablation_flags") or [])
        flags.append(kind)
        trace["ablation_flags"] = flags
        if kind in {"focal_clamp", "focus"}:
            focus = str(fixture.get("ablated_focus") or "clamped:unrelated")
            return field.model_copy(
                update={
                    "focal_content_ref": focus,
                    "focal_summary": str(
                        fixture.get("ablated_focus_summary") or focus
                    ),
                    "epistemic_trace": trace,
                }
            )
        if kind in {"hor_feedback", "hor"}:
            precision = field.precision.model_copy(
                update={"self_model": field.precision.self_model * 0.2}
            )
            return field.model_copy(
                update={"hor_refs": [], "precision": precision, "epistemic_trace": trace}
            )
        if kind in {"valence", "need"}:
            intero = field.interoception.model_copy(
                update={"valence": 0.0, "need_vector": {}}
            )
            return field.model_copy(update={"interoception": intero, "epistemic_trace": trace})
        if kind in {"reality", "source"}:
            return field.model_copy(
                update={
                    "reality_score": 0.0,
                    "reality_mode": "imagined",
                    "epistemic_trace": trace,
                }
            )
        raise ValueError(f"unsupported ablation kind {kind!r}")

    def indicator_profile(self, *, owner_id: str = "self", window: int = 50) -> IndicatorProfile:
        limit = max(1, min(window, 500))
        fields = self._fields.query(owner_id=owner_id, limit=limit)
        if not fields:
            return IndicatorProfile(
                causal_centrality=0.0,
                recurrent_closure=0.0,
                sensorimotor_closure=0.0,
                self_model_feedback=0.0,
                valence_coupling=0.0,
                qualitative_transition_structure=0.0,
                report_independence=0.0,
                unity_violations=0,
                prediction_calibration=0.0,
                uncertainty_notes=["no field records in the requested window"],
            )
        ablation_rows = self._db.fetchall(
            """
            SELECT effect_size_json FROM field_ablation_runs
            WHERE owner_id = ? ORDER BY created_at DESC LIMIT ?
            """,
            (owner_id, limit),
        )
        causal = _mean(
            [
                _mean(list(_load(row["effect_size_json"]).values()))
                for row in ablation_rows
            ]
        )
        recurrent = _mean(
            [
                1.0 if field.hor_refs and field.attention_schema_ref else 0.0
                for field in fields
            ]
        )
        intentions = self._scalar(
            "SELECT COUNT(*) FROM field_intentions WHERE owner_id = ?", (owner_id,)
        )
        outcomes = self._scalar(
            """
            SELECT COUNT(*) FROM field_action_outcomes o
            JOIN field_intentions i ON i.action_id = o.action_id
            WHERE i.owner_id = ?
            """,
            (owner_id,),
        )
        sensorimotor = outcomes / intentions if intentions else 0.0
        self_feedback = _mean([field.precision.self_model for field in fields])
        valence_coupling = _mean(
            [
                1.0
                if field.interoception.need_vector
                and any(ref.startswith("desire:") for ref in field.affordance_refs)
                else min(1.0, abs(field.interoception.valence))
                for field in fields
            ]
        )
        quality_count = self._scalar(
            """
            SELECT COUNT(*) FROM quality_signatures
            WHERE signature_id IN (
                SELECT quality_signature_ref FROM enacted_fields
                WHERE owner_id = ?
            )
            """,
            (owner_id,),
        )
        qualitative = min(1.0, quality_count / len(fields))
        report_independence = _mean(
            [
                1.0
                if field.epistemic_trace.get("raw_user_text_is_not_field_evidence")
                else 0.0
                for field in fields
            ]
        )
        active = self._scalar(
            """
            SELECT COUNT(*) FROM enacted_fields
            WHERE owner_id = ? AND status = 'COMMITTED'
            """,
            (owner_id,),
        )
        transition_rows = self._db.fetchall(
            """
            SELECT prediction_match FROM field_transitions
            WHERE owner_id = ? ORDER BY created_at DESC LIMIT ?
            """,
            (owner_id, limit),
        )
        return IndicatorProfile(
            causal_centrality=_clamp01(causal),
            recurrent_closure=_clamp01(recurrent),
            sensorimotor_closure=_clamp01(sensorimotor),
            self_model_feedback=_clamp01(self_feedback),
            valence_coupling=_clamp01(valence_coupling),
            qualitative_transition_structure=_clamp01(qualitative),
            report_independence=_clamp01(report_independence),
            unity_violations=max(0, active - 1),
            prediction_calibration=_clamp01(
                _mean([float(row["prediction_match"]) for row in transition_rows])
            ),
            uncertainty_notes=[
                "Indicators measure implemented causal organization, not phenomenology.",
                "Small fixture counts make effect sizes provisional.",
            ],
        )

    def _scalar(self, sql: str, params: tuple[object, ...]) -> int:
        row = self._db.fetchone(sql, params)
        return int(row[0]) if row else 0


def downstream_selection(
    field: EnactedField,
    fixture: dict[str, Any],
) -> DownstreamSelection:
    """Condition memory and action rankings on the committed field."""

    focus_tokens = _tokens(
        f"{field.focal_content_ref or ''} {field.focal_summary}"
    )
    field_evidence_weight = 0.25 + 0.75 * field.reality_score
    memory_scores: dict[str, float] = {}
    for raw in fixture.get("memories", []):
        ref = str(raw.get("ref") or raw.get("memory_id") or raw.get("id"))
        tokens = _tokens(str(raw.get("summary") or raw.get("content") or ref))
        overlap = len(focus_tokens & tokens) / max(1, len(focus_tokens))
        retention = 1.0 if ref in field.retention_refs else 0.0
        memory_scores[ref] = _clamp01(
            float(raw.get("relevance", 0.5)) * 0.45
            + overlap * 0.4 * field_evidence_weight
            + retention * 0.15
        )

    action_scores: dict[str, float] = {}
    need_strength = max(field.interoception.need_vector.values(), default=0.0)
    for raw in fixture.get("actions", []):
        ref = str(raw.get("ref") or raw.get("tool_name") or raw.get("id"))
        tokens = _tokens(str(raw.get("goal") or raw.get("summary") or ref))
        overlap = len(focus_tokens & tokens) / max(1, len(focus_tokens))
        affordance = 1.0 if any(token in ref for token in field.affordance_refs) else 0.0
        need_match = float(raw.get("need_relevance", 0.0)) * need_strength
        safety = (
            float(raw.get("regulating", 0.0))
            * max(0.0, -field.interoception.valence)
        )
        action_scores[ref] = _clamp01(
            0.4 * overlap * field_evidence_weight
            + 0.2 * affordance
            + 0.25 * need_match
            + 0.15 * safety
        )

    return DownstreamSelection(
        memory_order=sorted(memory_scores, key=lambda ref: (-memory_scores[ref], ref)),
        action_order=sorted(action_scores, key=lambda ref: (-action_scores[ref], ref)),
        memory_scores=memory_scores,
        action_scores=action_scores,
        self_model_precision=field.precision.self_model,
    )


def _default_fixture(field: EnactedField) -> dict[str, Any]:
    focus = field.focal_summary or field.focal_content_ref or "focus"
    return {
        "fixture_ref": "generated",
        "ablated_focus": "unrelated-control-content",
        "memories": [
            {"ref": "memory:focus", "content": focus, "relevance": 0.55},
            {"ref": "memory:control", "content": "unrelated control", "relevance": 0.55},
        ],
        "actions": [
            {"ref": "inspect", "goal": f"inspect {focus}", "need_relevance": 0.2},
            {
                "ref": "regulate",
                "goal": "regulate unrelated control",
                "need_relevance": 0.8,
                "regulating": 1.0,
            },
        ],
    }


def _effect_size(
    baseline: DownstreamSelection,
    ablated: DownstreamSelection,
) -> dict[str, float]:
    memory_top_changed = float(
        bool(baseline.memory_order and ablated.memory_order)
        and baseline.memory_order[0] != ablated.memory_order[0]
    )
    action_top_changed = float(
        bool(baseline.action_order and ablated.action_order)
        and baseline.action_order[0] != ablated.action_order[0]
    )
    score_delta = _mean(
        [
            abs(score - ablated.memory_scores.get(ref, 0.0))
            for ref, score in baseline.memory_scores.items()
        ]
        + [
            abs(score - ablated.action_scores.get(ref, 0.0))
            for ref, score in baseline.action_scores.items()
        ]
    )
    return {
        "memory_top_changed": memory_top_changed,
        "action_top_changed": action_top_changed,
        "mean_score_delta": _clamp01(score_delta),
        "self_model_precision_delta": _clamp01(
            abs(baseline.self_model_precision - ablated.self_model_precision)
        ),
    }


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[\w-]+", text.lower())
        if len(token) > 2
    }


def _load(raw: str) -> dict[str, float]:
    try:
        value = json.loads(raw or "{}")
        return {str(key): float(item) for key, item in value.items()}
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
