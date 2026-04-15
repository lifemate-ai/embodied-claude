"""Schemas for joint-attention-mcp."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CameraPose(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pan_deg: float
    tilt_deg: float
    zoom: float


class ScenePerson(BaseModel):
    model_config = ConfigDict(extra="forbid")

    person_id: str | None = None
    display_name: str | None = None
    relative_position: str
    distance: str
    gaze_target: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)


class SceneObject(BaseModel):
    model_config = ConfigDict(extra="forbid")

    object_id: str
    label: str
    attributes: dict[str, Any] = Field(default_factory=dict)
    relative_position: list[str] = Field(default_factory=list)
    salience: float = Field(ge=0.0, le=1.0)


class SceneParse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ts: str
    camera_pose: CameraPose
    scene_summary: str
    people: list[ScenePerson] = Field(default_factory=list)
    objects: list[SceneObject] = Field(default_factory=list)


class ReferenceMatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    object_id: str
    confidence: float = Field(ge=0.0, le=1.0)
    why: list[str]


class ReferenceResolution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    matches: list[ReferenceMatch]
