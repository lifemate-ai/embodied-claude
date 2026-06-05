"""PEG rule definitions for the agent turn.

The intended shape is:

    Turn       <- Observe? Recall? Interpret Propose? Respond
    Observe    <- camera.see / wifi_cam.look_around / memory.search
    Interpret  <- Observed -> Hypothesis -> BoundaryCheck
    Respond    <- CasualReply / TechnicalReply / SensitiveReply

The PEG library has not been picked yet (see README). For now this module
exposes the textual grammar as a constant so downstream code can begin
reasoning about rule names even before the runtime is wired.
"""

from __future__ import annotations

TURN_GRAMMAR_DRAFT = """
Turn       <- Observe? Recall? Interpret Propose? Respond
Observe    <- ToolPerception+
Recall     <- MemoryQuery+
Interpret  <- Observation Hypothesis BoundaryCheck
Propose    <- Action
Respond    <- CasualReply / TechnicalReply / SensitiveReply

Observation   <- EpistemicClaim("observed")
Hypothesis    <- EpistemicClaim("inferred")
BoundaryCheck <- PolicyRule+
"""

__all__ = ["TURN_GRAMMAR_DRAFT"]
