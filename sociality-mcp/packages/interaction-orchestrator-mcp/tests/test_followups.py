"""Follow-up suggestions keep their tone-independent fields (#173)."""

from interaction_orchestrator_mcp.compose import _collect_followups


class _Store:
    def suggest_followup(self, *, person_id, context):
        return {
            "suggestions": [
                {
                    "text": "a ready-made line",
                    "reason": "why",
                    "intent": "check_in_after_stress",
                    "topic": "会議多くて疲れた",
                }
            ]
        }


def test_intent_and_topic_reach_the_interaction_context():
    [item] = _collect_followups(_Store(), "kouta", "hello")

    assert item.text == "a ready-made line"
    assert item.intent == "check_in_after_stress"
    assert item.topic == "会議多くて疲れた"
