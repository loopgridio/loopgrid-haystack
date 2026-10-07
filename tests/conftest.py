from __future__ import annotations

import threading


class FakeClient:
    def __init__(self):
        self.events = []
        self.decisions = []
        self.raise_events = False
        self.lock = threading.Lock()

    def record_decision(self, **kwargs):
        self.decisions.append(kwargs)
        return {"decision_id": f"dec_{len(self.decisions)}"}

    def add_event(self, decision_id, event_type, payload, actor_type="system", actor_id="loopgrid", **kwargs):
        if self.raise_events:
            raise RuntimeError("transport unavailable: secret detail")
        with self.lock:
            event = {
                "decision_id": decision_id,
                "event_type": event_type,
                "payload": payload,
                "actor_type": actor_type,
                "actor_id": actor_id,
                **kwargs,
            }
            self.events.append(event)
            return event

    def policy_evaluated(self, decision_id, payload, actor_id="policy", **kwargs):
        return self.add_event(decision_id, "policy_evaluated", payload, "policy", actor_id, **kwargs)

    def human_approved(self, decision_id, reviewer, reason="", **kwargs):
        return self.add_event(
            decision_id,
            "human_approved",
            {"approved": True, "reviewer": reviewer, "reason": reason},
            "human",
            reviewer,
            **kwargs,
        )

    def human_rejected(self, decision_id, reviewer, reason="", **kwargs):
        return self.add_event(
            decision_id,
            "human_rejected",
            {"approved": False, "reviewer": reviewer, "reason": reason},
            "human",
            reviewer,
            **kwargs,
        )

    def outcome_observed(self, decision_id, payload, actor_id="outcome-observer", **kwargs):
        return self.add_event(decision_id, "outcome_observed", payload, "system", actor_id, **kwargs)
