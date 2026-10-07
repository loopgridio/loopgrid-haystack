"""Native Haystack ConfirmationHook -> LoopGrid Core approval ordering E2E.

A deterministic custom ConfirmationUI auto-confirms a sandbox tool while recording the
application-authenticated reviewer explicitly in LoopGrid. No model-provider call is made.
"""
from __future__ import annotations

import json
import os
import urllib.request
import uuid
from typing import Any

from haystack import component
from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.hooks.human_in_the_loop import (
    AlwaysAskPolicy,
    BlockingConfirmationStrategy,
    ConfirmationHook,
    ConfirmationUIResult,
)
from haystack.hooks.human_in_the_loop.types import ConfirmationUI
from haystack.tools import Tool
from loopgrid import LoopGrid

from loopgrid_haystack import HAYSTACK_TARGET_VERSION, LoopGridHaystack

BASE_URL = os.getenv("LOOPGRID_BASE_URL", "http://127.0.0.1:8000").rstrip("/")


def post_json(path: str, body: dict):
    req = urllib.request.Request(BASE_URL + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.load(r)


def get_json(path: str):
    with urllib.request.urlopen(BASE_URL + path, timeout=10) as r:
        return json.load(r)


@component
class ScriptedChatGenerator:
    def __init__(self) -> None:
        self.calls = 0

    @component.output_types(replies=list[ChatMessage])
    def run(self, messages: list[ChatMessage], tools: Any = None, **kwargs: Any) -> dict[str, Any]:
        self.calls += 1
        if self.calls == 1:
            return {"replies": [ChatMessage.from_assistant(
                tool_calls=[ToolCall(id="call_refund_approval", tool_name="sandbox_refund", arguments={"amount": 25})],
                meta={"model": "scripted-haystack"},
            )]}
        return {"replies": [ChatMessage.from_assistant(
            "Approved sandbox refund completed.", meta={"model": "scripted-haystack"}
        )]}

    def to_dict(self) -> dict[str, Any]:
        return {"type": "ScriptedChatGenerator", "init_parameters": {}}


ready = get_json("/ready")
workspace = post_json("/api/v1/workspaces", {"name": f"haystack-approval-{uuid.uuid4().hex[:8]}"})
workspace_id = workspace["workspace_id"]
client = LoopGrid(base_url=BASE_URL, workspace_id=workspace_id)
integration = LoopGridHaystack(client=client, workspace_id=workspace_id, agent_id="support-agent")
integration.enable_tracing()

decision = integration.start_decision(
    decision_type="sandbox_refund",
    agent={"id": "support-agent", "version": "approval-e2e"},
    authority={"acting_for": "LoopGrid E2E", "scope": ["refund:create"], "limit_usd": 100},
    model={"provider": "scripted", "name": "scripted-haystack"},
    context={"prompt_version": "haystack-approval-v1"},
    proposed_action={"tool": "sandbox_refund", "amount": 25, "currency": "USD"},
    metadata={"sandbox": True, "real_money_moved": False},
    policy={"policy_id": "approval-policy", "version": "1", "decision": "human_approval_required"},
)
decision_id = decision["decision_id"]
reviewer = "reviewer.loopgrid-e2e"
executions = {"count": 0}


class LoopGridApprovalUI(ConfirmationUI):
    def get_user_confirmation(self, tool_name: str, tool_description: str, tool_params: dict[str, Any]) -> ConfirmationUIResult:
        integration.record_human_review(
            decision_id,
            reviewer=reviewer,
            approved=True,
            reason="Approved sandbox action",
        )
        return ConfirmationUIResult(action="confirm", feedback="approved by deterministic E2E reviewer")

    def to_dict(self) -> dict[str, Any]:
        return {"type": "LoopGridApprovalUI", "init_parameters": {}}


def sandbox_refund(amount: int) -> dict:
    executions["count"] += 1
    return {"status": "succeeded", "sandbox": True, "real_money_moved": False, "amount": amount}


tool = Tool(
    name="sandbox_refund",
    description="Execute a sandbox-only refund simulation",
    parameters={
        "type": "object",
        "properties": {"amount": {"type": "integer"}},
        "required": ["amount"],
    },
    function=sandbox_refund,
)
confirmation_hook = ConfirmationHook(
    confirmation_strategies={
        "sandbox_refund": BlockingConfirmationStrategy(
            confirmation_policy=AlwaysAskPolicy(),
            confirmation_ui=LoopGridApprovalUI(),
        )
    }
)

# Critical ordering: ConfirmationHook first; LoopGrid's before_tool request evidence second.
agent = Agent(
    chat_generator=ScriptedChatGenerator(),
    tools=[tool],
    hooks=integration.agent_hooks(decision_id, before_tool_prefix=[confirmation_hook]),
)
result = agent.run(messages=[ChatMessage.from_user("Handle the sandbox refund.")])
integration.flush()
integration.assert_healthy()
assert executions["count"] == 1

integration.record_outcome(
    decision_id,
    {"status": "succeeded", "sandbox": True, "real_money_moved": False},
)
detail = client.get_decision(decision_id)
event_types = [e.get("event_type") for e in detail.get("events", [])]
assert event_types.index("human_approved") < event_types.index("tool_requested"), event_types
assert event_types.index("tool_requested") < event_types.index("tool_executed"), event_types
summary = detail.get("summary", {})
coverage = detail.get("coverage", {})
verification = detail.get("verification", {})
state_name = summary.get("lifecycle", {}).get("state") if isinstance(summary.get("lifecycle"), dict) else summary.get("lifecycle")
assert state_name == "evidence_complete", detail
assert coverage.get("score") == 100 and coverage.get("complete") is True, detail
assert verification.get("valid") is True and verification.get("failures", []) == [], detail

print(json.dumps({
    "coreVersion": ready.get("version"),
    "haystackVersionTarget": HAYSTACK_TARGET_VERSION,
    "workspaceId": workspace_id,
    "decisionId": decision_id,
    "reviewer": reviewer,
    "toolExecutions": executions["count"],
    "finalOutput": result["last_message"].text,
    "lifecycle": state_name,
    "coverage": coverage.get("score"),
    "coverageComplete": coverage.get("complete"),
    "verifyValid": verification.get("valid"),
    "failures": verification.get("failures", []),
    "eventTypes": event_types,
}, indent=2))
