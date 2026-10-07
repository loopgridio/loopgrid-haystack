from __future__ import annotations

from typing import Any

import pytest
from haystack import component, tracing
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

from loopgrid_haystack import LoopGridHaystack
from conftest import FakeClient


@component
class ScriptedChatGenerator:
    def __init__(self) -> None:
        self.calls = 0

    @component.output_types(replies=list[ChatMessage])
    def run(self, messages: list[ChatMessage], tools: Any = None, **kwargs: Any) -> dict[str, Any]:
        self.calls += 1
        if self.calls == 1:
            return {
                "replies": [
                    ChatMessage.from_assistant(
                        tool_calls=[ToolCall(id="call_1", tool_name="refund", arguments={"amount": 25})],
                        meta={"model": "scripted-haystack"},
                    )
                ]
            }
        return {
            "replies": [
                ChatMessage.from_assistant(
                    "Sandbox refund completed.",
                    meta={"model": "scripted-haystack"},
                )
            ]
        }

    @component.output_types(replies=list[ChatMessage])
    async def run_async(self, messages: list[ChatMessage], tools: Any = None, **kwargs: Any) -> dict[str, Any]:
        return self.run(messages=messages, tools=tools, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {"type": "ScriptedChatGenerator", "init_parameters": {}}


class StaticConfirmationUI(ConfirmationUI):
    def __init__(
        self,
        *,
        action: str,
        feedback: str | None = None,
        new_tool_params: dict[str, Any] | None = None,
    ) -> None:
        self.action = action
        self.feedback = feedback
        self.new_tool_params = new_tool_params

    def get_user_confirmation(
        self,
        tool_name: str,
        tool_description: str,
        tool_params: dict[str, Any],
    ) -> ConfirmationUIResult:
        return ConfirmationUIResult(
            action=self.action,
            feedback=self.feedback,
            new_tool_params=self.new_tool_params,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "StaticConfirmationUI",
            "init_parameters": {
                "action": self.action,
                "feedback": self.feedback,
                "new_tool_params": self.new_tool_params,
            },
        }


def make_confirmation_hook(ui: ConfirmationUI) -> ConfirmationHook:
    return ConfirmationHook(
        confirmation_strategies={
            "refund": BlockingConfirmationStrategy(
                confirmation_policy=AlwaysAskPolicy(),
                confirmation_ui=ui,
            )
        }
    )



def test_real_haystack_agent_runtime_records_model_and_tool_boundaries():
    client = FakeClient()
    lg = LoopGridHaystack(client=client, agent_id="support-agent")
    lg.start_decision(
        decision_type="sandbox_refund",
        agent={"id": "support-agent"},
        authority={"acting_for": "test", "scope": ["refund:create"]},
        model={"provider": "scripted", "name": "scripted-haystack"},
        context={"prompt_version": "runtime-test"},
        proposed_action={"tool": "refund", "amount": 25},
        policy={"policy_id": "runtime", "version": "1", "decision": "auto_allowed"},
    )

    executions = {"count": 0}

    def refund(amount: int) -> str:
        executions["count"] += 1
        return f"sandbox refund {amount}"

    tool = Tool(
        name="refund",
        description="Execute a sandbox refund",
        parameters={
            "type": "object",
            "properties": {"amount": {"type": "integer"}},
            "required": ["amount"],
        },
        function=refund,
    )

    # Haystack's public tracing.tracer is a process-level facade. It must not be captured
    # and delegated to, because after enable_tracing() it routes back to the newly installed
    # tracer and would recurse. LoopGrid installs directly unless a concrete delegate tracer
    # instance is supplied explicitly by the application.
    tracing.disable_tracing()
    lg.enable_tracing()
    try:
        agent = Agent(
            chat_generator=ScriptedChatGenerator(),
            tools=[tool],
            hooks=lg.agent_hooks("dec_1"),
        )
        result = agent.run(messages=[ChatMessage.from_user("refund the sandbox charge")])
        lg.flush()
        lg.assert_healthy()
    finally:
        tracing.disable_tracing()

    assert executions["count"] == 1
    assert result["last_message"].text == "Sandbox refund completed."
    types = [e["event_type"] for e in client.events]
    assert "model_completed" in types
    assert "tool_requested" in types
    assert "tool_executed" in types
    assert types.index("tool_requested") < types.index("tool_executed")
    assert "outcome_observed" not in types


def test_native_confirmation_hook_rejection_produces_no_tool_request_or_execution_evidence():
    client = FakeClient()
    lg = LoopGridHaystack(client=client, agent_id="support-agent")
    executions = {"count": 0}

    def refund(amount: int) -> str:
        executions["count"] += 1
        return f"sandbox refund {amount}"

    tool = Tool(
        name="refund",
        description="Execute a sandbox refund",
        parameters={
            "type": "object",
            "properties": {"amount": {"type": "integer"}},
            "required": ["amount"],
        },
        function=refund,
    )
    confirmation_hook = make_confirmation_hook(
        StaticConfirmationUI(action="reject", feedback="reviewer rejected sandbox action")
    )
    agent = Agent(
        chat_generator=ScriptedChatGenerator(),
        tools=[tool],
        hooks=lg.agent_hooks("dec_1", before_tool_prefix=[confirmation_hook]),
    )

    agent.run(messages=[ChatMessage.from_user("refund the sandbox charge")])
    lg.flush()
    lg.assert_healthy()

    assert executions["count"] == 0
    types = [e["event_type"] for e in client.events]
    assert "tool_requested" not in types
    assert "tool_executed" not in types


def test_native_confirmation_hook_modification_records_and_executes_surviving_arguments():
    client = FakeClient()
    lg = LoopGridHaystack(client=client, agent_id="support-agent", capture_content=True)
    execution = {"amount": None}

    def refund(amount: int) -> str:
        execution["amount"] = amount
        return f"sandbox refund {amount}"

    tool = Tool(
        name="refund",
        description="Execute a sandbox refund",
        parameters={
            "type": "object",
            "properties": {"amount": {"type": "integer"}},
            "required": ["amount"],
        },
        function=refund,
    )
    confirmation_hook = make_confirmation_hook(
        StaticConfirmationUI(
            action="modify",
            feedback="reviewer reduced amount",
            new_tool_params={"amount": 7},
        )
    )
    agent = Agent(
        chat_generator=ScriptedChatGenerator(),
        tools=[tool],
        hooks=lg.agent_hooks("dec_1", before_tool_prefix=[confirmation_hook]),
    )

    agent.run(messages=[ChatMessage.from_user("refund the sandbox charge")])
    lg.flush()
    lg.assert_healthy()

    requested = [e for e in client.events if e["event_type"] == "tool_requested"]
    executed = [e for e in client.events if e["event_type"] == "tool_executed"]
    assert execution["amount"] == 7
    assert len(requested) == 1
    assert len(executed) == 1
    assert requested[0]["payload"]["input"] == {"amount": 7}


@pytest.mark.asyncio
async def test_real_haystack_agent_async_runtime_records_model_and_tool_boundaries():
    client = FakeClient()
    lg = LoopGridHaystack(client=client, agent_id="support-agent")
    lg.start_decision(
        decision_type="sandbox_refund",
        agent={"id": "support-agent"},
        authority={"acting_for": "test", "scope": ["refund:create"]},
        model={"provider": "scripted", "name": "scripted-haystack"},
        context={"prompt_version": "runtime-async-test"},
        proposed_action={"tool": "refund", "amount": 25},
    )
    executions = {"count": 0}

    async def refund_async(amount: int) -> str:
        executions["count"] += 1
        return f"sandbox refund {amount}"

    tool = Tool(
        name="refund",
        description="Execute a sandbox refund",
        parameters={
            "type": "object",
            "properties": {"amount": {"type": "integer"}},
            "required": ["amount"],
        },
        async_function=refund_async,
    )

    tracing.disable_tracing()
    lg.enable_tracing()
    try:
        agent = Agent(
            chat_generator=ScriptedChatGenerator(),
            tools=[tool],
            hooks=lg.agent_hooks("dec_1"),
        )
        result = await agent.run_async(messages=[ChatMessage.from_user("refund the sandbox charge")])
        lg.flush()
        lg.assert_healthy()
    finally:
        tracing.disable_tracing()

    assert executions["count"] == 1
    assert result["last_message"].text == "Sandbox refund completed."
    types = [e["event_type"] for e in client.events]
    assert "model_completed" in types
    assert "tool_requested" in types
    assert "tool_executed" in types
    assert types.index("tool_requested") < types.index("tool_executed")
