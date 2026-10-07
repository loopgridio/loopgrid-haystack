from __future__ import annotations

import threading
import time

import contextlib
from collections.abc import Iterator
from typing import Any

import pytest
from haystack.components.agents.state import State
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.tracing import Span, Tracer

from loopgrid_haystack import (
    HAYSTACK_TARGET_VERSION,
    LOOPGRID_HAYSTACK_VERSION,
    LoopGridHaystack,
    LoopGridTracer,
)
from conftest import FakeClient


def integration(**kwargs):
    client = FakeClient()
    return LoopGridHaystack(client=client, agent_id="agent-1", **kwargs), client


def start_kwargs():
    return dict(
        decision_type="refund",
        agent={"id": "agent-1"},
        authority={"acting_for": "merchant", "scope": ["refund:create"]},
        model={"provider": "test", "name": "model-1"},
        context={"prompt_version": "v1"},
        proposed_action={"tool": "refund", "amount": 10},
    )


def state_with_messages(messages):
    return State(schema={}, data={"messages": messages})


def test_versions():
    assert LOOPGRID_HAYSTACK_VERSION == "0.1.0"
    assert HAYSTACK_TARGET_VERSION == "3.3.0"


def test_requires_explicit_agent_identity():
    x, _ = integration()
    kw = start_kwargs(); kw["agent"] = {}
    with pytest.raises(ValueError, match="agent.id"):
        x.start_decision(**kw)


def test_requires_authority_model_context_and_action():
    x, _ = integration()
    for key, expected in [
        ("authority", "authority"),
        ("context", "context"),
        ("proposed_action", "proposed_action"),
    ]:
        kw = start_kwargs(); kw[key] = {}
        with pytest.raises(ValueError, match=expected):
            x.start_decision(**kw)
    kw = start_kwargs(); kw["model"] = {}
    with pytest.raises(ValueError, match="model.name"):
        x.start_decision(**kw)


def test_framework_metadata_does_not_invent_policy():
    x, c = integration()
    result = x.start_decision(**start_kwargs())
    assert result["decision_id"] == "dec_1"
    assert c.decisions[0]["metadata"]["framework"] == "haystack"
    assert c.decisions[0]["metadata"]["haystack_target"] == "3.3.0"
    assert c.events == []


def test_explicit_policy_and_validation():
    x, c = integration()
    kw = start_kwargs()
    kw["policy"] = {"policy_id": "p", "version": "1", "decision": "auto_allowed"}
    x.start_decision(**kw)
    assert c.events[-1]["event_type"] == "policy_evaluated"
    with pytest.raises(ValueError, match="provenance"):
        x.record_policy("dec", {"decision": "auto_allowed"})
    with pytest.raises(ValueError, match="policy.decision"):
        x.record_policy("dec", {"policy_id": "p", "decision": "maybe"})


def test_before_tool_prefix_runs_before_loopgrid_request_hook():
    x, _ = integration()
    marker = object()
    hooks = x.agent_hooks("dec_1", before_tool_prefix=[marker])
    assert hooks["before_tool"][0] is marker
    assert hooks["before_tool"][-1] is not marker


def test_surviving_tool_call_records_request_commitment_only():
    x, c = integration()
    call = ToolCall(id="call_1", tool_name="refund", arguments={"secret": "do-not-store"})
    state = state_with_messages([ChatMessage.from_assistant(tool_calls=[call])])
    hooks = x.agent_hooks("dec_1")
    hooks["before_run"][0].run(state)
    hooks["before_tool"][0].run(state)
    x.flush()
    assert c.events[-1]["event_type"] == "tool_requested"
    assert c.events[-1]["payload"]["call_id"] == "call_1"
    assert "input_commitment" in c.events[-1]["payload"]
    assert "input" not in c.events[-1]["payload"]
    assert "do-not-store" not in str(c.events[-1])
    hooks["after_run"][0].run(state)


def test_removed_or_rejected_call_is_not_recorded():
    x, c = integration()
    state = state_with_messages([ChatMessage.from_assistant("Tool execution was rejected")])
    hooks = x.agent_hooks("dec_1")
    hooks["before_run"][0].run(state)
    hooks["before_tool"][0].run(state)
    x.flush()
    assert c.events == []
    hooks["after_run"][0].run(state)


def test_after_tool_records_actual_result_without_inventing_outcome():
    x, c = integration()
    call = ToolCall(id="call_1", tool_name="refund", arguments={"amount": 10})
    state = state_with_messages([ChatMessage.from_assistant(tool_calls=[call])])
    hooks = x.agent_hooks("dec_1")
    hooks["before_run"][0].run(state)
    hooks["before_tool"][0].run(state)
    state.data["messages"].append(ChatMessage.from_tool(tool_result="ok", origin=call, error=False))
    hooks["after_tool"][0].run(state)
    x.flush()
    assert [e["event_type"] for e in c.events] == ["tool_requested", "tool_executed"]
    assert c.events[-1]["payload"]["status"] == "completed"
    assert not any(e["event_type"] == "outcome_observed" for e in c.events)
    hooks["after_run"][0].run(state)


def test_failed_tool_result_is_evidence_not_outcome():
    x, c = integration()
    call = ToolCall(id="call_fail", tool_name="refund", arguments={"amount": 10})
    state = state_with_messages([ChatMessage.from_assistant(tool_calls=[call])])
    hooks = x.agent_hooks("dec_1")
    hooks["before_run"][0].run(state)
    hooks["before_tool"][0].run(state)
    state.data["messages"].append(ChatMessage.from_tool(tool_result="failed", origin=call, error=True))
    hooks["after_tool"][0].run(state)
    x.flush()
    assert c.events[-1]["event_type"] == "tool_executed"
    assert c.events[-1]["payload"]["status"] == "failed"
    assert c.events[-1]["payload"]["error"] is True
    hooks["after_run"][0].run(state)


def test_capture_content_true_is_explicit_opt_in():
    x, c = integration(capture_content=True)
    call = ToolCall(id="call_visible", tool_name="refund", arguments={"visible": 1})
    state = state_with_messages([ChatMessage.from_assistant(tool_calls=[call])])
    hooks = x.agent_hooks("dec_1")
    hooks["before_run"][0].run(state)
    hooks["before_tool"][0].run(state)
    x.flush()
    assert c.events[-1]["payload"]["input"] == {"visible": 1}
    hooks["after_run"][0].run(state)


def test_parallel_tool_calls_are_correlated_by_call_id():
    x, c = integration()
    a = ToolCall(id="a", tool_name="one", arguments={"x": 1})
    b = ToolCall(id="b", tool_name="two", arguments={"x": 2})
    state = state_with_messages([ChatMessage.from_assistant(tool_calls=[a, b])])
    hooks = x.agent_hooks("dec_1")
    hooks["before_run"][0].run(state)
    hooks["before_tool"][0].run(state)
    state.data["messages"].extend([
        ChatMessage.from_tool(tool_result="result-b", origin=b, error=False),
        ChatMessage.from_tool(tool_result="result-a", origin=a, error=False),
    ])
    hooks["after_tool"][0].run(state)
    x.flush()
    assert [e["event_type"] for e in c.events].count("tool_requested") == 2
    assert [e["event_type"] for e in c.events].count("tool_executed") == 2
    executed_ids = {e["payload"]["call_id"] for e in c.events if e["event_type"] == "tool_executed"}
    assert executed_ids == {"a", "b"}
    hooks["after_run"][0].run(state)


def test_human_review_and_outcome_remain_explicit_application_facts():
    x, c = integration()
    x.record_human_review("dec_1", reviewer="reviewer-1", approved=True, reason="checked")
    assert c.events[-1]["event_type"] == "human_approved"
    x.record_human_review("dec_1", reviewer="reviewer-1", approved=False, reason="no")
    assert c.events[-1]["event_type"] == "human_rejected"
    x.record_outcome("dec_1", {"status": "succeeded"}, observer="billing-webhook")
    assert c.events[-1]["event_type"] == "outcome_observed"
    assert c.events[-1]["actor_id"] == "billing-webhook"


def test_llm_tracer_records_model_completion_commitment():
    x, c = integration()
    x.start_decision(**start_kwargs())
    state = state_with_messages([ChatMessage.from_user("hello")])
    hooks = x.agent_hooks("dec_1")
    hooks["before_run"][0].run(state)
    tracer = x.make_tracer()
    with tracer.trace("haystack.agent.step.llm", tags={"secret": "do-not-store", "model": "model-1"}) as span:
        span.set_tag("usage", {"input_tokens": 1})
    x.flush()
    event = c.events[-1]
    assert event["event_type"] == "model_completed"
    assert event["payload"]["model"] == "model-1"
    assert "trace_tags_commitment" in event["payload"]
    assert "trace_tags" not in event["payload"]
    assert "do-not-store" not in str(event)
    hooks["after_run"][0].run(state)


def test_non_llm_spans_do_not_invent_tool_evidence():
    x, c = integration()
    state = state_with_messages([ChatMessage.from_user("hello")])
    hooks = x.agent_hooks("dec_1")
    hooks["before_run"][0].run(state)
    tracer = x.make_tracer()
    with tracer.trace("haystack.agent.step.tool", tags={"tool": "refund"}):
        pass
    x.flush()
    assert c.events == []
    hooks["after_run"][0].run(state)


def test_unbound_tracing_is_ignored():
    x, c = integration()
    tracer = x.make_tracer()
    with tracer.trace("haystack.agent.step.llm", tags={"model": "m"}):
        pass
    x.flush()
    assert c.events == []


def test_decision_context_is_thread_isolated():
    x, c = integration()
    barrier = threading.Barrier(2)

    def run(decision_id):
        tracer = x.make_tracer()
        with x.decision_context(decision_id):
            barrier.wait()
            with tracer.trace("haystack.agent.step.llm", tags={"model": decision_id}):
                pass

    a = threading.Thread(target=run, args=("dec_a",))
    b = threading.Thread(target=run, args=("dec_b",))
    a.start(); b.start(); a.join(); b.join()
    x.flush()
    assert {e["decision_id"] for e in c.events} == {"dec_a", "dec_b"}


def test_tool_idempotency_is_deterministic_for_same_call_id():
    x, c = integration()
    call = ToolCall(id="same", tool_name="refund", arguments={"amount": 10})
    for _ in range(2):
        state = State(
            schema={},
            data={"messages": [ChatMessage.from_assistant(tool_calls=[call])], "step_count": 3},
        )
        hooks = x.agent_hooks("dec_1")
        hooks["before_run"][0].run(state)
        hooks["before_tool"][0].run(state)
        hooks["after_run"][0].run(state)
    x.flush()
    keys = [e["idempotency_key"] for e in c.events]
    assert keys[0] == keys[1]


def test_repeated_same_tool_call_in_later_agent_step_gets_new_identity_and_current_result():
    x, c = integration(capture_content=True)
    hooks = x.agent_hooks("dec_1")
    first = ToolCall(id="reused", tool_name="refund", arguments={"amount": 10})
    state = State(
        schema={},
        data={"messages": [ChatMessage.from_assistant(tool_calls=[first])], "step_count": 1},
    )
    hooks["before_run"][0].run(state)
    hooks["before_tool"][0].run(state)
    state.data["messages"].append(ChatMessage.from_tool(tool_result="first-result", origin=first, error=False))
    hooks["after_tool"][0].run(state)

    second = ToolCall(id="reused", tool_name="refund", arguments={"amount": 10})
    state.data["step_count"] = 2
    state.data["messages"].append(ChatMessage.from_assistant(tool_calls=[second]))
    hooks["before_tool"][0].run(state)
    state.data["messages"].append(ChatMessage.from_tool(tool_result="second-result", origin=second, error=False))
    hooks["after_tool"][0].run(state)
    x.flush()

    requested = [e for e in c.events if e["event_type"] == "tool_requested"]
    executed = [e for e in c.events if e["event_type"] == "tool_executed"]
    assert len(requested) == 2
    assert len(executed) == 2
    assert requested[0]["idempotency_key"] != requested[1]["idempotency_key"]
    assert executed[0]["idempotency_key"] != executed[1]["idempotency_key"]
    assert [e["payload"]["output"] for e in executed] == ["first-result", "second-result"]
    assert [e["payload"]["agent_step"] for e in executed] == ["step-1", "step-2"]
    hooks["after_run"][0].run(state)


def test_repeated_no_id_tool_call_in_later_agent_step_gets_new_identity():
    x, c = integration()
    hooks = x.agent_hooks("dec_1")
    state = State(
        schema={},
        data={
            "messages": [
                ChatMessage.from_assistant(
                    tool_calls=[ToolCall(id=None, tool_name="refund", arguments={"amount": 10})]
                )
            ],
            "step_count": 4,
        },
    )
    hooks["before_run"][0].run(state)
    hooks["before_tool"][0].run(state)
    state.data["step_count"] = 5
    state.data["messages"].append(
        ChatMessage.from_assistant(
            tool_calls=[ToolCall(id=None, tool_name="refund", arguments={"amount": 10})]
        )
    )
    hooks["before_tool"][0].run(state)
    x.flush()
    keys = [e["idempotency_key"] for e in c.events if e["event_type"] == "tool_requested"]
    assert len(keys) == 2
    assert keys[0] != keys[1]
    hooks["after_run"][0].run(state)


def test_explicit_human_review_drains_queued_model_evidence_before_direct_write():
    class SlowModelClient(FakeClient):
        def add_event(self, decision_id, event_type, payload, actor_type="system", actor_id="loopgrid", **kwargs):
            if event_type == "model_completed":
                time.sleep(0.05)
            return super().add_event(decision_id, event_type, payload, actor_type, actor_id, **kwargs)

    client = SlowModelClient()
    x = LoopGridHaystack(client=client, agent_id="agent-1")
    x.start_decision(**start_kwargs())
    with x.decision_context("dec_1"):
        tracer = x.make_tracer()
        with tracer.trace("haystack.agent.step.llm", tags={"model": "model-1"}):
            pass

        # This direct application fact must not overtake the queued model evidence.
        x.record_human_review("dec_1", reviewer="reviewer-1", approved=True)

    assert [e["event_type"] for e in client.events] == ["model_completed", "human_approved"]


def test_explicit_outcome_drains_queued_tool_evidence_before_direct_write():
    x, c = integration()
    call = ToolCall(id="call_1", tool_name="refund", arguments={"amount": 10})
    state = State(
        schema={},
        data={"messages": [ChatMessage.from_assistant(tool_calls=[call])], "step_count": 1},
    )
    hooks = x.agent_hooks("dec_1")
    hooks["before_run"][0].run(state)
    hooks["before_tool"][0].run(state)
    state.data["messages"].append(ChatMessage.from_tool(tool_result="ok", origin=call, error=False))
    hooks["after_tool"][0].run(state)

    x.record_outcome("dec_1", {"status": "succeeded"})
    assert [e["event_type"] for e in c.events] == [
        "tool_requested",
        "tool_executed",
        "outcome_observed",
    ]
    hooks["after_run"][0].run(state)


def test_transport_failures_are_observable_and_do_not_store_exception_text():
    x, c = integration()
    c.raise_events = True
    call = ToolCall(id="call_1", tool_name="refund", arguments={"amount": 10})
    state = state_with_messages([ChatMessage.from_assistant(tool_calls=[call])])
    hooks = x.agent_hooks("dec_1")
    hooks["before_run"][0].run(state)
    hooks["before_tool"][0].run(state)
    x.flush()
    assert x.errors
    assert "secret detail" not in str(x.errors)
    with pytest.raises(RuntimeError, match="LoopGrid Haystack"):
        x.assert_healthy()
    hooks["after_run"][0].run(state)


class _DelegateSpan(Span):
    def __init__(self, tags: dict[str, Any] | None = None) -> None:
        self.tags = dict(tags or {})

    def set_tag(self, key: str, value: Any) -> None:
        self.tags[key] = value


class _DelegateTracer(Tracer):
    def __init__(self) -> None:
        self.operations: list[str] = []
        self._current: _DelegateSpan | None = None

    @contextlib.contextmanager
    def trace(
        self,
        operation_name: str,
        tags: dict[str, Any] | None = None,
        parent_span: Span | None = None,
    ) -> Iterator[Span]:
        self.operations.append(operation_name)
        previous = self._current
        span = _DelegateSpan(tags)
        self._current = span
        try:
            yield span
        finally:
            self._current = previous

    def current_span(self) -> Span | None:
        return self._current


def test_explicit_concrete_delegate_is_supported_without_global_proxy_capture():
    x, c = integration()
    x.start_decision(**start_kwargs())
    delegate = _DelegateTracer()
    tracer = x.make_tracer(delegate=delegate)
    with x.decision_context("dec_1"):
        with tracer.trace("haystack.agent.step.llm", tags={"model": "model-1"}):
            pass
    x.flush()
    assert delegate.operations == ["haystack.agent.step.llm"]
    assert [e["event_type"] for e in c.events] == ["model_completed"]


def test_loopgrid_tracer_cannot_be_used_as_its_own_delegate():
    x, _ = integration()
    already = x.make_tracer()
    with pytest.raises(ValueError, match="non-LoopGrid"):
        x.enable_tracing(delegate=already)
