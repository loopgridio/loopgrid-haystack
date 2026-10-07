from __future__ import annotations

import contextlib
import contextvars
import hashlib
import json
import queue
import threading
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from haystack import tracing
from haystack.tracing import Span, Tracer
from loopgrid import LoopGrid

LOOPGRID_HAYSTACK_VERSION = "0.1.0"
HAYSTACK_TARGET_VERSION = "3.3.0"
_FRAMEWORK = "haystack"


def _require_nonempty(name: str, value: Any) -> None:
    if value is None or value == "" or value == {} or value == []:
        raise ValueError(f"{name} is required")


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        ).encode("utf-8")
    except Exception:
        return repr(value).encode("utf-8", errors="replace")


def _commitment(value: Any) -> dict[str, Any]:
    raw = _canonical_bytes(value)
    return {
        "algorithm": "sha256",
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }


def _idempotency(*parts: Any) -> str:
    raw = "|".join("" if p is None else str(p) for p in parts).encode("utf-8")
    return "lg-haystack-" + hashlib.sha256(raw).hexdigest()


def _call_identity(tool_call: Any) -> str:
    call_id = getattr(tool_call, "id", None)
    name = getattr(tool_call, "tool_name", None) or "tool"
    arguments = getattr(tool_call, "arguments", None)
    arguments_hash = _commitment(arguments)["sha256"]
    if call_id:
        # Tool-call IDs are provider/framework supplied and are not guaranteed to be unique
        # across an entire Agent decision. Include the surviving tool name/arguments so a
        # reused ID with modified parameters cannot collapse correlation within one step.
        return f"id-{call_id}-{name}-{arguments_hash}"
    return f"local-{name}-{arguments_hash}"


def _tool_step_identity(state: Any) -> str:
    """Return a stable identity for the current Agent tool step.

    Haystack 3.x exposes ``step_count`` in Agent state. For direct hook/unit usage where a
    synthetic ``State`` may omit that field, the current conversation length is a deterministic
    fallback: retrying the same hook against the same pending message keeps the same identity,
    while a later tool step has additional messages and therefore a different identity.
    """

    step_count = state.data.get("step_count")
    if step_count is not None:
        return f"step-{step_count}"
    messages = state.data.get("messages") or []
    return f"messages-{len(messages)}"


def _tool_calls_from_state(state: Any) -> list[Any]:
    messages = state.data.get("messages") or []
    if not messages:
        return []
    calls = getattr(messages[-1], "tool_calls", None)
    return list(calls or [])


def _tool_results_from_state(state: Any) -> list[Any]:
    results: list[Any] = []
    for message in state.data.get("messages") or []:
        result = getattr(message, "tool_call_result", None)
        if result is not None:
            results.append(result)
    return results


def _safe_tags(tags: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(tags or {})


def _model_name_from_tags(tags: Mapping[str, Any], fallback: str | None) -> str | None:
    preferred = (
        "haystack.chat_generator.model",
        "haystack.component.model",
        "gen_ai.request.model",
        "model",
    )
    for key in preferred:
        value = tags.get(key)
        if isinstance(value, (str, int, float)) and value != "":
            return str(value)
    for key, value in tags.items():
        if "model" in str(key).lower() and isinstance(value, (str, int, float)) and value != "":
            return str(value)
    return fallback


@dataclass(frozen=True)
class _QueuedEvent:
    decision_id: str
    event_type: str
    payload: dict[str, Any]
    actor_type: str
    actor_id: str
    idempotency_key: str


@dataclass(frozen=True)
class _PendingToolCall:
    call: Any
    call_identity: str
    occurrence_identity: str
    step_identity: str
    ordinal: int


@dataclass(frozen=True)
class _PendingToolBatch:
    decision_id: str
    calls: tuple[_PendingToolCall, ...]
    result_offset: int


class _EventTransport:
    """Single FIFO worker so hook/tracer callbacks do not block Haystack execution."""

    def __init__(self, owner: "LoopGridHaystack") -> None:
        self._owner = owner
        self._queue: queue.Queue[_QueuedEvent | object] = queue.Queue()
        self._stop = object()
        self._worker: threading.Thread | None = None
        self._worker_lock = threading.Lock()

    def _ensure_worker(self) -> None:
        with self._worker_lock:
            if self._worker is not None and self._worker.is_alive():
                return
            self._worker = threading.Thread(
                target=self._worker_main,
                name="loopgrid-haystack",
                daemon=True,
            )
            self._worker.start()

    def _worker_main(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is self._stop:
                    return
                assert isinstance(item, _QueuedEvent)
                try:
                    self._owner.client.add_event(
                        item.decision_id,
                        item.event_type,
                        item.payload,
                        actor_type=item.actor_type,
                        actor_id=item.actor_id,
                        idempotency_key=item.idempotency_key,
                    )
                except Exception as exc:  # observability/evidence must not break the Agent
                    self._owner._record_error(item.event_type, item.decision_id, exc)
            finally:
                self._queue.task_done()

    def enqueue(
        self,
        decision_id: str,
        event_type: str,
        payload: dict[str, Any],
        *,
        actor_type: str,
        actor_id: str,
        identity: Sequence[Any],
    ) -> None:
        self._ensure_worker()
        self._queue.put(
            _QueuedEvent(
                decision_id=decision_id,
                event_type=event_type,
                payload=payload,
                actor_type=actor_type,
                actor_id=actor_id,
                idempotency_key=_idempotency(decision_id, event_type, *identity),
            )
        )

    def flush(self) -> None:
        self._ensure_worker()
        self._queue.join()

    def shutdown(self) -> None:
        self.flush()
        worker = self._worker
        if worker is not None and worker.is_alive():
            self._queue.put(self._stop)
            self._queue.join()
            worker.join(timeout=5)


@dataclass
class LoopGridSpan(Span):
    """Haystack Span implementation used by :class:`LoopGridTracer`."""

    owner: "LoopGridHaystack"
    operation_name: str
    decision_id: str | None
    parent_operation: str | None = None
    tags: dict[str, Any] = field(default_factory=dict)
    delegate_span: Span | None = None
    sequence: int | None = None

    def set_tag(self, key: str, value: Any) -> None:
        self.tags[key] = value
        if self.delegate_span is not None:
            try:
                self.delegate_span.set_tag(key, value)
            except Exception as exc:
                self.owner._record_error("delegate_set_tag", self.decision_id, exc)


class LoopGridTracer(Tracer):
    """Native Haystack tracer that records LLM-completion evidence.

    Tool request/execution evidence is intentionally *not* inferred from tracing spans. It is
    recorded by Agent hooks around Haystack's actual tool execution boundary so approval-gated
    or rejected calls are not misreported as executed actions.

    A delegate tracer may be supplied so LoopGrid can coexist with an existing Haystack tracer.
    """

    def __init__(self, owner: "LoopGridHaystack", delegate: Tracer | None = None) -> None:
        self._owner = owner
        self._delegate = delegate
        self._current: contextvars.ContextVar[LoopGridSpan | None] = contextvars.ContextVar(
            "loopgrid_haystack_current_span", default=None
        )

    @contextlib.contextmanager
    def trace(
        self,
        operation_name: str,
        tags: dict[str, Any] | None = None,
        parent_span: Span | None = None,
    ) -> Iterator[Span]:
        delegate_parent = parent_span.delegate_span if isinstance(parent_span, LoopGridSpan) else None
        delegate_span: Span | None = None
        error: BaseException | None = None

        with contextlib.ExitStack() as stack:
            if self._delegate is not None:
                try:
                    delegate_span = stack.enter_context(
                        self._delegate.trace(operation_name, tags=tags, parent_span=delegate_parent)
                    )
                except Exception as exc:
                    self._owner._record_error("delegate_trace_start", self._owner.current_decision_id, exc)
                    delegate_span = None

            parent_operation = parent_span.operation_name if isinstance(parent_span, LoopGridSpan) else None
            decision_id = self._owner.current_decision_id
            span = LoopGridSpan(
                owner=self._owner,
                operation_name=operation_name,
                decision_id=decision_id,
                parent_operation=parent_operation,
                tags=_safe_tags(tags),
                delegate_span=delegate_span,
                sequence=self._owner._next_span_sequence(decision_id) if decision_id else None,
            )
            token = self._current.set(span)
            try:
                yield span
            except BaseException as exc:  # preserve application exception behavior
                error = exc
                raise
            finally:
                self._current.reset(token)
                try:
                    self._owner._finish_span(span, error)
                except Exception as exc:
                    self._owner._record_error("trace_finish", span.decision_id, exc)

    def current_span(self) -> Span | None:
        return self._current.get()


class _BeforeRunHook:
    allowed_hook_points = frozenset({"before_run"})

    def __init__(self, owner: "LoopGridHaystack", decision_id: str) -> None:
        self._owner = owner
        self._decision_id = decision_id

    def run(self, state: Any) -> None:
        self._owner._enter_run(state, self._decision_id)

    async def run_async(self, state: Any) -> None:
        self.run(state)


class _BeforeToolHook:
    allowed_hook_points = frozenset({"before_tool"})

    def __init__(self, owner: "LoopGridHaystack", decision_id: str) -> None:
        self._owner = owner
        self._decision_id = decision_id

    def run(self, state: Any) -> None:
        # This hook MUST be registered after ConfirmationHook (or any other hook that can
        # remove/modify tool calls). We record only calls that survive those controls.
        calls = _tool_calls_from_state(state)
        step_identity = _tool_step_identity(state)
        pending_calls = [
            _PendingToolCall(
                call=call,
                call_identity=_call_identity(call),
                occurrence_identity=f"{step_identity}:{ordinal}:{_call_identity(call)}",
                step_identity=step_identity,
                ordinal=ordinal,
            )
            for ordinal, call in enumerate(calls)
        ]
        self._owner._set_pending(state, self._decision_id, pending_calls)
        for pending in pending_calls:
            call = pending.call
            name = str(getattr(call, "tool_name", None) or "tool")
            arguments = getattr(call, "arguments", None)
            payload: dict[str, Any] = {
                "framework": _FRAMEWORK,
                "haystack_target": HAYSTACK_TARGET_VERSION,
                "tool": name,
                "call_id": getattr(call, "id", None),
                "agent_step": pending.step_identity,
                "tool_ordinal": pending.ordinal,
                "input_commitment": _commitment(arguments),
                "source": "agent.before_tool",
            }
            if self._owner.capture_content:
                payload["input"] = arguments
            self._owner._transport.enqueue(
                self._decision_id,
                "tool_requested",
                payload,
                actor_type="tool",
                actor_id=name,
                identity=("before_tool", pending.occurrence_identity),
            )

    async def run_async(self, state: Any) -> None:
        self.run(state)


class _AfterToolHook:
    allowed_hook_points = frozenset({"after_tool"})

    def __init__(self, owner: "LoopGridHaystack", decision_id: str) -> None:
        self._owner = owner
        self._decision_id = decision_id

    def run(self, state: Any) -> None:
        pending_batch = self._owner._take_pending(state, self._decision_id)
        if pending_batch is None or not pending_batch.calls:
            return

        # Only consider results produced by this tool step. This prevents a legitimate later
        # invocation that reuses the same tool-call ID/arguments from binding to an older result.
        results = _tool_results_from_state(state)[pending_batch.result_offset :]
        by_identity: dict[str, list[Any]] = {}
        for result in results:
            origin = getattr(result, "origin", None)
            if origin is not None:
                by_identity.setdefault(_call_identity(origin), []).append(result)

        for pending in pending_batch.calls:
            call = pending.call
            matching_results = by_identity.get(pending.call_identity) or []
            result = matching_results.pop(0) if matching_results else None
            if result is None:
                self._owner._record_error(
                    "after_tool_missing_result",
                    self._decision_id,
                    RuntimeError(f"No Haystack tool result found for {pending.call_identity}"),
                )
                continue

            origin = getattr(result, "origin", None) or call
            name = str(getattr(origin, "tool_name", None) or getattr(call, "tool_name", None) or "tool")
            raw_result = getattr(result, "result", None)
            failed = bool(getattr(result, "error", False))
            payload: dict[str, Any] = {
                "framework": _FRAMEWORK,
                "haystack_target": HAYSTACK_TARGET_VERSION,
                "tool": name,
                "call_id": getattr(origin, "id", None),
                "agent_step": pending.step_identity,
                "tool_ordinal": pending.ordinal,
                "status": "failed" if failed else "completed",
                "error": failed,
                "output_commitment": _commitment(raw_result),
                "source": "agent.after_tool",
            }
            if self._owner.capture_content:
                payload["output"] = raw_result
            self._owner._transport.enqueue(
                self._decision_id,
                "tool_executed",
                payload,
                actor_type="tool",
                actor_id=name,
                identity=("after_tool", pending.occurrence_identity),
            )

    async def run_async(self, state: Any) -> None:
        self.run(state)


class _AfterRunHook:
    allowed_hook_points = frozenset({"after_run"})

    def __init__(self, owner: "LoopGridHaystack", decision_id: str) -> None:
        self._owner = owner
        self._decision_id = decision_id

    def run(self, state: Any) -> None:
        self._owner._leave_run(state, self._decision_id)

    async def run_async(self, state: Any) -> None:
        self.run(state)


class LoopGridHaystack:
    """LoopGrid evidence integration for Haystack 3.3 Agents.

    Native surfaces:
      * Haystack ``Tracer`` / ``Span`` for model-completion evidence.
      * Agent ``before_tool`` / ``after_tool`` hooks for actual tool execution boundaries.
      * Explicit application calls for policy, human review, and observed downstream outcome.
    """

    def __init__(
        self,
        *,
        client: LoopGrid | None = None,
        base_url: str = "http://127.0.0.1:8000",
        api_key: str | None = None,
        workspace_id: str = "default",
        agent_id: str = "haystack-agent",
        capture_content: bool = False,
        default_model_name: str | None = None,
        default_model_provider: str | None = None,
    ) -> None:
        _require_nonempty("agent_id", agent_id)
        self.client = client or LoopGrid(base_url=base_url, api_key=api_key, workspace_id=workspace_id)
        self.workspace_id = workspace_id
        self.agent_id = agent_id
        self.capture_content = bool(capture_content)
        self.default_model_name = default_model_name
        self.default_model_provider = default_model_provider

        self._transport = _EventTransport(self)
        self._errors: list[dict[str, Any]] = []
        self._errors_lock = threading.Lock()
        self._state_lock = threading.RLock()
        self._pending_by_state: dict[int, _PendingToolBatch] = {}
        self._previous_context_by_state: dict[int, tuple[str, ...]] = {}
        self._decision_models: dict[str, dict[str, Any]] = {}
        self._span_seq: dict[str, int] = {}
        self._span_seq_lock = threading.Lock()
        self._decision_stack: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar(
            "loopgrid_haystack_decision_stack", default=()
        )
        self._installed_tracer: LoopGridTracer | None = None

    @property
    def current_decision_id(self) -> str | None:
        stack = self._decision_stack.get()
        return stack[-1] if stack else None

    def _next_span_sequence(self, decision_id: str | None) -> int | None:
        if not decision_id:
            return None
        with self._span_seq_lock:
            next_value = self._span_seq.get(decision_id, 0) + 1
            self._span_seq[decision_id] = next_value
            return next_value

    def make_tracer(self, *, delegate: Tracer | None = None) -> LoopGridTracer:
        return LoopGridTracer(self, delegate=delegate)

    def enable_tracing(self, *, delegate: Tracer | None = None) -> LoopGridTracer:
        """Explicitly enable LoopGrid as Haystack's process-level tracer.

        Haystack exposes a process-level tracing facade. The public ``haystack.tracing.tracer``
        object must not be captured and then used as a delegate: that facade routes to whichever
        tracer is currently enabled, so delegating to it after installing LoopGrid can recurse
        back into LoopGrid.

        To compose LoopGrid with another tracing backend, keep the concrete backend tracer
        instance and pass it explicitly as ``delegate``. Constructing ``LoopGridHaystack`` alone
        never changes global Haystack tracing.
        """

        if delegate is not None and (delegate is self._installed_tracer or isinstance(delegate, LoopGridTracer)):
            raise ValueError("delegate must be a concrete non-LoopGrid Haystack tracer")
        tracer = self.make_tracer(delegate=delegate)
        tracing.enable_tracing(tracer)
        self._installed_tracer = tracer
        return tracer

    def agent_hooks(
        self,
        decision_id: str,
        *,
        before_tool_prefix: Sequence[Any] = (),
        after_tool_suffix: Sequence[Any] = (),
    ) -> dict[str, list[Any]]:
        """Return decision-scoped native Haystack Agent hooks.

        Put ``ConfirmationHook`` (and any other hook that may remove/modify tool calls) in
        ``before_tool_prefix``. LoopGrid's request hook is appended after those controls and
        therefore records only the calls that will actually proceed to Haystack execution.
        """

        _require_nonempty("decision_id", decision_id)
        return {
            "before_run": [_BeforeRunHook(self, decision_id)],
            "before_tool": [*list(before_tool_prefix), _BeforeToolHook(self, decision_id)],
            "after_tool": [_AfterToolHook(self, decision_id), *list(after_tool_suffix)],
            "after_run": [_AfterRunHook(self, decision_id)],
        }

    @contextlib.contextmanager
    def decision_context(self, decision_id: str) -> Iterator[None]:
        """Exception-safe explicit correlation scope for advanced/custom run loops.

        Normal ``Agent`` usage gets correlation from ``before_run``/``after_run`` hooks. This
        context manager is useful when application code needs a guaranteed cleanup boundary even
        if a generator/tool raises before ``after_run`` can execute.
        """

        _require_nonempty("decision_id", decision_id)
        previous = self._decision_stack.get()
        token = self._decision_stack.set((*previous, decision_id))
        try:
            yield
        finally:
            self._decision_stack.reset(token)

    def start_decision(
        self,
        *,
        decision_type: str,
        agent: Mapping[str, Any],
        authority: Mapping[str, Any],
        model: Mapping[str, Any],
        context: Mapping[str, Any],
        proposed_action: Mapping[str, Any],
        policy: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        privacy_mode: str = "redacted",
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        _require_nonempty("decision_type", decision_type)
        _require_nonempty("agent.id", dict(agent).get("id"))
        _require_nonempty("authority", authority)
        _require_nonempty("model.name", dict(model).get("name"))
        _require_nonempty("context", context)
        _require_nonempty("proposed_action", proposed_action)

        md = dict(metadata or {})
        md.update(
            {
                "framework": _FRAMEWORK,
                "integration": "loopgrid-haystack",
                "integration_version": LOOPGRID_HAYSTACK_VERSION,
                "haystack_target": HAYSTACK_TARGET_VERSION,
            }
        )
        result = self.client.record_decision(
            decision_type=decision_type,
            agent=dict(agent),
            authority=dict(authority),
            model=dict(model),
            context=dict(context),
            proposed_action=dict(proposed_action),
            metadata=md,
            privacy_mode=privacy_mode,
            idempotency_key=idempotency_key,
        )
        decision_id = result["decision_id"]
        with self._state_lock:
            self._decision_models[decision_id] = dict(model)
        if policy is not None:
            self.record_policy(decision_id, policy)
        return result

    def record_policy(self, decision_id: str, policy: Mapping[str, Any]) -> Any:
        p = dict(policy)
        decision = p.get("decision")
        if decision not in {"auto_allowed", "human_approval_required", "blocked", "block"}:
            raise ValueError(
                "policy.decision must be auto_allowed, human_approval_required, blocked, or block"
            )
        if not (p.get("policy_id") or p.get("version")):
            raise ValueError("policy provenance requires policy_id or version")
        return self.client.policy_evaluated(
            decision_id,
            p,
            actor_id=str(p.get("policy_id") or "application-policy"),
        )

    def record_human_review(
        self,
        decision_id: str,
        *,
        reviewer: str,
        approved: bool,
        reason: str = "",
    ) -> Any:
        _require_nonempty("reviewer", reviewer)
        # Explicit application facts are written synchronously by the LoopGrid SDK. Drain
        # framework-generated evidence first so a slow transport cannot invert the real lifecycle
        # (for example model_completed arriving after human_approved).
        self.flush()
        if approved:
            return self.client.human_approved(decision_id, reviewer, reason)
        return self.client.human_rejected(decision_id, reviewer, reason)

    def record_outcome(
        self,
        decision_id: str,
        outcome: Mapping[str, Any],
        *,
        observer: str = "application",
    ) -> Any:
        _require_nonempty("outcome", outcome)
        # Preserve ordering with queued model/tool evidence before recording the authoritative
        # downstream observation.
        self.flush()
        return self.client.outcome_observed(decision_id, dict(outcome), actor_id=observer)

    def flush(self) -> None:
        self._transport.flush()

    def shutdown(self) -> None:
        self._transport.shutdown()

    def _enter_run(self, state: Any, decision_id: str) -> None:
        state_id = id(state)
        previous = self._decision_stack.get()
        with self._state_lock:
            self._previous_context_by_state[state_id] = previous
        self._decision_stack.set((*previous, decision_id))

    def _leave_run(self, state: Any, decision_id: str) -> None:
        state_id = id(state)
        with self._state_lock:
            previous = self._previous_context_by_state.pop(state_id, None)
            self._pending_by_state.pop(state_id, None)
        if previous is not None:
            self._decision_stack.set(previous)
        else:
            # Fail closed on correlation rather than retaining a stale decision.
            self._decision_stack.set(())

    def _set_pending(
        self,
        state: Any,
        decision_id: str,
        calls: list[_PendingToolCall],
    ) -> None:
        with self._state_lock:
            self._pending_by_state[id(state)] = _PendingToolBatch(
                decision_id=decision_id,
                calls=tuple(calls),
                result_offset=len(_tool_results_from_state(state)),
            )

    def _take_pending(self, state: Any, decision_id: str) -> _PendingToolBatch | None:
        with self._state_lock:
            stored = self._pending_by_state.pop(id(state), None)
        if stored is None:
            return None
        if stored.decision_id != decision_id:
            self._record_error(
                "pending_decision_mismatch",
                decision_id,
                RuntimeError(f"pending tool calls belong to {stored.decision_id}"),
            )
            return None
        return stored

    def _finish_span(self, span: LoopGridSpan, error: BaseException | None) -> None:
        # Haystack 3.x emits one LLM child span per Agent step. We intentionally ignore
        # tool spans because hook ordering is the authoritative execution boundary.
        if span.decision_id is None:
            return
        if span.operation_name != "haystack.agent.step.llm":
            return

        with self._state_lock:
            model = dict(self._decision_models.get(span.decision_id, {}))
        model_name = _model_name_from_tags(span.tags, model.get("name") or self.default_model_name)
        provider = model.get("provider") or self.default_model_provider
        payload: dict[str, Any] = {
            "framework": _FRAMEWORK,
            "haystack_target": HAYSTACK_TARGET_VERSION,
            "operation": span.operation_name,
            "parent_operation": span.parent_operation,
            "provider": provider,
            "model": model_name,
            "status": "failed" if error is not None else "completed",
            "trace_tags_commitment": _commitment(span.tags),
            "source": "haystack.tracer",
        }
        if error is not None:
            payload["error_type"] = type(error).__name__
        if self.capture_content:
            payload["trace_tags"] = span.tags

        self._transport.enqueue(
            span.decision_id,
            "model_completed",
            payload,
            actor_type="agent",
            actor_id=self.agent_id,
            identity=("llm_span", span.sequence, _commitment(span.tags)["sha256"]),
        )

    def _record_error(self, stage: str, decision_id: str | None, exc: BaseException) -> None:
        with self._errors_lock:
            self._errors.append(
                {
                    "stage": stage,
                    "decision_id": decision_id,
                    "error_type": type(exc).__name__,
                    # Do not retain arbitrary exception text; it can contain user data/secrets.
                    "message": "LoopGrid Haystack integration error",
                }
            )

    @property
    def errors(self) -> tuple[dict[str, Any], ...]:
        with self._errors_lock:
            return tuple(dict(x) for x in self._errors)

    def assert_healthy(self) -> None:
        errors = self.errors
        if errors:
            first = errors[0]
            raise RuntimeError(
                f"LoopGrid Haystack transport/integration failed ({len(errors)} error(s)); "
                f"first={first['stage']}:{first['error_type']}"
            )
