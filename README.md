# LoopGrid for Haystack

Native Haystack 3.3 Agent hooks + tracing integration for LoopGrid, the evidence plane for AI agents.

`loopgrid-haystack` maps Haystack's real Agent lifecycle into LoopGrid decision evidence without changing the LoopGrid Core contract.

## Evidence contract

```text
application starts consequential decision
        ↓
Haystack Agent LLM span completes
        ↓
LoopGrid model_completed
        ↓
explicit application policy / human review
        ↓
Haystack ConfirmationHook / other controls
        ↓
LoopGrid before_tool observes the surviving tool call
        ↓
LoopGrid tool_requested
        ↓
actual Haystack tool execution
        ↓
Haystack after_tool
        ↓
LoopGrid tool_executed
        ↓
application explicitly observes downstream outcome
        ↓
LoopGrid outcome_observed
        ↓
evidence_complete → cryptographic verification
```

LoopGrid **does not** execute tools, invent delegated authority, invent policy, infer a reviewer, treat confirmation as authorization by itself, or infer a real-world outcome from a tool return.

## Compatibility

- Python 3.10–3.14
- `haystack-ai` 3.3.x
- `loopgrid` 0.8.x
- LoopGrid Core validation target: `0.8.1-design-partner`

## Install

```bash
pip install loopgrid-haystack
```

## Quickstart

```python
from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage
from loopgrid_haystack import LoopGridHaystack

loopgrid = LoopGridHaystack(
    base_url="http://127.0.0.1:8000",
    workspace_id="default",
    agent_id="support-agent",
)

# Explicit: Haystack never enables tracing automatically. If the application already owns
# a concrete Haystack tracer backend, pass that backend explicitly as delegate=...
loopgrid.enable_tracing()

decision = loopgrid.start_decision(
    decision_type="customer_refund",
    agent={"id": "support-agent", "version": "1"},
    authority={"acting_for": "Example Store", "scope": ["refund:create"], "limit_usd": 100},
    model={"provider": "openai", "name": "gpt-5"},
    context={"prompt_version": "support-v1"},
    proposed_action={"tool": "refund.create", "amount": 25, "currency": "USD"},
    policy={
        "policy_id": "refund-policy",
        "version": "1",
        "decision": "auto_allowed",
        "reason": "Within delegated threshold",
    },
)

agent = Agent(
    chat_generator=...,
    tools=[...],
    hooks=loopgrid.agent_hooks(decision["decision_id"]),
)

result = agent.run(messages=[ChatMessage.from_user("Handle this duplicate charge.")])
loopgrid.flush()
loopgrid.assert_healthy()

# Only after the application observes the authoritative downstream business result:
loopgrid.record_outcome(
    decision["decision_id"],
    {"status": "succeeded", "external_reference": "refund_123"},
    observer="billing-webhook",
)
```

## Human approval / ConfirmationHook

Haystack's `ConfirmationHook` remains the execution-control mechanism. LoopGrid records evidence about an application-authenticated reviewer; it does not replace Haystack confirmation or application authorization.

**Ordering is critical:** pass confirmation/modification hooks through `before_tool_prefix`. LoopGrid's request hook is appended after them, so a rejected call cannot become `tool_requested` evidence and a modified call is recorded with the final model-visible parameters that survive confirmation.

```python
confirmation_hook = ConfirmationHook(confirmation_strategies={...})

agent = Agent(
    chat_generator=...,
    tools=[...],
    hooks=loopgrid.agent_hooks(
        decision_id,
        before_tool_prefix=[confirmation_hook],
    ),
)
```

Your authenticated approval UI/strategy should explicitly record the reviewer when the human makes the decision:

```python
loopgrid.record_human_review(
    decision_id,
    reviewer="reviewer@example.com",
    approved=True,
    reason="Reviewed refund evidence",
)
```

LoopGrid never derives reviewer identity merely from the fact that a Haystack tool call survived `ConfirmationHook`.

`record_human_review()` drains earlier queued framework evidence before writing the explicit
review event. This preserves the real lifecycle ordering even when the LoopGrid transport is
temporarily slower than the application thread.

## Native integration surfaces

The integration uses two different Haystack-native boundaries for two different facts:

- `Tracer` / `Span`: `haystack.agent.step.llm` completion → `model_completed`.
- Agent `before_tool` / `after_tool`: actual surviving tool request and resulting tool message → `tool_requested` / `tool_executed`.

Haystack tool spans are deliberately ignored for action evidence. The hooks are authoritative because they run after confirmation/modification and around actual Agent-owned tool execution.

## Existing Haystack tracing backends

Haystack exposes one process-level tracing facade. `enable_tracing()` is explicit:

```python
loopgrid.enable_tracing()
```

Do **not** capture `haystack.tracing.tracer` and pass that facade back as a delegate. The facade routes to the currently enabled tracer; after LoopGrid is installed that would route back into LoopGrid and recurse.

If your application already owns a concrete Haystack tracer instance, compose it explicitly:

```python
existing_backend = MyConcreteHaystackTracer(...)
loopgrid.enable_tracing(delegate=existing_backend)
```

LoopGrid delegates spans directly to that concrete backend while recording its own evidence. Constructing `LoopGridHaystack` by itself never changes global Haystack tracing.

## Privacy

LoopGrid defaults to `capture_content=False`.

Model span tags and tool inputs/outputs are represented by SHA-256 commitments rather than raw content. Haystack itself also disables content tracing by default. Do not enable `HAYSTACK_CONTENT_TRACING_ENABLED=true` unless your application intentionally permits content export.

If you explicitly set `capture_content=True`, LoopGrid will include the tool input/result and captured Haystack trace tags in evidence payloads.

## Failure behavior

Tracer/hook transport is queued and does not raise into the Haystack Agent execution path. Failures are retained and surfaced explicitly:

```python
loopgrid.flush()
loopgrid.assert_healthy()
```

A successful Agent run is not evidence that LoopGrid transport succeeded unless `assert_healthy()` passes.

## Pre-release validation gates

The release candidate is **not a public release**. Required before `v0.1.0`:

- exact `haystack-ai==3.3.0` runtime install
- semantic/unit tests
- real deterministic Haystack Agent runtime test
- real LoopGrid Core E2E
- real native `ConfirmationHook` approval ordering E2E
- rejection and modification regression tests
- parallel tool-call correlation test
- repeated-call idempotency across later Agent steps, including missing/reused call IDs
- queued-event ordering before explicit human review/outcome writes
- real `Agent.run_async()` runtime regression
- privacy regression
- concurrency isolation
- transport-failure observability
- `evidence_complete`, applicable coverage 100%, `verify.valid=true`, failures `[]`
- wheel/sdist build + `twine check`
- fresh wheel install/import test
- GitHub CI, exact tested commit tag/release
- PyPI Trusted Publishing + fresh public install
- website/docs after public install
- official Haystack Integrations PR after all release gates pass

## License

Apache-2.0.
