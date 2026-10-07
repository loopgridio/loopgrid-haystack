# Integration contract

`loopgrid-haystack` is evidence infrastructure, not a Haystack execution engine.

## Evidence boundaries

- `decision_created`: explicit application call to `start_decision`.
- `policy_evaluated`: explicit application policy only.
- `model_completed`: completion of Haystack's native `haystack.agent.step.llm` span.
- `human_approved` / `human_rejected`: explicit authenticated application fact only.
- `tool_requested`: `before_tool` after prior confirmation/modification/rejection hooks have run.
- `tool_executed`: `after_tool` from the matching Haystack `ToolCallResult`.
- `outcome_observed`: explicit observation from the application or authoritative downstream system.

## Non-inference rules

The integration must never:

- treat a proposed tool call as executed;
- record a rejected call as `tool_requested`;
- infer approval or reviewer identity from `ConfirmationHook` success;
- infer a real-world outcome from a local tool return;
- claim that cryptographic verification proves business correctness.

## Ordering rule

Any `before_tool` hook that can remove or modify pending tool calls (especially `ConfirmationHook`) must run before the LoopGrid request hook. Use:

```python
hooks=loopgrid.agent_hooks(decision_id, before_tool_prefix=[confirmation_hook])
```

Explicit application facts (`record_human_review`, `record_outcome`) drain previously queued
framework evidence before writing synchronously, so the recorded lifecycle cannot be inverted by
transport timing.

## Tool correlation rule

Tool request/execution identity is decision-scoped and occurrence-scoped. A retry of the same
Haystack tool step is idempotent, while a legitimate later Agent step receives a distinct identity
even if the provider reuses a tool-call ID or supplies no ID. Result matching is limited to the
current tool step so earlier results cannot be rebound to later calls.
