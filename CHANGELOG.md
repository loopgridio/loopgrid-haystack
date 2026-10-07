# Changelog

## 0.1.0 — release candidate 3

- Native Haystack 3.3 `Tracer` / `Span` model evidence.
- Decision-scoped Agent lifecycle hooks.
- Approval-safe `before_tool` ordering helper.
- Tool request/result commitments with explicit privacy opt-in for raw content.
- Explicit policy, human review, and observed outcome helpers.
- Parallel call correlation, deterministic tool idempotency, concurrency isolation, and transport health reporting.
- Deterministic Haystack Agent + LoopGrid Core E2E examples.

### RC3 hardening

- Tool occurrence identity now includes the Haystack Agent step and per-step ordinal, preventing
  legitimate repeated calls from collapsing when tool-call IDs are reused or omitted.
- Tool-result correlation is scoped to results created by the current tool step.
- Explicit human-review and observed-outcome writes drain queued framework evidence first so
  lifecycle ordering cannot invert under slow transport.
- Added native `ConfirmationHook` rejection and modification runtime regressions.
- Added repeated-call/no-ID idempotency regressions and explicit ordering regressions.
- Added a real `Agent.run_async()` runtime regression with an async tool.
- Corrected README release-candidate and tracer-delegation wording.

### RC2 fix retained

- Fixed recursion when the public `haystack.tracing.tracer` facade was captured and reused as a delegate.
- Tracer composition requires an explicit concrete backend tracer instance.
- Added runtime cleanup and explicit-delegate regression coverage.
