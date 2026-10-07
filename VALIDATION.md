# Validation record — RC3

Target matrix:

- `loopgrid-haystack`: 0.1.0 RC3
- Haystack: `haystack-ai==3.3.0`
- LoopGrid Python SDK: `loopgrid==0.8.0`
- LoopGrid Core: `0.8.1-design-partner`
- Python: 3.10–3.14

## Required local checks

```powershell
python -m pytest -v
python -m compileall -q src examples tests
python -m build
python -m twine check dist\*
```

## Real Core release gate

Both deterministic E2Es must finish with:

```text
lifecycle = evidence_complete
coverage.score = 100
coverage.complete = true
verification.valid = true
verification.failures = []
```

`examples/human_approval_e2e.py` must additionally prove:

```text
human_approved < tool_requested < tool_executed
```

No public-release claim should be made from RC3 until these checks have actually passed.

## RC2 regression focus retained

RC1 exposed a real Haystack tracing-composition edge case: the public `haystack.tracing.tracer`
object is a process-level facade and cannot safely be captured as a delegate before replacing the
enabled tracer. Doing so can recurse into the newly installed LoopGrid tracer.

RC2 removed automatic/global-facade delegation. Coexistence remains supported only through an
explicitly supplied concrete Haystack tracer instance. The real-runtime test disables any prior
tracing state, installs LoopGrid explicitly, and restores disabled tracing afterward.

## RC3 hardening focus

RC3 does not add product scope. It closes release-quality edge cases identified during the
pre-package audit:

- tool event identity includes the Haystack Agent step + per-step ordinal, so a legitimate later
  call is not collapsed merely because a provider reuses the same call ID or omits IDs;
- result correlation is scoped to the current tool step instead of searching all earlier results;
- explicit `record_human_review()` and `record_outcome()` drain queued framework evidence first,
  preserving lifecycle order under slow transport;
- native `ConfirmationHook` rejection regression proves rejected calls produce neither
  `tool_requested` nor `tool_executed` evidence;
- native `ConfirmationHook` modification regression proves LoopGrid records the surviving modified
  parameters that Haystack actually executes;
- a real `Agent.run_async()` runtime regression exercises async hooks and an async tool;
- README tracing language now matches the RC2 concrete-delegate rule.

The full local suite is expected to contain 29 tests after this hardening pass. Real-Core E2Es are
rerun only after all local tests and compile checks pass.
