# Release checklist

1. Run the full 29-test RC3 suite against exact `haystack-ai==3.3.0`.
2. Confirm native runtime coverage includes sync Agent, async Agent, ConfirmationHook rejection,
   ConfirmationHook modification, repeated-call idempotency, and slow-transport ordering.
3. Start LoopGrid Core `0.8.1-design-partner` and run both E2Es.
4. Confirm evidence-complete / 100% applicable coverage / valid verification / no failures.
5. Build wheel and sdist; run `twine check`.
6. Inspect artifacts; install wheel in a clean venv and verify imports.
7. Push to `github.com/loopgridio/loopgrid-haystack`.
8. Require green CI on the exact release commit.
9. Tag that exact commit `v0.1.0` and create GitHub release.
10. Publish through PyPI Trusted Publishing.
11. Fresh-install the public package and repeat runtime + Core E2E.
12. Publish website integration page.
13. Submit official Haystack Integrations directory PR.
