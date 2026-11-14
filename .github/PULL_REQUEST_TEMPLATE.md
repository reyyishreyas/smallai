## What changed

<!-- One or two sentences. Link the issue if there is one. -->

## Evidence

<!-- For behavior changes: commands run and numbers before/after. -->

- [ ] `make check` passes (ruff + full test suite)
- [ ] New or changed behavior is covered by a test
- [ ] Docs updated in this PR if behavior changed:
  - [ ] `README.md`
  - [ ] `reports/STAGE_B_RESULTS.md`
  - [ ] `docs/ARCHITECTURE.md`
- [ ] No change to the `RETAKE | REVIEW | PASS` contract (or the change was
      explicitly agreed and documented)

## Invariants

- [ ] No LLM / neural network / high-level ML library in the inference path
- [ ] Deterministic (fixed seeds, same result on re-run)
- [ ] Offline, CPU-only, no new runtime dependency without justification
