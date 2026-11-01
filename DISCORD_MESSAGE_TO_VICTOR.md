# Discord message — to Victor Romero

**Send this to Victor (Discord DM).** The full detail stays in
`AI-267_STAGE_A1_RESULTS.md` (the new round) and `AI-267_STAGE_A_RESULTS.md` (the first
round) — this message only points at them and asks the three questions that are still open.

---

Hi Victor — Stage A.1 is done and scored, so here are the updated results ahead of the
roadmap discussion.

**📄 Report file:** `~/Desktop/oss-contrib/smallai/AI-267_STAGE_A1_RESULTS.md`
It's also going into `reyyishreyas/smallai` under that name — say the word and I'll push it
so you can read it from your side.

**What Stage A.1 is:** a second, separately pre-registered round (`EXPERIMENT_A1.md`, written
and committed *before* any training run). Same protocol, same policies, fresh seeds —
**5 fresh master seeds × 10,000 test scenarios each** (50,000 paired scenarios, versus 1,000
in Stage A). Stage A itself is untouched and still reported separately. The round was
designed to answer one question precisely: *is the accuracy gain real, or was Stage A's
number a lucky draw?*

Short version:

- **The accuracy gain is not established — and now that's a measured answer, not a vague
  one.** Journey-Q beats the decision tree by **+0.170 pp** across the 50,000 paired
  scenarios, but across five independent retrainings the 95% interval is **[−0.127, +0.467]**
  (t = 1.59). Two of the five replicates are negative. The per-replicate spread (sd 0.240 pp)
  is bigger than the effect, so five replicates still can't call it.
- **Two levels of noise make this worth knowing.** On the scenario-level view alone the
  delta is significant (pooled CI [+0.070, +0.268], McNemar p = 0.0010) — but that treats
  one trained policy as fixed. The replicate-level question ("does it still win if we
  retrain?") is the one the ticket is actually asking, and that one comes out negative.
- **The advantage is real where the capture is clean.** The protocol's noise sweep, now
  reported with intervals: at **σ = 0.04 the gain is +0.508 pp with both intervals clear of
  zero** (seed-level [+0.373, +0.643], t = 10.5); at the primary σ = 0.08 it's +0.170 pp;
  at σ = 0.12 it's +0.040 pp. So the answer to "does journey learning help?" is *it depends
  on how noisy the capture is* — and on a phone capture in good light, the relevant σ may
  well be nearer 0.04 than 0.08.
- **The two metrics that map to money are decisive, not marginal.**
  - **Invalid-verdict rate: 1.230% vs 7.874% — 6.4× lower**, paired diff +6.644 pp
    [+6.396, +6.890], 3,769 vs 447 discordant scenarios, McNemar p < 1e-300. (Stage A
    reported 22×; that was one seed on 1,000 scenarios. 6.4× is the honest five-seed
    number — still decisive.)
  - **Mean measurements per decision: 4.592 vs 6.000** — 1.408 fewer [1.392, 1.423], 23.5%
    fewer signal acquisitions. The tree reads all six by construction; the policy
    early-exits and only re-measures when the evidence is thin.
- **The ablations line up with the mechanism** (pooled over the five seeds): removing
  re-check gives 97.142% accuracy at a 6.132% invalid rate; removing journey learning gives
  98.050% at 1.008%. Both halves of the design are pulling.
- **The one candidate I thought was promising was rejected.** Halving the consistency
  reward term and letting a suspect reading support "retake" came out at **+0.034 pp
  [−0.062, +0.132], p = 0.5161** — nothing — and it nearly doubled the invalid-verdict rate
  (2.182%). It had looked like a win on the seeds it was selected on (+0.375 ± 0.137) and
  didn't reproduce on fresh ones (−0.100 ± 0.134). The shipped reward is unchanged.
- **Continuity check, same policies, Stage A's sample size:** the first 1,000 scenarios of
  each seed give +0.040 pp [−0.260, +0.340], p = 0.8939. Stage A's split size simply
  cannot resolve a sub-percentage-point effect in either direction — which is why this
  round was needed.
- Also in the report: the full three-sigma sweep with intervals, calibration agreeing
  between isotonic and temperature scaling, ~3,100 stored Q-entries against the 100K
  budget, and the integrity trail (Stage A re-run and diffed leaf-by-leaf: 23,947 measured
  leaves, identical apart from wall-clock fields).

**No pass/fail claim is made against the ticket's threshold.** The "X" in "journey learning ≥
X% over shortcut" has still never been supplied, and per §2 any X supplied now is post-hoc
relative to the pre-registration.

**Three things I'd like to close before the Kotlin port:**

1. **Is `path` part of the contract?** You named verdict, reasons and confidence, but
   `Decision` also carries `path` — the ordered sequence of checks performed. It matters
   because that's what tells the app whether a re-check is needed; without it in the
   contract the 4.592-vs-6.000 win can't be realised on device (the app would have to
   re-measure everything, or never re-check). Is `path` part of the stable shape, internal,
   or optional?
2. **Binary or ternary action space?** Still open — `EXPERIMENT.md` §11 pre-declares the
   binary fallback, but the `review` class only has 23 outcomes in the calibration split,
   which is thin enough to matter for the confidence map. A ruling either way is fine; I'd
   rather know before Stage B.
3. **X** — and, given the sweep above, it's worth deciding *at which noise level* X should
   apply, since the answer differs by more than tenfold between σ = 0.04 and σ = 0.12.
   Noting explicitly that any X supplied now is post-hoc relative to the pre-registration,
   so the report makes no threshold claim.

**On OCR — answering your question directly:** no, SmallAI doesn't have OCR today, and I
don't think it should try to own it. On-device OCR is solved: ML Kit's bundled text
recognition (`com.google.mlkit:text-recognition`) links the model at build time, ~4 MB,
fully offline with no Play Services dependency. That's the right primitive for document
capture, and it stays out of Stage A entirely as you said.

**On the field officer app** — I had a look and the seam is clean. It's Compose
Multiplatform, and in `feature/client/.../ClientProfileEditViewModel.kt` the flow is
`crop()` → `uploadImage()`, with the upload happening immediately after the crop. A
quality gate belongs exactly between those two: verify, and on `retake` keep the officer
in the flow instead of uploading a bad ID. Since the app is KMP with
`kotlinx.serialization` already in `commonMain`, the policy ports as a pure-Kotlin module
reading the Q-table JSON — no Android APIs, no inference runtime, ~3 KB artifact. The CV
signal extraction is the only platform-specific piece, which is exactly what Stage B is
for.

On the GSoC framing: a natural split would be the Stage B spike (real captures, empirical
signal distributions) as the GSoC task, with the policy port as the follow-on. Happy to
draft the task description if that shape works.

Will keep the current experiments as scoped and won't assume timelines.

---

## Notes (not part of the message)

- Don't post any of this as a PR comment without asking Victor first.
- PR #634 is deliberately kept out of the message body. It's the SmallAI *reasoning* layer
  (llama.rn / on-device LLM inference), which is a different concern from this
  *perception-gate* layer — mention it only if Victor asks about on-device inference.
- The report file lives on this machine; the copy in `reyyishreyas/smallai` needs a push
  before Victor can open it. Ask before pushing, or push and tell him.
- `smallai` is private and Apache-2.0. `mcp-mifosx` is MPL-2.0 and the field officer app
  is under a different license — worth aligning before anything is made public or
  contributed upstream.
- If the Stage A message has already gone out, the Stage A.1 half is written to stand alone
  as a follow-up: it re-states the numbers rather than referring back to the earlier
  message.
- Stage A.1's numbers come from `reports/poc_a1.json` (machine-readable source for every
  figure); Stage A's are in `reports/poc.json`. Both are separately reported — different
  seeds, different split size, different analysis.
