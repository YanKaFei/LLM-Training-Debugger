# OPTIMIZER WITNESS RFC

**Design and cost analysis only. No hot-path instrumentation was added in v0.1.4.**

---

## 1. The problem

v0.1.4's association rule requires the propagation to be observable at **every** stage:

```
gradient -> optimizer update -> parameters -> loss
```

The midstream stage is observed today through `param.global_norm`. That is enough for the
corpus in this repository, but it is an **inference**, not a measurement of the update
boundary itself. Two gaps remain:

1. `param.global_norm` is a *post-hoc norm*, not a statement that `optimizer.step()` ran.
   A skipped update and an update that ran but did not overflow look **identical**.
2. The delay bound `PROPAGATION_DELAY_STEPS = 2` assumes `one training step = one
   optimizer update`. Gradient accumulation breaks that assumption directly.

## 2. Three distinct facts, deliberately not merged

The brief is explicit, and it is correct: these are not one boolean.

| Candidate | Fact asserted | What it does NOT say |
|---|---|---|
| `optimizer.step_attempted` | the call was made | whether it changed anything |
| `optimizer.step_skipped` | the call was made and intentionally not applied (accumulation, AMP skipped step, `lr=0`) | whether the parameters are sound |
| `post_update_nonfinite_detected` | after the update, a parameter (or its norm) was non-finite | whether the update *caused* it |

Collapsing these into a single `optimizer.update_overflow` would lose the distinction
between "no update happened" and "an update happened and broke something" — which is
exactly the distinction the adversarial corpus needed.

## 3. Can existing APIs expose it?

| Question | Answer |
|---|---|
| Can existing APIs expose the signal? | **Partially.** `torch.optim.Optimizer` exposes no hook for "did step() apply". PyTorch 2.x offers `register_step_post_hook`, but it is not available on all supported optimisers and would need a version guard. |
| Does it require parameter scanning? | **Yes, for `post_update_nonfinite_detected`** — unless the norm already computed for `param.global_norm` is reused, in which case no extra scan is needed. |
| Does it require a GPU synchronization? | **Possibly.** Any `.isnan().any()` reduction forces a device sync. That is the dominant cost risk. |
| Computational complexity? | `step_attempted` / `step_skipped`: **O(1)** per step. `post_update_nonfinite_detected` reusing the existing norm: **O(1)**. A fresh per-parameter scan: **O(parameter_count)**, not O(1). |
| Additional memory? | O(1) for the booleans. A saved finite parameter copy — needed to attribute the change — is **O(parameter_count)** and is not acceptable on the hot path. |
| Under gradient accumulation? | `step_skipped` is **the normal case** on most micro-steps. A rule that treats "no update" as a gap would break on every accumulation schedule. The witness must be reported per *optimizer* step, and the training step index must carry the distinction. |
| Under mixed precision? | `GradScaler` skips the step on inf/nan gradients. That produces `step_skipped` **with** a non-finite gradient — observationally identical to A1/A2/A8Y in the adversarial corpus. This is the strongest argument for the witness: today the system cannot tell a scaled-skip from a real overflow. |
| Can it distinguish propagation from independent injection? | **Yes, and this is its main value.** `post_update_nonfinite_detected` immediately after an `applied` step, with the downstream failure following, is direct evidence of propagation. An independent injected fault has no such witness. |

## 4. Recommendation

```
RECOMMENDED:  post_update_nonfinite_detected, reusing the parameter norm already
              computed for param.global_norm, PLUS a per-optimizer-step boolean
              `step_applied`.

NOT RECOMMENDED:  a fresh O(parameter_count) scan.
NOT ACCEPTABLE:   claiming O(1) per step for anything that scans parameters.
```

Estimated overhead if implemented as described: **one boolean and one reuse of an
existing reduction per optimizer step — no new reduction, no new sync.** The honest
statement is that this has **not been measured**, because implementing it requires
touching the training hot path.

**AUTHORISATION REQUIRED.** Adding hot-path data collection was explicitly out of scope
for v0.1.4 and remains so. The correct next step is a decision by the project owner, not
an autonomous change.

## 5. Known limitations of the proposal

* A version guard is needed for `register_step_post_hook`; on optimisers without it the
  witness gracefully degrades to `UNKNOWN` rather than being fabricated.
* Under gradient accumulation, the rule must reason in *optimizer* steps, so the training
  step index and the update index must both be present. This is a schema addition.
* The witness resolves the "did the update apply" question. It does **not** by itself
  establish that the update caused the downstream failure — it makes the propagation
  *observable*, which is a stronger claim than v0.1.3's inference and still weaker than
  causation.
