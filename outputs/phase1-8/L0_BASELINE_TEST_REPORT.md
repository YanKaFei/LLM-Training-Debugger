# L0 — BASELINE TEST REPORT

**Status: COMPLETE.** The contamination failure is pinned as a regression **before**
the feature was changed.

```
before:  390
added:    14
total:   404 passing, 1 skipped
```

---

## 1. The failure, written as a test

`test_sustained_scaling_does_not_self_normalise` asserts the property Phase 1.7's
680-run corpus showed was missing:

```
sustained x5 fault -> ratio must stay at 5 for the whole episode
```

`test_trailing_median_would_have_decayed` asserts the **defect itself**, so it cannot
silently return:

```
the old feature, under a sustained fault, reports ratio ~ 1.0
```

`test_contaminated_rolling_median_is_pinned_as_a_regression` shows the arithmetic
directly: once the rolling window lies entirely inside the fault, `value / median` is
1.0 while the truth is 5.0. **That is why no threshold could ever have worked**, and
it is now a failing-if-reintroduced test rather than a paragraph in a report.

## 2. State semantics under test

| Requirement | Test |
|---|---|
| learns while NORMAL | `test_baseline_learns_while_normal` |
| freezes on SUSPECT | `test_baseline_freezes_on_suspect` |
| stays frozen on CONFIRMED | `test_baseline_stays_frozen_when_confirmed` |
| RESUMING after a rejected candidate | `test_baseline_resumes_after_a_rejected_candidate` |
| a false suspicion does not poison the baseline | `test_a_false_suspicion_does_not_poison_the_baseline` |
| RECOVERING does not relearn poisoned values | `test_recovering_does_not_relearn_poisoned_values` |
| warm-up abstains rather than classifying | `test_warmup_abstains_rather_than_classifying` |
| no aggregate before the baseline is ready | `test_joint_feature_reports_nothing_before_the_baseline_is_ready` |

The freeze-on-**suspicion** test is the load-bearing one. Freezing on *confirmation*
would be too late: by then the fault has already contaminated any rolling reference.
The reference must be captured at the first hint of deviation, and
`freeze_at_ratio = 1.3` is deliberately far below any confirmation threshold because
its only job is to catch a clean reference early.

## 3. Two semantics worth stating

**An unready baseline returns `None`, not a number.** An unready reference means the
statistical verdict is `INSUFFICIENT_EVIDENCE`; classifying during warm-up is what the
brief's L2.4 forbids. `ratio()` returns `None` so there is no number to mis-classify.

**Resume requires quiet steps, not just a state change.** After a rejected candidate
the baseline resumes only after `resume_after_steps` consecutive quiet observations,
so a transient excursion cannot permanently redefine what "normal" means.
