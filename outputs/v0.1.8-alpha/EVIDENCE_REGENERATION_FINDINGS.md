# G — Regenerating the evidence: what it found

The corpora were regenerated from a clean revision so that committed evidence would
correspond to committed code. That part worked. **It also revealed that the repository's
headline numbers cannot be reproduced from the repository**, which is a larger finding than
the drift it set out to fix.

```
freeze point   c5351a2   working tree clean
regenerated    9 scripts, all exit 0, ~5 minutes total
artifacts      7 rewritten, every one recording sha=c5351a277e dirty=False
```

## 1. What was fixed

**Provenance.** No committed artifact said what produced it. `write_artifact` now records
`code_sha`, `code_dirty` and `generated_at`, so the next drift is visible without a manual
diff. `code_dirty` is scoped to `src`/`tests`/`pyproject.toml`: the first version asked
about the whole tree, and every script after the first in a sequence recorded `dirty=True`
because the *earlier scripts had rewritten the evidence*. The code was identical; only the
evidence had moved.

**A scoring script exists.** The numbers in the README were produced by an ad-hoc command
that left nothing behind. `tests/experiments/level_scoring.py` now scores a corpus from its
store, records the store and ground-truth paths it used, and fails closed.

## 2. What it found: the headline numbers are not reproducible

The README states RUN precision 0.947 / recall 0.500, EPISODE precision 0.326, INCIDENT
precision 0.720, and `missed_primary_faults 18`. **None of these can be re-derived**, for
four independent reasons:

1. **No `primary_fault_type` in any ground truth.** Both corpora write `scenario` labels
   (`grad_extreme`, `loss_spike`, `M1_gradient_then_nonfinite`, ...) and neither carries the
   field the scorer reads. `validate_scoring_inputs` refuses — correctly — with
   `SCORING_INVALID / MISSING_PRIMARY_FAULT_TYPE`.
2. **No `acceptable_window` in the development ground truth.** Without it every correct
   detection is classified `UNRELATED_FALSE_POSITIVE` and precision reads 0.0, which is the
   silent-zero failure the code already documents twice.
3. **The mapping table does not cover the labels.** `FAULT_CLASS_TO_ANOMALY` contains
   `F1_loss_spike`, `A1_single_huge_harmless_batch` and similar. The corpora use
   `grad_extreme`, `grad_large`, `grad_mid`, `grad_small`, `loss_spike`, `nan_loss` and
   `M1_`…`M5_`. **The two sets do not intersect.**
4. **The old scoring artifacts record no source at all** — no store path, no ground-truth
   path, no command. `level_scoring_v012.json` contains only `status`, `validity`, `levels`
   and `unit_accounting`.

So the previous scoring used an enriched input that was never checked in. `level_scoring.py`
fails closed rather than inventing the missing fields, which is why running it now produces
`SCORING_INVALID` instead of a number.

## 3. What is not being done about it

**The missing fields are not being guessed.** Deriving `grad_small` → `GRADIENT_EXPLOSION`
is mechanical and would probably be right. Deriving an `acceptable_window` for the
development corpus is not: it encodes how many steps a detector is allowed to take, which is
a measurement claim, and choosing one now to make a number appear is the same class of
mistake as D-015 and D-018 — a plausible figure standing in for an absent one.

The numbers in the README are therefore **not withdrawn but downgraded**: they are recorded
as having been measured, by an input that no longer exists, and they **cannot be re-derived
from this repository as it stands**. That statement belongs in the README next to them.

## 4. What would make them reproducible

Either of these, and the second is more useful:

* **Reconstruct the mapping and windows explicitly**, as evaluation-only ground truth, and
  re-run. The resulting figures would be **new measurements**, not reproductions of the old
  ones, and must be labelled as such — they would not confirm the old numbers even if they
  matched.
* **Make the corpora emit evaluation-ready ground truth** (`primary_fault_type`,
  `acceptable_window`) at generation time, so the fields travel with the evidence instead of
  being bolted on later by a command nobody kept. The mixed corpus already computes both; it
  simply does not write `primary_fault_type`.

Both ground-truth writers also bypass `write_artifact`, which is why they carry no
provenance. That is a smaller version of the same problem.

## 5. Status

```
REGENERATED   all 9 corpora, from c5351a2, provenance recorded
REPRODUCIBLE  the corpora themselves
NOT REPRODUCIBLE  the scoring, until the evaluation fields exist
```
