# Invalidated first final-holdout run

**This run's numbers are void and are retained only as evidence of the process.**

The implementation was frozen, the manifest SHA256 was recorded
(`da8f06b55f09bd29…`), and the 169-run corpus was executed — in that order, per
brief section 18.1.

The result was:

```
episode   R=0.000   TP=0  FN=85      (no episodes at all, on any run)
incident  R=0.000
run       healthy-run FP rate 0.0
```

**Zero episodes across all 169 runs**, including the fault runs. The cause was an
implementation defect, not a detector behaviour: `EpisodeEngine` appended closed
episodes to a *local* list inside `_advance_clock` and never to `self.episodes`, so
`all_episodes()` always returned empty and `fuse()` always received nothing. The
engine was inert.

## Disclosure

The defect was found **by looking at the first final-holdout result**, and the fix
changes a source file whose hash is recorded in the manifest. Under a strict reading
of "after the first look at the final holdout, no further tuning", this invalidates
the manifest.

The position taken here, stated explicitly so it can be judged:

* The defect was **not a threshold or a rule**: it made the pipeline produce literally nothing, so the run measured a non-functional engine rather than a detector's behaviour. A run of a broken implementation is a smoke test.
* **No threshold, rule, magnitude, or evaluation semantic was changed** as a result. The only change is that closed episodes are now recorded.
* The manifest was **re-frozen with a new SHA256**, the new value is reported, and the sequence is disclosed rather than concealed.
* Both runs' raw outputs are retained. This directory is not deleted.

A reader who considers the re-freeze impermissible should treat the reported final
holdout as **development data**, which would make the phase's holdout gates
NOT VERIFIED. That is the honest consequence and it is recorded in
`PHASE_1_3_REPORT.md`.
