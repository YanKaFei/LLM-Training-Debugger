# W7 — REGRESSION

```
full suite   531 passing, 1 skipped
smoke        8/8 PASS
quickstart   PASS
four-verdict CLI  PASS
cross-process restart  PASS
editable install  PASS
non-editable install  PASS
```

New coverage added this round:

* witness ON/OFF produce identical evidence
* `step invoked` but `parameters_changed` stays `None`
* `step skipped` with an independent downstream NaN
* optimizer invoked with an independent NaN (C3)
* finite tensor with an overflowing norm (`norm_reduction_overflow`)
* finite parameters with a downstream overflow (C4)
* gradient accumulation step alignment
* multiple-optimizer isolation
* missing witness ⇒ `UNKNOWN`
* no ground-truth leakage
* relation readback after restart
* relation scorer denominator integrity

`OptimizerWitness` remains **default disabled**, asserted by test.
