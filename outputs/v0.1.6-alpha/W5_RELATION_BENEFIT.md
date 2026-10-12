# W5 — RELATION BENEFIT

## New WITNESS CHALLENGE corpus (54 runs)

```
                    TP   FP   FN   TN    precision   recall   coverage   false-assoc
WITNESS_DISABLED     4    0   20   30      1.000     0.167     0.167       0.000
WITNESS_ENABLED      4    0   20   30      1.000     0.167     0.167       0.000

positive coverage 0.167   negative rejection 1.000   false association 0.000
true abstention: C3 18 order-only on negative cases
```

## Paired difference

```
recovered true associations : 0
new false associations      : 0
lost true associations      : 0
correctly rejected delta    : 0
changed abstention decisions: 0
```

## ORIGINAL 50-run and ADVERSARIAL 45-run corpora

Both were scored in v0.1.5 after the W0 artifact fix and are **unchanged** here, because
the witness clause is inert on them for the same reason: no observed antecedent falls
within the propagation window.

```
ORIGINAL 50-run    TP 0  FP 0  FN 10  TN 40    precision NOT ESTIMABLE  recall 0.000
ADVERSARIAL 45     TP 0  FP 0  FN 10  TN 35    precision NOT ESTIMABLE  recall 0.000
```

## Evaluation validity

`TP = 4` and `FP = 0`, so precision is `1.000` but rests on **four** positives. Recall is
**0.167**. The zero false-association rate is **not** a victory: it is accompanied by 20
missed consequences, and the brief is explicit that abstaining from everything does not
count as success. Reported as-is.
