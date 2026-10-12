# W2 — PAIRED EXPERIMENT

**54 runs, 9 cases, 2 model configurations, 3 seeds each.** Both configurations read the
**same persisted episodes** from the same runs; nothing about training differs.

```
A  WITNESS_DISABLED  midstream evidence = a genuinely non-finite parameter observation
                                        (*.global_norm == nan, i.e. non-finite ELEMENTS)
B  WITNESS_ENABLED   A, OR an INVOKED optimizer step on the boundary
```

B was given its **strongest fair reading**: an invoked step may establish the mechanism
by itself, for either a non-finite upstream antecedent or a confirmed primary anomaly.
A weaker clause would have been inert and the comparison would have proved nothing.

```
                         TP   FP   FN   TN    precision   recall   coverage
WITNESS_DISABLED          4    0   20   30      1.000     0.167     0.167
WITNESS_ENABLED           4    0   20   30      1.000     0.167     0.167

paired difference:  recovered_true 0 · new_false 0 · lost_true 0
                    correctly_rejected_delta 0 · abstention_delta 0
```

**Byte-identical on every run.** The witness neither helps nor harms.
