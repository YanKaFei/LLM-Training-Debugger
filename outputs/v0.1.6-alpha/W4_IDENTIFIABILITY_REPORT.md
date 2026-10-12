# W4 — IDENTIFIABILITY

Three different facts, audited separately:

```
1  observed optimizer boundary     -> the hook fired. DIRECT.
2  observed parameter mutation      -> NEVER OBSERVED. UNKNOWN by construction.
3  proven downstream propagation    -> NOT OBSERVED. INFERRED, and only when a
                                       genuinely non-finite parameter value appears
                                       between the antecedent and the target.
```

**Fact 2 is not available at all.** Proving "the parameters changed" needs a comparison
against a saved copy — an O(parameter_count) scan this module may not perform. The
witness therefore reports `parameters_changed = None` on every record.

**What actually discriminates, measured:**

```
C1  param.global_norm == nan (count 4160)  -> real corruption  -> claim
C3  param.global_norm absent               -> never corrupted  -> refuse
C5  param.global_norm == inf (count 0)     -> reduction artifact -> refused
```

The discriminating evidence is the **non-finite element observation**, which is already
available and does not need the witness. The optimizer boundary tells us *when* an update
was attempted; it does not tell us *whether that update caused* a later value.

```
IDENTIFIABILITY = INSUFFICIENT
```

No ground truth or injection metadata reaches any production decision.
