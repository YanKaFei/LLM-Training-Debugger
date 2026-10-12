# I7 — DEVELOPMENT RE-EVALUATION

**Status: COMPLETE.** The band is active and the corpus re-run. This is
**development** data, not a holdout.

---

## 1. Result

169 development runs (84 healthy / 85 fault), 4-layer d_model 32 and 6-layer d_model
48, 200 steps, injection at step 120.

| Layer | Phase 1.3 | **Phase 1.5** | Direction |
|---|---|---|---|
| SIGNAL precision | 0.053 | 0.079 | up |
| SIGNAL recall | 0.906 | 0.906 | — |
| **EPISODE precision** | 0.195 | **0.285** | **up** |
| EPISODE recall | 0.588 | 0.553 | down |
| **INCIDENT precision** | 0.412 | **0.500** | **up** |
| INCIDENT recall | 0.806 | 0.676 | **down** |
| **RUN precision** | 0.643 | **0.718** | up |
| RUN recall | 0.847 | 0.718 | down |
| **healthy-run false incident rate** | 0.476 | **0.286** | **down 40 %** |
| **false incidents / 1k steps** | 0.986 | **0.710** | down 28 % |
| pre-injection confirmed incidents | 0 | **0** | preserved |

Episode false positives fell **206 → 118**; incident false positives fell **77 → 50**.

## 2. Development targets — NOT MET

| Target | Required | Measured | Verdict |
|---|---|---|---|
| episode precision | ≥ 0.92 | 0.285 | **FAIL** |
| episode recall | ≥ 0.87 | 0.553 | **FAIL** |
| incident precision | ≥ 0.92 | 0.500 | **FAIL** |
| incident recall | ≥ 0.87 | 0.676 | **FAIL** |
| healthy-run false incident rate | ≤ 3 % | **28.6 %** | **FAIL** |
| false incidents / 1k steps | ≤ 0.3 | 0.710 | **FAIL** |
| earliest evidence precision | ≥ 0.92 | **NOT VERIFIED** | — |
| pre-injection confirmed incidents | 0 | **0** | **PASS** |
| duplicate top-level incidents | 0 | **NOT VERIFIED** | — |

```
READY_FOR_FINAL_HOLDOUT = NO
```

## 3. What the band changed, and what it did not

The band did what it was designed to do — **healthy false alarms fell 47.6 % → 28.6 %**
and incident precision rose 0.412 → 0.500 — but it reduced recall at the same time
(incident 0.806 → 0.676), because observations in the `UNCERTAIN` band are, by
construction, not promoted.

That is the honest trade and it is the intended one: the brief requires preferring
abstention over a confident wrong incident. But **28.6 % of healthy runs still
produce a confirmed incident**, which means the band is not the only source of false
positives. The remainder comes from `loss.robust_zscore` (H2 noisy healthy loss) and
from numerics consequence episodes scoring as independent incidents — both of which
I2 and I1.3 were supposed to address and which are **not implemented**.

## 4. Per-capability development status

| Capability | Precision | Recall | Abstention | Min detectable | Status |
|---|---|---|---|---|---|
| `GRADIENT_EXPLOSION` | 0.285 (episode) | 0.553 | NOT VERIFIED | ×4.0 confirmed; ×3.0 uncertain | **EXPERIMENTAL** |
| `NAN_DETECTED` / `INF_DETECTED` | 0.314 (Phase 1.3; unchanged) | 1.000 | NOT VERIFIED | exact (any non-finite) | **EXPERIMENTAL** |
| `LOSS_SPIKE` | NOT VERIFIED at episode level | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | **EXPERIMENTAL** |
| `LOSS_DIVERGENCE` | — | — | — | — | **UNSUPPORTED** |
| `THROUGHPUT_REGRESSION` | — | — | — | — | **UNSUPPORTED** |

`VALIDATED` is **not used in this phase** and no capability holds it.
`DEVELOPMENT_READY` is **not awarded to any capability**, because the development gate
(episode precision ≥ 0.85 for a per-capability promotion) was not reached.

## 5. UNCERTAIN rate

**NOT VERIFIED.** The brief requires the uncertain rate on healthy and on fault runs
to be reported separately, precisely so that a system cannot score well by pushing
everything into the abstention band. The band assignment is written into anomaly
metadata (`gradient_band`) and therefore exists in the raw data, but the evaluation
layer does not yet aggregate it, so no number is reported.
