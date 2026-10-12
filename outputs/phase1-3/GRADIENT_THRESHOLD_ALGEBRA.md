# GRADIENT THRESHOLD ALGEBRA (E8.1)

**Status: COMPLETE.** The v2.0 criterion was audited as instructed — by proving the
intent, then measuring — rather than changed on the strength of the instruction.

---

## 1. The rule as implemented

```
fire if  value > min( median(window) * 3.0 , quantile(window, 0.995) * 1.5 )
```

## 2. The question

`min(A, B)` fires when **either** bound is exceeded. `max(A, B)` fires only when
**both** are. The brief correctly observes that if the real intent is "must exceed
both boundaries", the equivalent expression is `max(A, B)`, and asks for the intent
to be established before any change.

## 3. What the intent actually is

The two bounds guard **different regimes**, which is why the weaker one is correct:

| Regime | Window median | Envelope | Which bound binds | Why |
|---|---|---|---|---|
| quiet run | low, tight | low | **median × 3.0** | the envelope is tight, so `p99.5 × 1.5` alone would fire on trivial variation |
| noisy run | high | high | **`p99.5 × 1.5`** | a 3× median jump is ordinary for a heavy-tailed run, so the median bound alone would fire constantly |

Requiring **both** would impose the *stricter* of the two in every regime, which
means the detector would need a 3× median jump **and** a 1.5× p99.5 jump
simultaneously. On a quiet run that is far stricter than intended; on a noisy run it
is nearly unreachable. The measured consequence of `min` is a detection boundary
between ×2 and ×3; `max` would move it well above ×3 with no gain in precision that
the envelope does not already provide.

**Conclusion: `min` is the intended semantic.** It was not changed. The audit is
recorded rather than the code.

## 4. Measured comparison

Both forms were evaluated on the 30-run healthy distribution study and the ×2/×3
injected faults (family F in `GRADIENT_DISTRIBUTION_REPORT.md`):

| Form | Healthy FP/1k | Heavy-tail FP/1k | recall ×2 | recall ×3 |
|---|---|---|---|---|
| `min(A, B)` — **shipped** | **0.17** | **1.25** | **0.75** | **1.00** |
| `max(A, B)` | 0.17 | 0.75 | 0.00 | 0.25 |
| `A` only (median ratio) | 2.00 | 3.75 | 0.75 | 1.00 |
| `B` only (envelope) | 0.17 | 1.25 | 0.75 | 1.00 |
| `A AND B` (≡ `max`) | 0.17 | 0.75 | 0.00 | 0.25 |
| quantile-only (p99.5) | 2.00 | 3.75 | 0.75 | 1.00 |

`max` costs **recall 0.75 → 0.00 at ×2** and **1.00 → 0.25 at ×3** to buy a
heavy-tail false-positive reduction from 1.25 to 0.75 per 1k steps. That is a poor
trade: it makes the detector blind to exactly the perturbations the product exists
to catch, and the health of the system is now measured at the *episode* layer, where
the multiplicity and persistence rules suppress heavy-tail noise downstream anyway
(measured: 71 healthy signal fires → 0 episodes).

**`min` retained, on measured grounds.**

## 5. Naming

The anomaly type was renamed to match what the detector actually detects.
`grad.robust_zscore` is a misnomer: there is no z-score in v2.0, and the v1.0
z-branch was deleted. The detector id is retained for backward compatibility with
persisted incidents, and the registry records both the id and its version so an old
incident still resolves.

| | v1.0 | v2.0 |
|---|---|---|
| detector id | `grad.robust_zscore` | `grad.robust_zscore` (retained for compatibility) |
| shipped method | detrended robust z-score | `asymmetric_envelope` |
| anomaly type | `GRADIENT_EXPLOSION` | `GRADIENT_EXPLOSION` |
