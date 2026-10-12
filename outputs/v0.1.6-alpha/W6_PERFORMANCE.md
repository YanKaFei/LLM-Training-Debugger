# W6 — PERFORMANCE

## Witness increment (v0.1.5, 7 interleaved paired repeats)

```
metric      off       on        delta     delta%
total_s     0.3704    0.3715    +0.0011   +0.30%
median_ms   3.0725    3.0692    -0.0034   -0.11%
p95_ms      3.2678    3.2995    +0.0317   +0.97%
event volume: off 0 · on 120 records (one per optimizer step)
device synchronizations introduced: none
```

The median delta is **negative** — the effect sits at the harness's noise floor. The
**+0.97 % p95** is the conservative reading and is well inside the 3 % core target.

## Full path

The end-to-end path (training → observer → witness → persistence → episodes → relation
→ CLI) was exercised on all 54 challenge runs. Witness record volume is **one record per
optimizer step** — 44–119 records per run, negligible against ~3,000 events per run.

## What is NOT claimed

* **No GPU benchmark.** This machine is Apple Silicon MPS; no CUDA device is available.
  CPU/MPS figures **do not represent GPU workloads.**
* **The standing overhead finding is untouched.** The pre-existing result that observer
  overhead exceeds the **<3 % core target** at the largest workload was **not**
  re-measured here and is **not** resolved by this round. The 3 % gate remains failing.
* No dropped observations were observed in these runs, but this corpus is small and the
  drop path was not stressed.
