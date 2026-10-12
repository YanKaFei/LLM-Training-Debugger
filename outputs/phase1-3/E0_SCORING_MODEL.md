# E0 — SCORING MODEL

**Status: COMPLETE.** Implemented in `src/training_debugger/evaluation/layers.py`.

Phase 1.2's headline was **event precision 0.121** — a count of raw detector fires.
That number was both the wrong product metric and actively misleading: a single
poisoned optimiser produced 810 fires, so one fault dominated the score, and fixing
the detector that generated them moved the metric without changing whether the
system did anything useful.

---

## 1. Four layers, four questions

| Layer | Question | Ground truth it is graded against |
|---|---|---|
| **SIGNAL** | did the detector see the anomaly at all? | the acceptable detection window |
| **EPISODE** | did it correctly delimit one contiguous anomaly? | window coverage **and** onset localisation |
| **INCIDENT** | did it correctly decide something happened? | one decision per (run, anomaly type) |
| **RUN** | was the run's overall verdict right? | what a user experiences |

`evaluate_layers()` returns all four side by side. **No single headline number is
produced**, because collapsing these is precisely what made Phase 1.2's figures
uninterpretable.

## 2. Why the layers disagree, and why that is the point

Measured on 14 healthy final-holdout runs:

```
signal    P=0.000  R=n/a   TP=0  FP=71  preFP=0  FN=0
episode   P=n/a    R=n/a   TP=0  FP=0   preFP=0  FN=0
incident  P=n/a    R=n/a   TP=0  FP=0   preFP=0  FN=0
run       n/a                          healthy-run false incident rate = 0.0
```

**71 raw signal fires produced zero episodes and zero incidents, and no healthy run
was reported abnormal.** Under the Phase 1.2 scoring model this run would have been
recorded as `precision = 0.000` — the worst possible score — for behaviour that is
exactly correct. That inversion is the reason the layers are now separated.

## 3. Metrics reported per layer

```
signal:   precision, recall, FP/1k, pre-injection, duplicates
episode:  precision, recall, onset localisation error, confirmation delay, abstention rate
incident: precision, recall, duplicate rate, abstention rate
run:      healthy-run false incident rate, false incidents/1k steps,
          abstention rate overall
```

## 4. Unchanged rules carried forward from Phase 1.2

* **Detection delay is taken from the first true-positive confirmation only.** A fire before the injection is a false positive with no delay, never a negative delay.
* **A crashed run is excluded from scoring**, reported separately, never counted as a false negative.
* **Precision and recall are always reported together**, and the abstention rate alongside them, so a system that abstains on everything scores high precision and low recall and is visibly FAILING rather than quietly rewarded. Brief section 21.

## 5. Candidate onset vs confirmation (E1.5)

The episode layer reports two numbers that Phase 1.2 conflated:

* `median_onset_error_steps` — how far the reported onset is from the true effect start;
* `median_confirmation_delay_steps` — how long after onset the system could first confirm.

The first is a statement about the fault; the second is a statement about the
system's knowledge. A run may legitimately have onset error 0 and confirmation
delay 2: the pattern began at step 80, and only became confirmable at 82.
