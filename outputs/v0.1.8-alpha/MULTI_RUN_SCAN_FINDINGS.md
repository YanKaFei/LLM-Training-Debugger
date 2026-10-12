# C — Multi-run consensus scan

`compare` needed a designated reference. The real question is *"here are forty runs from
last night — which one is wrong?"* Nobody wants to name a reference, and naming a bad one
poisons the answer.

```
scan <run1> <run2> ... <runN>
```

Every unordered pair is compared once. A run's score is how many peers it diverged from.

---

## 1. Real-data validation

### Positive control — like-for-like batch

Five identical clean replicas plus one faulty run, **all at one seed**, injection at step 90:

```
run                          diverged   of peers   median pivot
LIKEFORLIKE_faulty                  5          5             90    <- correctly named
LIKEFORLIKE_clean_0                 1          5             90
LIKEFORLIKE_clean_1                 1          5             90
LIKEFORLIKE_clean_2                 1          5             90
LIKEFORLIKE_clean_3                 1          5             90
LIKEFORLIKE_clean_4                 1          5             90

VERDICT: OUTLIER_FOUND
  "LIKEFORLIKE_faulty diverges from 5 of its 5 peer(s), most often at step 90 on
   grad.global_norm. 5 peer(s) localise a pivot within 2 step(s) of that."
```

Exact step, correct run, unanimous peer support.

### Negative control 1 — six healthy runs, three seeds

```
VERDICT: NO_SINGLE_OUTLIER
```

### Negative control 2 — five healthy + one faulty, but **mixed seeds**

```
VERDICT: NO_SINGLE_OUTLIER
```

## 2. The defect this found, on real data

The first version reported an outlier on **six healthy runs**. They were not one.

The runs were at three different seeds. Independently sampled runs differ from each other
legitimately, so there is **no majority for anything to be an outlier *from***. Consensus
requires a coherent agreeing majority, and that has to be checked rather than assumed.

### The first fix was wrong

I first keyed on *pairwise agreement*: exclude the top run, ask whether the rest agree with
each other. It still reported an outlier on six healthy runs, because the healthy runs did
mostly agree with each other — they just also differed from the candidate.

### The right discriminator is pivot agreement

A genuine outlier departs at the **same step** from every peer, because there is one event.
Independently sampled runs diverge at **scattered** steps, because accumulated variance has
no single onset. Measured:

```
one faulty run among five clean    ->  5/5 peers localise within 2 steps
six healthy runs, three seeds      ->  0/4 peers localise within 2 steps
```

The rule now requires a majority of the diverging peers to agree on **when**. Below that,
`NO_SINGLE_OUTLIER` with the reason naming seed/config variance and telling the user to
compare like with like.

## 3. What it deliberately does not say

* **Which side moved.** A divergence between two runs is evidence about *both*; the verdict
  says "a run to investigate, **NOT** a faulty run" and states that the comparison cannot
  say which side of a divergence moved.
* **A single answer when there is a tie.** If two runs each diverge from everyone, the
  ranking shows that and the reason says it is a tie, not an answer.
* **More than a step and a metric.** No kernel, no tensor, no cause.

## 4. Requirement discovered

**`scan` needs like-for-like runs.** Hold the seed and configuration fixed and vary one
factor. On a batch that mixes seeds it correctly refuses — which means it will say
`NO_SINGLE_OUTLIER` to a user who expected a name. That is the honest answer to a badly
formed question, but it is a usability cliff worth documenting, and it is now in the
command's own reason string.

## 5. Status

```
tests  570 passing, 1 skipped   (560 -> 570, +10)
smoke  8/8 · quickstart PASS · wheel builds · compare/scan/diagnose all available
       in a clean-venv install

EXPERIMENTAL — one synthetic paired corpus, 4-layer, no blind holdout.
```
