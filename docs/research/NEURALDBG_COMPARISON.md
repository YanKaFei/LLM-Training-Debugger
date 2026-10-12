# B — NeuralDBG compared, empirically

Ran on the same training run this project's corpus uses. Not a documentation comparison —
the package was downloaded and executed.

```
neuraldbg 1.3.2 · MIT · 1,447 lines across 2 files
"Summary: Causal inference engine for deep learning training dynamics"
```

---

## 1. What was run

Real autograd, 4-layer model, one **finite gradient excursion (x30)** injected at step 60.
Both tools saw the same run.

```python
dbg = NeuralDbg(model)
with dbg:
    for step in ...:
        loss.backward()
        if step == 60: p.grad.mul_(30.0)
        dbg.record_loss(...)
        optimizer.step()
        dbg.step_iteration()
```

## 2. What it reported

```
NeuralDBG events                : 101
NeuralDBG first event step      : 0          <- before any training happened
NeuralDBG causal hypotheses     : 3, all at confidence 0.72

  conf=0.72  Cascade: activation_regime_shift propagates through
             ['Embedding_token_embedding', 'Embedding_positio...']
  conf=0.72  Cascade: data_anomaly propagates through
             ['LayerNorm_blocks.0.norm1', 'LayerNorm_blocks.0.norm2', ...]
  conf=0.72  Cascade: gradient_health_transition propagates through
             ['Linear_lm_head', 'root', 'LayerNorm_final_n...']
```

**Three causal "cascade propagation" hypotheses at identical confidence 0.72, on a run
whose only anomaly is a single injected gradient excursion at step 60.** The reported layer
chains are the model's own module order — `Embedding → LayerNorm → Linear → root` — which
is read from the module tree, not discovered. `'root'` is not a layer.

## 3. Why — read from the source

`neuraldbg/enhanced_causality.py::_build_causal_graph`:

```python
if current_issues and next_issues:
    current_first = min(e.step for e in current_issues)
    next_first = min(e.step for e in next_issues)
    if current_first <= next_first:
        # Causal relationship likely
        weight = min(1.0, (next_first - current_first + 1) / 10)
        self.causal_graph.add_edge(layer, next_layer, weight)
```

The "causal graph" is built from **temporal precedence plus module adjacency**, and the
edge weight is `(time gap + 1) / 10` — **closer in time scores as more causal**. The public
entry point is named `enhance_with_granger_style`; Granger causality is predictive
precedence, and no counterfactual, intervention, or confounder control appears anywhere in
the 1,447 lines.

## 4. The comparison that matters

**This project measured that exact rule and removed it.** In v0.1.3 the relation linker used
"temporal proximity + scope overlap" and produced:

```
M3 (two DELIBERATELY INDEPENDENT faults)  -> 10 of 10 claimed as POSSIBLE_CONSEQUENCE_OF
M1 (the genuine causal chain)             ->  0 of 10
                                             precision 0.0, recall 0.0
```

The rule NeuralDBG reports at 0.72 confidence is the rule this project's corpus falsified.

## 5. So: discipline or capability gap?

**Discipline for the causal layer; a real gap for the instrumentation layer.**

### Where it is genuinely better than this project

| | |
|---|---|
| **Per-module health states** | `Vanishing` / `Exploding` / `Dead` / `Saturated` **per named module**. This project has no dead-neuron or saturation detection at all — a real capability gap |
| **Ergonomics** | `with NeuralDbg(model):` — zero configuration |
| **Activation observation** | It hooks activations directly. This project observes gradient *norms* per layer, not activation distributions |

### Where its claims are unsupported

| | |
|---|---|
| "Causal" | temporal precedence + adjacency; no counterfactual, no intervention |
| Confidence 0.72 | identical for all three hypotheses; derived from a time-gap formula, not from evidence strength |
| Layer chains | the module order, not a discovered path |
| First event at step 0 | reported before any training occurred |

### Where this project is stricter

| | |
|---|---|
| `CAUSED_BY` / `ROOT_CAUSE_OF` | do not exist; asserted absent by test |
| A consequence claim | requires the propagation observed at **every** stage (gradient → parameters → target), not merely ordered |
| Order-only relations | recorded as `PRECEDES`, which explicitly is not a consequence |
| Negative results | the fabricated-association measurement above is published in this repository |

## 6. Verdict

```
MECHANISM CLAIMS:  this project is stricter, and has the measurement to justify it.
INSTRUMENTATION:   NeuralDBG is broader. Per-module activation health is a real gap.
```

**The honest next step is to take the instrumentation, not the causal framing.** Adding
per-module activation/health classification would be a genuine capability gain and does not
require any causal claim: "module X's activations collapsed to zero at step 47" is an
observation, and a useful one this project cannot currently make.

**What should not be copied** is the causal graph. It is the same temporal-precedence rule
this project already falsified once, and shipping it under the word "causal" would undo the
discipline that the rest of this repository is built on.

## 7. Reproduce

```bash
pip download --no-deps --dest /tmp/ndbg-dl neuraldbg
mkdir /tmp/ndlib && cd /tmp/ndlib && unzip -q /tmp/ndbg-dl/neuraldbg-*.whl
# run with the project venv (it already has torch)
PYTHONPATH=/tmp/ndlib .venv/bin/python <script>
```

Working script: the inline block in §1. The source reading is
`/tmp/ndlib/neuraldbg/enhanced_causality.py`, lines 92–145.
