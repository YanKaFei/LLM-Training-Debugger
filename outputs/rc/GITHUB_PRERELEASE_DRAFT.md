# GITHUB RELEASE DRAFT — NOT PUBLISHED

```
Title:  LLM-Training-Debugger v0.1.7-alpha
Tag:    v0.1.7-alpha
Type:   Experimental Pre-release  (check "This is a pre-release")
Target: main
```

**Do not publish this without owner review.** The commands are listed for the owner to run
manually; nothing here has been executed.

---

## Release body (draft)

> ## Experimental Pre-release
>
> An experimental **PyTorch training evidence monitor**. It records what a training run
> did, detects some abnormal signs, persists the evidence, and lets you query it later
> from a separate process — with explicit abstention when the evidence is insufficient.
>
> **This is not a validated root-cause diagnoser, and it is not suitable for making
> automated decisions to stop, roll back or repair a training run.**
>
> ### Read this first
>
> ```
> Research-stage diagnostic accuracy.
> No blind holdout validation.
> No validated detector capabilities.
> Observer performance target not fully met.
> Automatic root-cause inference is not validated.
> ```
>
> ### What works
>
> Non-blocking fail-open observer · Parquet + DuckDB storage · deterministic detectors
> (loss spike, gradient explosion, numeric finiteness) · explicit episode state machine ·
> append-only episodes/incidents/relations/analysis state · cross-process recovery of
> episodes **and** the protected baseline · four-verdict CLI · evidence-completeness
> reporting on every verdict.
>
> ### Detection boundary (development-supported)
>
> ```
> gradient scaling  x2 -> 0/6    x5 -> 0/6    x20 -> 6/6    x40 -> 6/6
> run-level recall 0.500    healthy-run false incidents ~7.7 %
> ```
>
> Mild gradient faults are **not distinguishable** from healthy variation by any signal
> collected. The system abstains rather than guessing.
>
> ### Relation presentation
>
> `POSSIBLE_CONSEQUENCE_OF` is **disabled by default** — it was measured to be
> undiscriminable from an independently injected fault in every case that could be
> constructed. The default records `PRECEDES` / `CO_OCCURS_WITH` / `UNKNOWN_RELATION`
> only. Mechanism relations are **not** causal proof.
>
> ### Install
>
> ```bash
> python3 -m venv .venv && . .venv/bin/activate
> pip install --upgrade "setuptools>=61" wheel
> pip install -e ".[torch,dev]"        # or: pip install training_debugger-0.1.7a1-py3-none-any.whl
> python examples/quickstart.py
> ```
>
> ### Not included
>
> No LLM/agent · no replay · no multi-rank · no GPU telemetry · no learned detector ·
> no web dashboard.
>
> Full detail: `KNOWN_LIMITATIONS.md`, `EVALUATION_REPORT.md`, `REPRODUCIBILITY.md`.

---

## Owner commands (NOT executed)

> **This directory is now a dedicated project checkout.** The project was moved out of the
> shared working directory that also held ~1.9 GB of unrelated material (a pnpm cache, two
> unrelated projects, research clones, a personal contact database, proxy/VPN scripts).
> Everything that is not this project is now physically absent, so `git add -A` is safe
> **by construction** rather than by ignore rule -- verified with `git check-ignore`:
>
> ```
> would stage  22 entries -- .gitignore, docs, examples, outputs, src, tests,
>              all *.md, LICENSE, MANIFEST.in, pyproject.toml
> IGNORED      .venv, tests/experiments/restart_{worker,continue}.py (runtime-generated)
> absent       .pnpm-store  ascii-cinema  puppet-global  research
>              puppetry-contacts.json  proxy-mode.sh
> ```

```bash
# 1. review the audit
cat outputs/rc/V0_1_7_PRERELEASE_REPORT.md

# 2. initialise and commit
git init
git add -A
git status --short          # sanity check: all project content, nothing else
git commit -m "LLM-Training-Debugger v0.1.7-alpha: experimental pre-release"

# 3. tag the pre-release
git tag -a v0.1.7-alpha -m "LLM-Training-Debugger v0.1.7-alpha (0.1.7a1) — experimental pre-release"
git push origin main --follow-tags

# 4. create the GitHub pre-release FROM the draft above, ticking "pre-release"
gh release create v0.1.7-alpha \
  --prerelease \
  --title "LLM-Training-Debugger v0.1.7-alpha" \
  --notes-file RELEASE_NOTES.md \
  /tmp/rcfinal/training_debugger-0.1.7a1-py3-none-any.whl \
  /tmp/rcfinal/training_debugger-0.1.7a1.tar.gz
```

**Recommended instead:** move `src/`, `tests/`, `examples/`, `docs/`, `outputs/` and the
project root files into a **dedicated directory** and initialise the repository there.
That removes the risk of a future `git add -A` re-introducing unrelated material, and it is
the only form of this workflow that is safe by construction rather than by ignore rule.

**Note:** this workspace is currently **not a git repository**, so steps 2–4 presuppose
initialising one. That is an owner decision, not an autonomous action.
