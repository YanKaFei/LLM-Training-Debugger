"""Score every level against a persisted corpus, from the store as it stands now.

This exists because the repository's headline numbers were computed by an ad-hoc
command that left no script behind. When the corpora were regenerated, nothing could
re-derive the scores that the README quotes -- so the numbers and the evidence could
drift apart with no way to notice. A measurement that cannot be repeated is not
evidence.

    python tests/experiments/level_scoring.py <corpus> [<ground-truth.json>]

Writes `outputs/.../level_scoring.json` next to the corpus, with provenance.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))
from artifacts import write_artifact  # noqa: E402

sys.path.insert(0, str(ROOT / "src"))

from training_debugger.evaluation import (  # noqa: E402
    build_scenarios_from_store, evaluate_levels,
)

CORPORA = {
    "development": (
        ROOT / "outputs" / "alpha" / "corpus" / "store",
        ROOT / "outputs" / "alpha" / "corpus" / "ground_truth" / "development_ground_truth.json",
    ),
    "mixed": (
        ROOT / "outputs" / "alpha" / "mixed" / "v015" / "store",
        ROOT / "outputs" / "alpha" / "mixed" / "v015" / "ground_truth" / "mixed_fault_ground_truth.json",
    ),
    "adversarial": (
        ROOT / "outputs" / "alpha" / "adversarial" / "v015" / "store",
        ROOT / "outputs" / "alpha" / "adversarial" / "v015" / "ground_truth" / "adversarial_ground_truth.json",
    ),
}


def main(argv) -> int:
    name = argv[1] if len(argv) > 1 else "development"
    if name not in CORPORA:
        print("unknown corpus {!r}; choose from {}".format(name, sorted(CORPORA)))
        return 2
    store, truth_path = CORPORA[name]
    if len(argv) > 2:
        truth_path = Path(argv[2])
    if not store.exists():
        print("no store at {}".format(store))
        return 1
    truth = json.loads(truth_path.read_text(encoding="utf-8")) if truth_path.exists() else None
    if truth is None:
        print("no ground truth at {} -- scoring refused rather than guessed".format(truth_path))
        return 1

    started = time.perf_counter()
    scenarios = build_scenarios_from_store(store, ground_truth=truth)
    result = evaluate_levels(scenarios)
    result["corpus"] = name
    result["store"] = str(store.relative_to(ROOT))
    result["ground_truth"] = str(truth_path.relative_to(ROOT))
    result["scenarios"] = len(scenarios)
    result["elapsed_s"] = round(time.perf_counter() - started, 2)

    target = store.parent / "level_scoring.json"
    write_artifact(target, result)

    print("corpus       : {}  ({} scenarios)".format(name, len(scenarios)))
    print("status       : {}".format(result.get("status")))
    if result.get("status") != "VALID":
        print("validity     : {}".format(json.dumps(result.get("validity"))[:200]))
        return 0
    print()
    print("%-10s %-6s %-6s %-6s %-6s %-11s %s" % (
        "level", "tp", "fp", "fn", "tn", "precision", "recall"))
    for level, score in result.get("levels", {}).items():
        def fmt(x):
            return "-" if x is None else ("%.4f" % x if isinstance(x, float) else str(x))
        print("%-10s %-6s %-6s %-6s %-6s %-11s %s" % (
            level, fmt(score.get("tp")), fmt(score.get("fp")), fmt(score.get("fn")),
            fmt(score.get("tn")), fmt(score.get("precision")), fmt(score.get("recall"))))
    missing = result.get("missed_primary_faults")
    if missing is not None:
        print("\nmissed_primary_faults: {}".format(missing))
    print("written      : {}".format(target.relative_to(ROOT)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
