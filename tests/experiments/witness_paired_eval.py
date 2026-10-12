"""W2: paired OFF/ON evaluation on byte-identical evidence.

Both configurations read the SAME persisted episodes from the same runs. Nothing about
training differs -- only whether the optimizer-boundary witness is permitted to serve as
the midstream evidence in the relation decision.

```
A  WITNESS_DISABLED  midstream evidence = a genuinely non-finite (*.global_norm == nan)
                     parameter observation, i.e. non-finite ELEMENTS
B  WITNESS_ENABLED   midstream evidence = the above, OR an INVOKED optimizer step
```

B is the hypothesis under test: that knowing the optimizer was invoked adds
discriminative power. C3 is the case designed to falsify it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from training_debugger.evaluation import (  # noqa: E402
    RelationOpportunityScore, persisted_run_ids, score_relation_opportunities,
)
from training_debugger.incidents.relations import (  # noqa: E402
    PROPAGATION_DELAY_STEPS,
    NONFINITE_TYPES, is_nonfinite_observation, metric_class,
)
from training_debugger.persistence import EpisodeStore  # noqa: E402

OUT = ROOT / "outputs" / "v0.1.6-alpha"
STORE = OUT / "store"
TRUTH = OUT / "ground_truth" / "witness_challenge_ground_truth.json"


def witness_status(run_id: str) -> Dict[str, Any]:
    path = STORE / "runs" / run_id / "witness.json"
    if not path.exists():
        return {"attached": False, "records": [], "summary": {}}
    return json.loads(path.read_text())


def link(episodes: List[Dict[str, Any]], witness: Dict[str, Any],
         use_witness: bool) -> List[Dict[str, Any]]:
    """The frozen relation rule, with the witness optionally supplying midstream evidence.

    This mirrors `link_consequences` exactly and adds ONE clause, so the comparison is
    not confounded by any other difference.
    """
    def confirmed(episode: Dict[str, Any]) -> bool:
        return (episode.get("state") == "CONFIRMED"
                or episode.get("outcome") == "CONFIRMED_ANOMALY")

    antecedents = []
    for episode in episodes:
        if episode.get("candidate_onset_step") is None:
            continue
        is_primary = (episode.get("anomaly_type") in ("GRADIENT_EXPLOSION", "LOSS_SPIKE")
                      and confirmed(episode)
                      and episode.get("confirmed_at_step") is not None)
        is_upstream = (is_nonfinite_observation(episode)
                       and metric_class(episode.get("metric_name")) == "upstream")
        if is_primary or is_upstream:
            antecedents.append((episode, is_primary, is_upstream))

    consequences = [e for e in episodes
                    if e.get("anomaly_type") in ("NAN_DETECTED", "INF_DETECTED")]

    # The witness clause: an INVOKED optimizer step counts as boundary evidence.
    invoked_steps = sorted(
        r["training_step_id"] for r in witness.get("records", [])
        if r.get("optimizer_step_status") == "INVOKED"
    ) if use_witness else []

    relations: List[Dict[str, Any]] = []
    for target in consequences:
        onset = target.get("candidate_onset_step")
        if onset is None:
            continue
        target_class = metric_class(target.get("metric_name"))
        best: Optional[tuple] = None
        for antecedent, is_primary, is_upstream in antecedents:
            if antecedent.get("episode_id") == target.get("episode_id"):
                continue
            reference = (antecedent.get("confirmed_at_step") if is_primary
                         and antecedent.get("confirmed_at_step") is not None
                         else antecedent.get("candidate_onset_step"))
            if reference is None:
                continue
            gap = onset - int(reference)
            if gap < 0 or gap > 60:
                continue
            layers_a = set(antecedent.get("affected_layers") or [])
            layers_t = set(target.get("affected_layers") or [])
            if layers_a and layers_t and not (layers_a & layers_t):
                continue

            midstream = any(
                is_nonfinite_observation(other)
                and metric_class(other.get("metric_name")) == "midstream"
                and reference <= (other.get("candidate_onset_step") or -1) <= onset
                for other in episodes
            )
            witness_clause = any(reference <= s <= onset for s in invoked_steps)
            # The witness is given its STRONGEST fair reading: an invoked optimizer
            # step may establish the mechanism by itself, for either a non-finite
            # upstream antecedent or a confirmed primary anomaly. Making it weaker
            # would leave it inert and the comparison would prove nothing.
            if use_witness:
                mechanism = bool(
                    (is_upstream or is_primary)
                    and (midstream or witness_clause)
                    and gap <= PROPAGATION_DELAY_STEPS
                    and target_class in ("midstream", "downstream", "unknown")
                )
            else:
                mechanism = bool(
                    is_upstream
                    and midstream
                    and gap <= PROPAGATION_DELAY_STEPS
                    and target_class in ("midstream", "downstream", "unknown")
                )
            tier = "MECHANISM" if mechanism else "TEMPORAL"
            candidate = (0 if mechanism else 1, gap)
            if best is None or candidate < best[0]:
                best = (candidate, antecedent, gap, tier,
                        {"midstream_nonfinite_witness": bool(midstream),
                         "witness_clause_used": bool(use_witness and witness_clause)})
        if best is None:
            continue
        _, antecedent, gap, tier, evidence = best
        relations.append({
            "relation_type": ("POSSIBLE_CONSEQUENCE_OF" if tier == "MECHANISM"
                              else "PRECEDES"),
            "source_incident_id": antecedent.get("episode_id"),
            "target_incident_id": target.get("episode_id"),
            "target_anomaly_type": target.get("anomaly_type"),
            "gap_steps": gap,
            "evidence": dict(evidence, tier=tier),
        })
    return relations


def main() -> int:
    truth = json.loads(TRUTH.read_text())
    run_ids = persisted_run_ids(STORE)
    store = EpisodeStore(STORE)
    report: Dict[str, Any] = {"runs": len(run_ids), "configurations": {}}

    for label, use_witness in (("WITNESS_DISABLED", False), ("WITNESS_ENABLED", True)):
        scenarios = []
        for run_id in run_ids:
            import pyarrow.parquet as pq
            directory = STORE / "runs" / run_id / "episodes" / "episodes"
            ids = set()
            if directory.exists():
                for path in sorted(directory.glob("*.parquet")):
                    for row in pq.read_table(str(path)).to_pylist():
                        ids.add(row["episode_id"])
            episodes = []
            for episode_id in sorted(ids):
                try:
                    episodes.append(store.load(run_id, episode_id).to_dict())
                except Exception:
                    continue
            witness = witness_status(run_id)
            scenarios.append({
                "run_id": run_id,
                "relations": link(episodes, witness, use_witness),
                "episodes": episodes,
            })
        score = score_relation_opportunities(scenarios, truth)
        payload = score.to_dict()
        payload["per_case"] = {}
        for scenario in scenarios:
            case = truth[scenario["run_id"]]["case"]
            claims = sum(1 for r in scenario["relations"]
                         if r["relation_type"] == "POSSIBLE_CONSEQUENCE_OF")
            bucket = payload["per_case"].setdefault(
                case, {"runs": 0, "consequence_claims": 0,
                       "order_only": 0, "opportunity": truth[scenario["run_id"]]["opportunity"]})
            bucket["runs"] += 1
            bucket["consequence_claims"] += claims
            bucket["order_only"] += len(scenario["relations"]) - claims
        report["configurations"][label] = payload

    off = report["configurations"]["WITNESS_DISABLED"]
    on = report["configurations"]["WITNESS_ENABLED"]
    report["paired_difference"] = {
        "recovered_true": on["true_positive"] - off["true_positive"],
        "new_false": on["false_positive"] - off["false_positive"],
        "lost_true": max(0, off["true_positive"] - on["true_positive"]),
        "correctly_rejected_delta": on["true_negative"] - off["true_negative"],
        "abstention_delta": on["order_only_on_negative"] - off["order_only_on_negative"],
    }
    (OUT / "paired_evaluation.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")

    f = lambda v: "NOT ESTIMABLE" if v is None else round(v, 3)
    for label in ("WITNESS_DISABLED", "WITNESS_ENABLED"):
        row = report["configurations"][label]
        print()
        print(label)
        print("  TP=%d FP=%d FN=%d TN=%d" % (row["true_positive"], row["false_positive"],
                                              row["false_negative"], row["true_negative"]))
        print("  precision=%s recall=%s coverage=%s" % (
            f(row["precision"]), f(row["recall"]), f(row["positive_coverage"])))
        print("  false_association=%s negative_rejection=%s" % (
            f(row["false_association_rate"]), f(row["negative_rejection_rate"])))
        for case in sorted(row["per_case"]):
            b = row["per_case"][case]
            print("    %-4s %-9s runs=%-2d consequence_claims=%d order_only=%d" % (
                case, b["opportunity"], b["runs"], b["consequence_claims"], b["order_only"]))
    print()
    print("paired difference:", json.dumps(report["paired_difference"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
