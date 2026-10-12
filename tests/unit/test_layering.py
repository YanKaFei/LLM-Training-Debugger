"""Enforces the Phase 1 layer direction (brief section 19).

    Adapter -> Observer -> TrainingEvent -> Storage -> Detector -> Incident

Dependencies may point *downward only*. This is checked by parsing imports
statically rather than by trusting review, because the failure mode is a
convenient local import added under deadline pressure.

The check is deliberately simple: it reads module-level ``import`` and
``from ... import`` statements with :mod:`ast` and rejects an import whose
target layer sits above the importing layer.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, Iterator, List, Set, Tuple

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "training_debugger"

#: Layer index. A module may import from a layer with a *lower or equal* index.
#: The schema layer is the floor: everything depends on it and it depends on
#: nothing inside the package.
LAYER_ORDER: List[str] = [
    "schema",
    "observer",
    "storage",
    "detectors",
    "incidents",
    "adapters",
    "cli",
]
LAYER_INDEX: Dict[str, int] = {name: i for i, name in enumerate(LAYER_ORDER)}

#: Explicitly allowed upward edges, each with a reason. Empty by design: adding
#: an entry here should require an argument in review.
ALLOWED_UPWARD: Set[Tuple[str, str]] = set()


def _iter_modules() -> Iterator[Tuple[str, Path]]:
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC)
        parts = rel.parts
        if parts[0] == "__init__.py":
            yield "", path
            continue
        yield parts[0], path


def _imported_package_modules(path: Path) -> List[str]:
    """Return the ``training_debugger.<layer>`` targets imported by ``path``."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            module = node.module
            if module == "training_debugger" or module.startswith("training_debugger."):
                parts = module.split(".")
                if len(parts) >= 2:
                    found.append(parts[1])
            elif node.level and node.level > 0:
                # Relative import: resolve against the file's own package.
                found.append(path.relative_to(SRC).parts[0])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("training_debugger."):
                    parts = alias.name.split(".")
                    if len(parts) >= 2:
                        found.append(parts[1])
    return found


def test_no_module_imports_an_upward_layer():
    violations: List[str] = []
    for layer, path in _iter_modules():
        if not layer or layer not in LAYER_INDEX:
            continue
        own = LAYER_INDEX[layer]
        for target in _imported_package_modules(path):
            if target not in LAYER_INDEX or target == layer:
                continue
            if LAYER_INDEX[target] > own and (layer, target) not in ALLOWED_UPWARD:
                violations.append(
                    "{} (layer {}) imports {} (layer {})".format(
                        path.relative_to(SRC), layer, target, LAYER_ORDER[LAYER_INDEX[target]]
                    )
                )
    assert not violations, "upward dependencies found:\n  " + "\n  ".join(violations)


def test_schema_layer_has_no_intra_package_dependencies():
    """The schema is the floor. If it grows a dependency, the layering is wrong."""
    schema = SRC / "schema"
    offenders: List[str] = []
    for path in sorted(schema.glob("*.py")):
        for target in _imported_package_modules(path):
            if target and target != "schema":
                offenders.append("{} -> {}".format(path.name, target))
    assert not offenders, "schema layer must not depend on higher layers: %s" % offenders


def test_adapters_do_not_implement_detection_or_storage():
    """A detector in an adapter is the classic layering leak."""
    forbidden = ("training_debugger.detectors", "training_debugger.incidents", "training_debugger.storage")
    offenders: List[str] = []
    adapters = SRC / "adapters"
    if adapters.exists():
        for path in sorted(adapters.rglob("*.py")):
            text = path.read_text(encoding="utf-8")
            for name in forbidden:
                if name in text:
                    offenders.append("{} references {}".format(path.name, name))
    assert not offenders, offenders


def test_no_broad_except_pass_in_core_modules():
    """Phase 1 code-quality rule: no silent ``except:``.

    A bare ``except Exception: pass`` in the observer would hide exactly the
    failures the fault-injection gate exists to surface.
    """
    offenders: List[str] = []
    for layer, path in _iter_modules():
        if layer not in ("schema", "observer", "storage", "detectors", "incidents"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                body = node.body
                if len(body) == 1 and isinstance(body[0], ast.Pass):
                    offenders.append("{}:{}".format(path.relative_to(SRC), node.lineno))
    assert not offenders, "silent except/pass found at: %s" % offenders
