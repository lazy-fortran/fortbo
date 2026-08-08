#!/usr/bin/env python3
"""Audit the pinned B5 TuRBO control ledgers without importing their runner.

The controls live in the external simsopt-dfo checkout and are deliberately
not copied into this repository.  This checker verifies the evidence at the
schema boundary: paired modes and seeds, fixed budget and worker count,
per-run ledgers, region assignments, and every raw-file digest recorded by the
aggregate documents.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional


SEEDS = tuple(range(1, 6))
BUDGET = 256
WORKERS = 8
CASE_ID = "b5_simple_to_build_qi"


class B5AuditError(ValueError):
    """A B5 control document does not satisfy the campaign contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise B5AuditError(message)


def _load(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise B5AuditError(f"cannot read {path}: {error}") from error
    _require(isinstance(value, dict), f"{path} must contain an object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise B5AuditError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def _raw_path(comparison: Path, raw_file: str,
              source_root: Optional[Path]) -> Path:
    candidates = []
    if source_root is not None:
        candidates.append(source_root / raw_file)
    candidates.append(comparison.parent / raw_file)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise B5AuditError(
        f"raw ledger is missing: {raw_file} (searched "
        + ", ".join(str(path) for path in candidates) + ")"
    )


def _check_runtime(source: Mapping[str, Any], label: str,
                   require_constellaration: bool = True) -> None:
    _require(source.get("git_dirty") is False,
             f"{label} source checkout is dirty")
    commit = source.get("git_commit")
    _require(isinstance(commit, str) and len(commit) == 40,
             f"{label} source commit is not a full object id")
    if require_constellaration:
        constellaration = source.get("constellaration_commit")
        _require(isinstance(constellaration, str) and len(constellaration) == 40,
                 f"{label} ConStellaration commit is not pinned")


def _check_ledger(raw: Mapping[str, Any], path: Path, *, mode: str,
                  regions: int) -> None:
    expected_schema = ("simsopt-dfo.b5-async-turbo-m"
                       if regions > 1 else "simsopt-dfo.b5-async-turbo")
    label = str(path)
    _require(raw.get("schema_name") == expected_schema,
             f"{label}: unexpected schema")
    _require(raw.get("schema_version") == 1, f"{label}: schema version")
    _require(raw.get("passed") is True, f"{label}: ledger is not passed")
    _check_runtime(raw.get("source", {}), label)

    problem = raw.get("problem", {})
    _require(problem.get("case_id") == CASE_ID, f"{label}: wrong B5 case")
    _require(problem.get("mode") == mode, f"{label}: wrong coordinate mode")
    dimension = problem.get("dimension")
    expected_dimension = 80 if mode == "raw" else 20
    _require(dimension == expected_dimension,
             f"{label}: expected dimension {expected_dimension}")

    configuration = raw.get("configuration", {})
    _require(configuration.get("seed") in SEEDS, f"{label}: invalid seed")
    _require(configuration.get("budget") == BUDGET,
             f"{label}: budget is not {BUDGET}")
    _require(configuration.get("workers") == WORKERS,
             f"{label}: workers are not {WORKERS}")
    if regions == 1:
        _require(configuration.get("method") ==
                 "completion-driven TuRBO-1 with Thompson sampling",
                 f"{label}: wrong TuRBO-1 method")
    else:
        _require(configuration.get("method") ==
                 "completion-driven TuRBO-m with Thompson sampling",
                 f"{label}: wrong TuRBO-m method")
        _require(configuration.get("regions") == regions,
                 f"{label}: wrong region count")
        _require(configuration.get("initial_points_per_region") == 40,
                 f"{label}: wrong initial points per region")

    evaluations = raw.get("evaluations")
    _require(isinstance(evaluations, list) and len(evaluations) == BUDGET,
             f"{label}: expected {BUDGET} evaluation rows")
    identifiers = [row.get("candidate_id") for row in evaluations
                   if isinstance(row, dict)]
    _require(identifiers == list(range(BUDGET)),
             f"{label}: candidate ids are not contiguous")
    for index, row in enumerate(evaluations):
        _require(isinstance(row, dict), f"{label}: row {index} is not an object")
        _require(row.get("status") in {"ok", "failed"},
                 f"{label}: row {index} has an invalid status")
        point = row.get("unit_x")
        _require(isinstance(point, list) and len(point) == dimension,
                 f"{label}: row {index} has the wrong unit width")
        _require(all(isinstance(value, (int, float)) and math.isfinite(value)
                     and 0.0 <= value <= 1.0 for value in point),
                 f"{label}: row {index} has an invalid unit point")
        if row["status"] == "ok":
            _require(isinstance(row.get("value"), (int, float)) and
                     math.isfinite(row["value"]),
                     f"{label}: successful row {index} has no finite value")
        else:
            _require(row.get("value") is None,
                     f"{label}: failed row {index} invented an objective value")
        if regions > 1:
            _require(row.get("region_id") in range(regions),
                     f"{label}: row {index} has an invalid region")

    result = raw.get("result", {})
    _require(result.get("truth_calls") == BUDGET,
             f"{label}: result truth-call count is not {BUDGET}")
    _require(result.get("peak_concurrency") == WORKERS,
             f"{label}: peak concurrency is not {WORKERS}")
    _require(isinstance(result.get("failed_evaluations"), int) and
             result["failed_evaluations"] == sum(
                 row["status"] != "ok" for row in evaluations),
             f"{label}: failure count disagrees with the ledger")


def _check_comparison(path: Path, *, source_root: Optional[Path], multi: bool) -> int:
    document = _load(path)
    expected_schema = ("simsopt-dfo.b5-turbo-m-comparison"
                       if multi else "simsopt-dfo.b5-turbo-comparison")
    _require(document.get("schema_name") == expected_schema,
             f"{path}: unexpected comparison schema")
    _require(document.get("schema_version") == 1,
             f"{path}: comparison schema version")
    _require(document.get("passed") is True, f"{path}: comparison is not passed")
    _check_runtime(document.get("source", {}), str(path),
                   require_constellaration=False)
    _check_runtime(document.get("verification_source", {}), str(path),
                   require_constellaration=False)

    configuration = document.get("configuration", {})
    _require(configuration.get("case_id") == CASE_ID,
             f"{path}: comparison case id")
    _require(configuration.get("budget") == BUDGET,
             f"{path}: comparison budget")
    _require(configuration.get("workers") == WORKERS,
             f"{path}: comparison workers")
    _require(configuration.get("paired_seeds") == list(SEEDS),
             f"{path}: comparison seeds")
    raw_documents = document.get("raw_documents")
    _require(isinstance(raw_documents, list) and len(raw_documents) == 10,
             f"{path}: expected ten raw ledgers")

    seen = set()
    for raw_document in raw_documents:
        _require(isinstance(raw_document, dict),
                 f"{path}: malformed raw-document record")
        if multi:
            method = raw_document.get("method")
            _require(method in {"turbo-1", "turbo-m-4"},
                     f"{path}: invalid TuRBO-m comparison method")
            mode = "data-informed"
            regions = 4 if method == "turbo-m-4" else 1
            key = (method, raw_document.get("seed"))
        else:
            mode = raw_document.get("mode")
            _require(mode in {"raw", "data-informed"},
                     f"{path}: invalid TuRBO-1 comparison mode")
            regions = 1
            key = (mode, raw_document.get("seed"))
        _require(key not in seen, f"{path}: duplicate raw ledger {key}")
        seen.add(key)
        raw_path = _raw_path(path, raw_document.get("file", ""), source_root)
        _require(_sha256(raw_path) == raw_document.get("sha256"),
                 f"{raw_path}: SHA-256 disagrees with comparison")
        raw = _load(raw_path)
        _require(raw.get("configuration", {}).get("seed") == raw_document.get("seed"),
                 f"{raw_path}: seed disagrees with comparison")
        _check_ledger(raw, raw_path, mode=mode, regions=regions)

    expected = ({("turbo-1", seed) for seed in SEEDS} |
                {("turbo-m-4", seed) for seed in SEEDS}) if multi else (
                    {(mode, seed) for mode in ("raw", "data-informed")
                     for seed in SEEDS})
    _require(seen == expected, f"{path}: paired ledger set is incomplete")
    accounting = document.get("accounting", {})
    _require(accounting.get("truth_calls_per_run") == BUDGET,
             f"{path}: accounting call count")
    _require(accounting.get("total_truth_calls") == 10*BUDGET,
             f"{path}: accounting total call count")
    return len(raw_documents)


def check(turbo_one: Path, turbo_m: Path, source_root: Optional[Path]) -> None:
    _check_comparison(turbo_one, source_root=source_root, multi=False)
    _check_comparison(turbo_m, source_root=source_root, multi=True)
    print("B5 controls: PASS (10 TuRBO-1 and 10 TuRBO-m ledgers, 256 calls, 8 workers)")


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("turbo_one", type=Path)
    parser.add_argument("turbo_m", type=Path)
    parser.add_argument("--source-root", type=Path,
                        help="campaign checkout for raw paths recorded by TuRBO-m")
    args = parser.parse_args(argv)
    try:
        check(args.turbo_one, args.turbo_m, args.source_root)
    except B5AuditError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
