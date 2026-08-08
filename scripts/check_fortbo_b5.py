#!/usr/bin/env python3
"""Audit FortBO B5 value-only rows and their paired comparison document."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional


CASE_ID = "b5_simple_to_build_qi"
UPSTREAM_COMMIT = "112b20ae07193910d467d26033fe51022e641b9f"
SEEDS = tuple(range(1, 6))
WORKERS = 8
BUDGET = 256


class FortBOB5AuditError(ValueError):
    """A FortBO B5 row or comparison violates its evidence contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FortBOB5AuditError(message)


def _load(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FortBOB5AuditError(f"cannot read {path}: {error}") from error
    _require(isinstance(value, dict), f"{path} must contain an object")
    return value


def check_row(path: Path, *, budget: int = BUDGET) -> Mapping[str, Any]:
    document = _load(path)
    label = str(path)
    _require(document.get("schema_name") == "fortbo.b5-value-only",
             f"{label}: wrong schema")
    _require(document.get("schema_version") == 1, f"{label}: schema version")
    _require(document.get("passed") is True, f"{label}: row is not passed")

    source = document.get("source", {})
    for name in ("fortbo_commit", "simsopt_dfo_commit", "constellaration_commit"):
        _require(isinstance(source.get(name), str) and len(source[name]) == 40,
                 f"{label}: {name} is not a full object id")
    _require(source.get("fortbo_dirty") is False, f"{label}: FortBO is dirty")
    _require(source.get("constellaration_commit") == UPSTREAM_COMMIT,
             f"{label}: wrong ConStellaration commit")

    problem = document.get("problem", {})
    mode = problem.get("mode")
    _require(problem.get("case_id") == CASE_ID, f"{label}: wrong B5 case")
    _require(mode in {"raw", "data-informed"}, f"{label}: wrong mode")
    dimension = 80 if mode == "raw" else 20
    _require(problem.get("dimension") == dimension,
             f"{label}: wrong dimension for {mode}")

    configuration = document.get("configuration", {})
    regions = configuration.get("regions")
    _require(regions in {1, 4}, f"{label}: invalid region count")
    if regions == 4:
        _require(mode == "data-informed", f"{label}: TuRBO-m is not data-informed")
    _require(configuration.get("budget") == budget, f"{label}: wrong budget")
    _require(configuration.get("workers") == WORKERS,
             f"{label}: worker count is not eight")
    _require(configuration.get("batch_size") == WORKERS,
             f"{label}: batch size is not eight")
    _require(configuration.get("seed") in SEEDS, f"{label}: invalid seed")

    rows = document.get("evaluations")
    _require(isinstance(rows, list) and len(rows) == budget,
             f"{label}: expected {budget} rows")
    for index, row in enumerate(rows):
        _require(isinstance(row, dict), f"{label}: row {index} is malformed")
        _require(row.get("candidate_id") == index,
                 f"{label}: candidate ids are not contiguous")
        unit = row.get("unit_x")
        scaled = row.get("scaled_x")
        _require(isinstance(unit, list) and len(unit) == dimension,
                 f"{label}: row {index} has wrong unit width")
        _require(isinstance(scaled, list) and len(scaled) == dimension,
                 f"{label}: row {index} has wrong scaled width")
        _require(all(isinstance(value, (int, float)) and math.isfinite(value)
                     and 0.0 <= value <= 1.0 for value in unit),
                 f"{label}: row {index} has invalid unit coordinates")
        _require(all(isinstance(value, (int, float)) and math.isfinite(value)
                     for value in scaled), f"{label}: row {index} has invalid scaled coordinates")
        _require(row.get("status") in {"ok", "failed"},
                 f"{label}: row {index} has invalid status")
        if row["status"] == "ok":
            _require(isinstance(row.get("value"), (int, float)) and
                     math.isfinite(row["value"]),
                     f"{label}: successful row {index} lacks a value")
        else:
            _require(row.get("value") is None,
                     f"{label}: failed row {index} invented a value")
        if regions == 4:
            _require(row.get("region_id") in range(1, regions + 1),
                     f"{label}: invalid region id")

    result = document.get("result", {})
    _require(result.get("truth_calls") == budget, f"{label}: wrong truth-call count")
    _require(result.get("peak_concurrency") == WORKERS,
             f"{label}: wrong peak concurrency")
    _require(result.get("failed_evaluations") == sum(
        row["status"] == "failed" for row in rows),
             f"{label}: failure count disagrees with rows")
    return document


def check_comparison(path: Path, *, row_root: Optional[Path] = None) -> None:
    document = _load(path)
    _require(document.get("schema_name") == "fortbo.b5-value-only-comparison",
             f"{path}: wrong comparison schema")
    _require(document.get("schema_version") == 1, f"{path}: comparison version")
    _require(document.get("passed") is True, f"{path}: comparison is not passed")
    configuration = document.get("configuration", {})
    _require(configuration.get("case_id") == CASE_ID, f"{path}: wrong case")
    _require(configuration.get("budget") == BUDGET, f"{path}: wrong budget")
    _require(configuration.get("workers") == WORKERS, f"{path}: wrong workers")
    _require(configuration.get("paired_seeds") == list(SEEDS),
             f"{path}: wrong paired seeds")

    records = document.get("raw_documents")
    _require(isinstance(records, list) and len(records) == 15,
             f"{path}: expected fifteen FortBO rows")
    seen = set()
    for record in records:
        _require(isinstance(record, dict), f"{path}: malformed row reference")
        key = (record.get("method"), record.get("mode"), record.get("seed"))
        _require(key not in seen, f"{path}: duplicate row {key}")
        seen.add(key)
        expected = {"turbo-1", "turbo-m-4"}
        _require(record.get("method") in expected, f"{path}: invalid method")
        _require(record.get("mode") in {"raw", "data-informed"},
                 f"{path}: invalid mode")
        _require(record.get("seed") in SEEDS, f"{path}: invalid seed")
        row_path = Path(record.get("file", ""))
        if row_root is not None and not row_path.is_absolute():
            row_path = row_root / row_path
        row = check_row(row_path)
        _require(row["problem"]["mode"] == record["mode"],
                 f"{row_path}: mode disagrees with comparison")
        _require(row["configuration"]["seed"] == record["seed"],
                 f"{row_path}: seed disagrees with comparison")
        expected_method = "turbo-m-4" if row["configuration"]["regions"] == 4 else "turbo-1"
        _require(record["method"] == expected_method,
                 f"{row_path}: method disagrees with comparison")
    expected_keys = {
        (method, mode, seed)
        for method, mode in (("turbo-1", "raw"), ("turbo-1", "data-informed"),
                             ("turbo-m-4", "data-informed"))
        for seed in SEEDS
    }
    _require(seen == expected_keys, f"{path}: paired row set is incomplete")
    accounting = document.get("accounting", {})
    _require(accounting.get("truth_calls_per_run") == BUDGET,
             f"{path}: wrong per-run accounting")
    _require(accounting.get("total_truth_calls") == 15 * BUDGET,
             f"{path}: wrong total accounting")


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--comparison", action="store_true")
    parser.add_argument("--row-root", type=Path)
    parser.add_argument("--budget", type=int, default=BUDGET)
    args = parser.parse_args(argv)
    try:
        if args.comparison:
            check_comparison(args.path, row_root=args.row_root)
            print("FortBO B5 comparison: PASS (15 rows, 256 calls, 8 workers)")
        else:
            check_row(args.path, budget=args.budget)
            print(f"FortBO B5 row: PASS ({args.path})")
    except FortBOB5AuditError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
