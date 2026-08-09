#!/usr/bin/env python3
"""Independently audit a Landreman FortBO bridge ledger."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional


class LandremanFortBOCheckError(ValueError):
    """The FortBO Landreman ledger is not internally consistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LandremanFortBOCheckError(message)


def _load(path: Path) -> Mapping[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LandremanFortBOCheckError(f"cannot read ledger: {error}") from error
    _require(isinstance(document, dict), "ledger root must be an object")
    _require(document.get("schema_name") == "fortbo.landreman-value-only",
             "wrong Landreman FortBO schema")
    _require(document.get("schema_version") == 1, "wrong Landreman FortBO schema version")
    _require(document.get("passed") is True, "ledger did not pass its bridge run")
    return document


def _check_commit(value: Any, label: str, length: int = 40) -> None:
    _require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{%d}" % length, value) is not None,
             f"{label} is not a pinned hexadecimal revision")


def check(
    path: Path,
    *,
    expected_budget: Optional[int] = None,
    expected_workers: Optional[int] = None,
    expected_archive_sha256: Optional[str] = None,
) -> dict[str, Any]:
    document = _load(path)
    source = document.get("source", {})
    _require(isinstance(source, dict), "source metadata is missing")
    _check_commit(source.get("fortbo_commit"), "FortBO commit")
    _check_commit(source.get("alpha_opt_commit"), "alpha_opt commit")
    archive_sha256 = source.get("archive_sha256")
    _require(isinstance(archive_sha256, str) and re.fullmatch(r"[0-9a-f]{64}", archive_sha256),
             "archive SHA-256 is missing")
    if expected_archive_sha256 is not None:
        _require(archive_sha256 == expected_archive_sha256, "archive SHA-256 differs")

    problem = document.get("problem", {})
    configuration = document.get("configuration", {})
    _require(isinstance(problem, dict), "problem metadata is missing")
    _require(isinstance(configuration, dict), "configuration metadata is missing")
    _require(problem.get("case_id") == "landreman-alpha-loss", "wrong Landreman case")
    dimension = problem.get("dimension")
    _require(dimension == 20, "Landreman bridge dimension is not 20")
    _require(problem.get("failure_policy") ==
             "FAIL retained by FortBO and excluded from surrogate",
             "failure policy is not recorded")
    budget = configuration.get("budget")
    workers = configuration.get("workers")
    _require(isinstance(budget, int) and budget > 0, "invalid budget")
    _require(isinstance(workers, int) and workers > 0, "invalid worker count")
    _require(configuration.get("mpi_ranks") == workers + 1,
             "MPI rank count is not manager plus workers")
    if expected_budget is not None:
        _require(budget == expected_budget, "budget differs from requested audit budget")
    if expected_workers is not None:
        _require(workers == expected_workers, "worker count differs from requested audit count")

    evaluations = document.get("evaluations")
    _require(isinstance(evaluations, list) and len(evaluations) == budget,
             "evaluation count does not equal the budget")
    _require(all(isinstance(row, dict) for row in evaluations),
             "evaluation row is not an object")
    ids = [row.get("candidate_id") for row in evaluations if isinstance(row, dict)]
    _require(ids == list(range(budget)), "candidate IDs are not contiguous")
    successful: list[float] = []
    failed = 0
    for row in evaluations:
        _require(isinstance(row, dict), "evaluation row is not an object")
        point = row.get("unit_x")
        _require(isinstance(point, list) and len(point) == dimension,
                 "evaluation point has the wrong shape")
        _require(all(isinstance(value, (int, float)) and math.isfinite(float(value))
                     and 0.0 <= float(value) <= 1.0 for value in point),
                 "evaluation point is outside the unit box")
        _require(row.get("region_id") == 1, "Landreman bridge used more than one region")
        worker = row.get("worker_rank")
        _require(isinstance(worker, int) and 1 <= worker <= workers,
                 "evaluation has an invalid worker rank")
        status = row.get("status")
        if status == "ok":
            value = row.get("value")
            _require(isinstance(value, (int, float)) and math.isfinite(float(value)),
                     "successful evaluation has no finite value")
            successful.append(float(value))
        elif status == "failed":
            _require(row.get("value") is None, "failed evaluation invented a value")
            failed += 1
        else:
            raise LandremanFortBOCheckError(f"unknown evaluation status: {status!r}")

    result = document.get("result", {})
    _require(isinstance(result, dict), "result metadata is missing")
    _require(result.get("truth_calls") == budget, "truth-call count is inconsistent")
    _require(result.get("failed_evaluations") == failed,
             "failed-evaluation count is inconsistent")
    expected_best = min(successful) if successful else None
    actual_best = result.get("best_value")
    if expected_best is None:
        _require(actual_best is None, "best value exists without a successful call")
    else:
        _require(isinstance(actual_best, (int, float)) and math.isfinite(float(actual_best)),
                 "best value is not finite")
        _require(math.isclose(float(actual_best), expected_best, rel_tol=1e-12, abs_tol=1e-12),
                 "best value is not the minimum successful value")
    completion_order = result.get("completion_order")
    _require(isinstance(completion_order, list) and
             sorted(completion_order) == list(range(budget)),
             "completion order is not a permutation of candidate IDs")
    return {
        "status": "PASS",
        "budget": budget,
        "workers": workers,
        "successful": len(successful),
        "failed": failed,
        "best_value": actual_best,
        "archive_sha256": archive_sha256,
    }


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledger", type=Path)
    parser.add_argument("--budget", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--archive-sha256")
    args = parser.parse_args(argv)
    try:
        result = check(
            args.ledger,
            expected_budget=args.budget,
            expected_workers=args.workers,
            expected_archive_sha256=args.archive_sha256,
        )
    except LandremanFortBOCheckError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
