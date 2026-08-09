#!/usr/bin/env python3
"""Check that a concurrent original-control/FortBO pair is comparable."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional


class PairCheckError(ValueError):
    """The pair does not contain an independent oracle comparison."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PairCheckError(message)


def _load(path: Path) -> Mapping[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PairCheckError(f"cannot read pair: {error}") from error
    _require(isinstance(document, dict), "pair root must be an object")
    _require(document.get("schema_name") == "fortbo.oracle-pair", "wrong pair schema")
    _require(document.get("schema_version") == 1, "wrong pair schema version")
    _require(document.get("status") == "complete", "pair is not complete")
    return document


def _output_path(
    entry: Mapping[str, Any],
    rebase_from: Optional[Path],
    rebase_to: Optional[Path],
) -> Path:
    output = Path(entry.get("output", ""))
    if rebase_from is None:
        return output
    assert rebase_to is not None
    try:
        relative = output.resolve().relative_to(rebase_from.resolve())
    except ValueError as error:
        raise PairCheckError(
            f"{output} is outside the requested rebase root {rebase_from}"
        ) from error
    return rebase_to.resolve() / relative


def _check_run(
    pair: Mapping[str, Any],
    side: str,
    *,
    rebase_from: Optional[Path] = None,
    rebase_to: Optional[Path] = None,
) -> Mapping[str, Any]:
    entry = pair.get(side)
    _require(isinstance(entry, dict), f"missing {side} entry")
    output = _output_path(entry, rebase_from, rebase_to)
    _require(output.is_file(), f"missing {side} output: {output}")
    expected_digest = entry.get("sha256")
    if expected_digest:
        digest = hashlib.sha256(output.read_bytes()).hexdigest()
        _require(digest == expected_digest, f"{side} output digest changed")
    document = json.loads(output.read_text(encoding="utf-8"))
    _require(isinstance(document, dict) and document.get("passed") is True, f"{side} output did not pass")
    problem = document.get("problem", {})
    result = document.get("result", {})
    evaluations = document.get("evaluations")
    _require(isinstance(evaluations, list), f"{side} evaluations are missing")
    _require(result.get("truth_calls") == len(evaluations), f"{side} truth-call count is inconsistent")
    statuses = {row.get("status") for row in evaluations if isinstance(row, dict)}
    _require(statuses <= {"ok", "failed"}, f"{side} contains unknown statuses: {statuses}")
    successful = [row["value"] for row in evaluations if row.get("status") == "ok"]
    expected_best = min(successful) if successful else None
    actual_best = result.get("best_value")
    if expected_best is None:
        _require(actual_best is None, f"{side} reports a best value without successes")
    else:
        _require(math.isclose(float(actual_best), float(expected_best), rel_tol=1e-12, abs_tol=1e-12), f"{side} best value is not the minimum successful value")
    run_configuration = dict(document.get("configuration", {}))
    run_configuration.setdefault("regions", 1)
    return {
        "problem": problem,
        "configuration": run_configuration,
        "source": document.get("source", {}),
        "result": result,
    }


def check(
    path: Path,
    *,
    rebase_from: Optional[Path] = None,
    rebase_to: Optional[Path] = None,
) -> None:
    if (rebase_from is None) != (rebase_to is None):
        raise PairCheckError("rebase-from and rebase-to must be supplied together")
    pair = _load(path)
    configuration = pair.get("configuration", {})
    oracle = _check_run(pair, "oracle", rebase_from=rebase_from, rebase_to=rebase_to)
    fortbo = _check_run(pair, "fortbo", rebase_from=rebase_from, rebase_to=rebase_to)
    for key in ("mode", "regions", "seed", "budget", "workers"):
        _require(oracle["configuration"].get(key) == configuration.get(key), f"oracle {key} disagrees with pair")
        _require(fortbo["configuration"].get(key) == configuration.get(key), f"fortbo {key} disagrees with pair")
    _require(oracle["problem"].get("case_id") == fortbo["problem"].get("case_id"), "case IDs differ")
    _require(oracle["problem"].get("dimension") == fortbo["problem"].get("dimension"), "dimensions differ")
    _require(
        oracle["source"].get("constellaration_commit")
        == fortbo["source"].get("constellaration_commit"),
        "the oracle and FortBO use different ConStellaration commits",
    )
    oracle_best = oracle["result"].get("best_value")
    fortbo_best = fortbo["result"].get("best_value")
    delta = None if oracle_best is None or fortbo_best is None else float(fortbo_best) - float(oracle_best)
    print(json.dumps({"status": "PASS", "oracle_best_value": oracle_best, "fortbo_best_value": fortbo_best, "fortbo_minus_oracle": delta}, sort_keys=True))


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pair", type=Path)
    parser.add_argument("--rebase-from", type=Path,
                        help="remote run root embedded in pair output paths")
    parser.add_argument("--rebase-to", type=Path,
                        help="local run root containing archived output files")
    args = parser.parse_args(argv)
    try:
        check(args.pair, rebase_from=args.rebase_from, rebase_to=args.rebase_to)
    except PairCheckError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
