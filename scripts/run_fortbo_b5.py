#!/usr/bin/env python3
"""Run one value-only FortBO B5 TuRBO row against the pinned evaluator.

FortBO proposes either batches or one point at a time through a line-oriented
ask/tell executable. The truth calls are owned by this Python process and run
in eight independent worker threads. A failed truth call is sent back as
``FAIL`` rather than as a fabricated objective value.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from pathlib import Path
from typing import Any

import numpy as np


FORTBO_ROOT = Path(__file__).resolve().parents[1]
DFO_ROOT = Path(os.environ.get("SIMSOPT_DFO_SOURCE", "/mnt/storage/code/simsopt-dfo"))
sys.path.insert(0, str(DFO_ROOT / "src"))

from simsopt_dfo.b5_constellaration import (  # noqa: E402
    B5_BOX_HALF_WIDTH,
    B5_CASE_ID,
    B5_UPSTREAM_COMMIT,
    b5_scaled_start,
    evaluate_scaled_candidate,
    require_pinned_constellaration,
)
from simsopt_dfo.b5_data_informed import load_transform  # noqa: E402
from simsopt_dfo.evaluator import ExpectedEvaluationFailure  # noqa: E402
from simsopt_dfo.schema import load_json_document  # noqa: E402


BUDGET = 256
WORKERS = 8
SEEDS = (1, 2, 3, 4, 5)
TRANSFORM = Path("results/constellaration/b5-data-transform.json")


def main() -> int:
    args = _parser().parse_args()
    if args.output.resolve().is_relative_to(FORTBO_ROOT.resolve()):
        raise ValueError("B5 output must be outside the FortBO source tree")
    if args.scratch.resolve().is_relative_to(FORTBO_ROOT.resolve()):
        raise ValueError("B5 scratch must be outside the FortBO source tree")
    if args.output.exists():
        raise ValueError(f"output already exists: {args.output}")
    if args.mode == "data-informed" and args.regions != 1:
        # TuRBO-m is intentionally data-informed in the published B5 control.
        pass
    if args.regions == 4 and args.mode != "data-informed":
        raise ValueError("FortBO B5 TuRBO-m requires data-informed coordinates")

    fortbo_commit = _clean_commit(FORTBO_ROOT)
    dfo_commit = _clean_commit(DFO_ROOT)
    require_pinned_constellaration()
    mapper, dimension, transform_digest = _mapper(args.mode)
    result = _run(args, mapper, dimension)
    document = _document(
        args,
        result,
        mapper,
        dimension,
        transform_digest,
        fortbo_commit,
        dfo_commit,
    )
    if _clean_commit(FORTBO_ROOT) != fortbo_commit or _clean_commit(DFO_ROOT) != dfo_commit:
        raise RuntimeError("source checkout changed during the B5 run")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("raw", "data-informed"), required=True)
    parser.add_argument("--regions", type=int, choices=(1, 4), default=1)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=BUDGET)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--fortbo-command", default="fo")
    parser.add_argument(
        "--completion-driven",
        action="store_true",
        help="use one-point asynchronous completion-driven ask/tell",
    )
    return parser


def _clean_commit(root: Path) -> str:
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError(f"source checkout is dirty: {root}")
    return commit


def _mapper(mode: str):
    start = np.asarray(b5_scaled_start(), dtype=np.float64)
    if mode == "raw":
        return lambda unit: start - B5_BOX_HALF_WIDTH + np.asarray(unit), 80, None
    path = DFO_ROOT / TRANSFORM
    document = load_json_document(path)
    transform = load_transform(document, path.parent)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return transform.to_scaled, 20, digest


def _run(args: argparse.Namespace, mapper, dimension: int) -> dict[str, Any]:
    if args.completion_driven:
        return _run_completion_driven(args, mapper, dimension)
    if args.budget < 1 or args.budget % 8:
        raise ValueError("budget must be a positive multiple of FortBO batch size 8")
    if args.workers != WORKERS:
        raise ValueError("B5 parity requires eight workers")
    initial = 160 if args.mode == "raw" else 40
    if args.regions == 4:
        initial = 40
    command = [
        args.fortbo_command,
        "exec",
        "fortbo_b5_protocol",
        str(dimension),
        str(args.budget),
        str(initial),
        str(args.seed),
        str(args.regions),
        str(WORKERS),
    ]
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        cwd=FORTBO_ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    records: list[dict[str, Any]] = []
    best = float("inf")
    peak_concurrency = 0
    candidate_id = 0
    try:
        while True:
            line = _next_protocol_line(process.stdout, "ASK")
            if line.startswith("DONE "):
                break
            parts = line.split()
            if len(parts) != 2 or parts[0] != "ASK":
                raise RuntimeError(f"malformed FortBO protocol line: {line!r}")
            batch_size = int(parts[1])
            points: list[np.ndarray] = []
            regions: list[int] = []
            for _ in range(batch_size):
                point_line = process.stdout.readline()
                if not point_line:
                    raise RuntimeError("FortBO protocol ended while reading points")
                point_parts = point_line.split()
                if point_parts[0] != "POINT" or len(point_parts) != dimension + 3:
                    raise RuntimeError(f"malformed FortBO point line: {point_line!r}")
                points.append(np.asarray(point_parts[3:], dtype=np.float64))
                regions.append(int(point_parts[1]))

            batch_ids = list(range(candidate_id, candidate_id + batch_size))
            candidate_id += batch_size
            peak_concurrency = max(peak_concurrency, min(WORKERS, batch_size))
            results: dict[int, Any] = {}
            with ThreadPoolExecutor(max_workers=WORKERS) as executor:
                futures = {
                    executor.submit(_evaluate, mapper, points[i], args.scratch, batch_ids[i]): batch_ids[i]
                    for i in range(batch_size)
                }
                for future in as_completed(futures):
                    results[futures[future]] = future.result()

            process.stdin.write(f"TELL {batch_size}\n")
            for i, item in enumerate(batch_ids):
                outcome = results[item]
                if isinstance(outcome, ExpectedEvaluationFailure):
                    process.stdin.write(f"FAIL {i + 1}\n")
                    record = _record(item, points[i], regions[i], None, outcome, best)
                else:
                    process.stdin.write(f"VALUE {i + 1} {outcome.objective:.17g}\n")
                    best = min(best, outcome.minimized_objective)
                    record = _record(item, points[i], regions[i], outcome, None, best)
                records.append(record)
            process.stdin.flush()
    finally:
        if process.stdin is not None:
            process.stdin.close()
    stderr = process.stderr.read() if process.stderr is not None else ""
    return_code = process.wait()
    if return_code:
        raise RuntimeError(f"FortBO protocol failed ({return_code}): {stderr[-2000:]}")
    if candidate_id != args.budget:
        raise RuntimeError(f"FortBO returned {candidate_id} points, expected {args.budget}")
    return {
        "evaluations": records,
        "wall_seconds": time.perf_counter() - started,
        "peak_concurrency": peak_concurrency,
    }


def _run_completion_driven(
    args: argparse.Namespace, mapper, dimension: int
) -> dict[str, Any]:
    """Run one-point asks while keeping up to eight evaluator futures active."""

    if args.budget < 1:
        raise ValueError("budget must be positive")
    if args.workers != WORKERS:
        raise ValueError("B5 parity requires eight workers")
    initial = 160 if args.mode == "raw" else 40
    if args.regions == 4:
        initial = 40
    command = [
        args.fortbo_command,
        "exec",
        "fortbo_b5_completion_protocol",
        str(dimension),
        str(args.budget),
        str(initial),
        str(args.seed),
        str(args.regions),
        str(WORKERS),
    ]
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        cwd=FORTBO_ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    pending: dict[int, tuple[Any, np.ndarray, int]] = {}
    records: dict[int, dict[str, Any]] = {}
    completion_order: list[int] = []
    best = float("inf")
    peak_concurrency = 0
    dispatched = 0

    def complete_one() -> None:
        nonlocal best
        futures = [item[0] for item in pending.values()]
        done, _ = wait(futures, return_when=FIRST_COMPLETED)
        future = next(iter(done))
        candidate_id = next(
            item_id for item_id, item in pending.items() if item[0] is future
        )
        _, point, region = pending.pop(candidate_id)
        outcome = future.result()
        process.stdin.write(f"TELL {candidate_id}\n")
        if isinstance(outcome, ExpectedEvaluationFailure):
            process.stdin.write(f"FAIL {candidate_id}\n")
            record = _record(candidate_id, point, region, None, outcome, best)
        else:
            process.stdin.write(f"VALUE {candidate_id} {outcome.objective:.17g}\n")
            best = min(best, outcome.minimized_objective)
            record = _record(candidate_id, point, region, outcome, None, best)
        process.stdin.flush()
        records[candidate_id] = record
        completion_order.append(candidate_id)

    try:
        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            while True:
                if dispatched < args.budget and len(pending) < WORKERS:
                    line = _next_protocol_line(process.stdout, "ASK")
                    if line == "WAIT":
                        if not pending:
                            raise RuntimeError("FortBO requested a wait without pending work")
                        complete_one()
                        continue
                    if line.startswith("DONE "):
                        raise RuntimeError("FortBO completed before its budget")
                    parts = line.split()
                    if parts != ["ASK", "1"]:
                        raise RuntimeError(f"malformed completion ASK: {line!r}")
                    point_line = process.stdout.readline()
                    if not point_line:
                        raise RuntimeError("FortBO ended while reading a point")
                    point_parts = point_line.split()
                    if point_parts[0] != "POINT" or len(point_parts) != dimension + 3:
                        raise RuntimeError(f"malformed completion point: {point_line!r}")
                    candidate_id = int(point_parts[1])
                    if candidate_id != dispatched:
                        raise RuntimeError(
                            f"FortBO candidate ids are not contiguous: {candidate_id}"
                        )
                    region = int(point_parts[2])
                    point = np.asarray(point_parts[3:], dtype=np.float64)
                    pending[candidate_id] = (
                        executor.submit(
                            _evaluate, mapper, point, args.scratch, candidate_id
                        ),
                        point,
                        region,
                    )
                    dispatched += 1
                    peak_concurrency = max(peak_concurrency, len(pending))
                    continue
                if pending:
                    complete_one()
                    continue
                line = _next_protocol_line(process.stdout, "DONE")
                if not line.startswith("DONE "):
                    raise RuntimeError(f"malformed completion DONE: {line!r}")
                break
    except Exception as error:
        if process.stdin is not None:
            process.stdin.close()
        stderr = process.stderr.read() if process.stderr is not None else ""
        process.wait()
        detail = stderr[-2000:].strip()
        if detail:
            raise RuntimeError(
                f"FortBO completion protocol aborted: {detail}"
            ) from error
        raise
    finally:
        if process.stdin is not None:
            process.stdin.close()
    stderr = process.stderr.read() if process.stderr is not None else ""
    return_code = process.wait()
    if return_code:
        raise RuntimeError(f"FortBO completion protocol failed ({return_code}): {stderr[-2000:]}")
    if dispatched != args.budget or len(records) != args.budget:
        raise RuntimeError(
            f"FortBO returned {len(records)} completions, expected {args.budget}"
        )
    return {
        "evaluations": [records[index] for index in range(args.budget)],
        "completion_order": completion_order,
        "wall_seconds": time.perf_counter() - started,
        "peak_concurrency": peak_concurrency,
    }


def _next_protocol_line(stream, expected: str) -> str:
    while True:
        line = stream.readline()
        if not line:
            raise RuntimeError(f"FortBO protocol ended before {expected}")
        stripped = line.strip()
        if stripped.startswith(("ASK ", "DONE ")) or stripped == "WAIT":
            return stripped


def _evaluate(mapper, unit: np.ndarray, scratch: Path, candidate_id: int):
    try:
        return evaluate_scaled_candidate(
            mapper(unit), scratch / f"candidate-{candidate_id:08d}"
        )
    except ExpectedEvaluationFailure as error:
        return error


def _record(candidate_id, point, region, evaluation, failure, best):
    return {
        "candidate_id": candidate_id,
        "unit_x": point.tolist(),
        "region_id": region,
        "status": "failed" if failure is not None else "ok",
        "value": None if failure is not None else evaluation.minimized_objective,
        "upstream_objective": None if failure is not None else evaluation.objective,
        "feasibility": None if failure is not None else evaluation.feasibility,
        "score": None if failure is not None else evaluation.score,
        "failure_kind": None if failure is None else str(failure),
        "best_so_far": None if not np.isfinite(best) else best,
    }


def _document(args, result, mapper, dimension, transform_digest, fortbo_commit, dfo_commit):
    rows = result["evaluations"]
    for row in rows:
        row["scaled_x"] = mapper(np.asarray(row["unit_x"], dtype=np.float64)).tolist()
    return {
        "schema_name": "fortbo.b5-value-only",
        "schema_version": 1,
        "source": {
            "fortbo_commit": fortbo_commit,
            "fortbo_dirty": False,
            "simsopt_dfo_commit": dfo_commit,
            "constellaration_commit": B5_UPSTREAM_COMMIT,
        },
        "problem": {
            "case_id": B5_CASE_ID,
            "mode": args.mode,
            "dimension": dimension,
            "transform_file": str(TRANSFORM) if transform_digest else None,
            "transform_sha256": transform_digest,
        },
        "configuration": {
            "method": (
                "completion-driven FortBO TuRBO-1 with Thompson sampling"
                if args.completion_driven and args.regions == 1
                else "completion-driven FortBO TuRBO-m with Thompson sampling"
                if args.completion_driven
                else "batched FortBO TuRBO-1 with Thompson sampling"
                if args.regions == 1
                else "batched FortBO TuRBO-m with Thompson sampling"
            ),
            "protocol": "completion-driven" if args.completion_driven else "batched",
            "seed": args.seed,
            "budget": args.budget,
            "workers": args.workers,
            "batch_size": 1 if args.completion_driven else WORKERS,
            "regions": args.regions,
            "initial_points_per_region": 40 if args.regions == 4 else None,
        },
        "evaluations": rows,
        "result": {
            "truth_calls": len(rows),
            "peak_concurrency": result["peak_concurrency"],
            "failed_evaluations": sum(row["status"] != "ok" for row in rows),
            "wall_seconds": result["wall_seconds"],
            "completion_order": result.get("completion_order"),
        },
        "passed": True,
    }


if __name__ == "__main__":
    raise SystemExit(main())
