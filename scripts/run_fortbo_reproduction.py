#!/usr/bin/env python3
"""Run a common-schema synthetic reproduction or an available policy adapter.

The analytic/delay lane is intentionally independent of FortBO. It provides a
deterministic evaluator and completion scheduler for checking the ledger ABI.
The ``fortbo`` lane invokes the repository's Fortran ask/tell driver, while
the optional ``botorch`` lane uses BoTorch only when that dependency is
installed. Unsupported real-physics cases are recorded as refusals rather
than being represented by a synthetic result.
"""

from __future__ import annotations

import argparse
import heapq
import hashlib
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from .fortbo_environment import resolve_fo_command
except ImportError:  # pragma: no cover - used when this file is run directly
    from fortbo_environment import resolve_fo_command


SCHEMA_VERSION = 1
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIMENSION = 2
DEFAULT_BUDGET = 16
DEFAULT_INITIAL = 4
DEFAULT_WORKERS = 2


class BackendUnavailable(RuntimeError):
    """A requested optional or not-yet-adapted backend cannot run."""


@dataclass
class Job:
    evaluation: int
    point: np.ndarray
    worker: int
    submitted_at: int
    completed_at: int


def quadratic(point: np.ndarray) -> float:
    """Independent minimization oracle: the minimum is known to be zero."""
    values = np.asarray(point, dtype=np.float64)
    return float(np.sum((values - 0.25) ** 2))


def delay_for(evaluation: int, seed: int) -> int:
    """A deterministic evaluator delay, independent of every policy backend."""
    return 1 + ((seed + 7 * evaluation) % 4)


def objective_metadata(dimension: int) -> Dict[str, Any]:
    return {
        "name": "unit_quadratic",
        "dimension": dimension,
        "bounds": [[0.0] * dimension, [1.0] * dimension],
        "optimum": [0.25] * dimension,
        "optimal_value": 0.0,
        "evaluation_oracle": "sum((x - 0.25)**2)",
    }


def _base_payload(args: argparse.Namespace, status: str = "complete") -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run": {
            "case": args.case,
            "implementation": args.implementation,
            "seed": args.seed,
            "workers": args.workers,
            "requested_budget": args.budget,
            "initial_design": args.initial,
            "acquisition": args.acquisition,
            "frozen_candidates": (str(args.frozen_candidates)
                                  if args.frozen_candidates else None),
            "frozen_initial_design": (str(args.frozen_initial)
                                      if args.frozen_initial else None),
            "status": status,
        },
        "objective": objective_metadata(args.dimension),
        "ledger": [],
        "summary": {
            "evaluations": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "best_value": None,
            "best_evaluation": None,
        },
        "provenance": {
            "runner": "scripts/run_fortbo_reproduction.py",
            "schema_version": SCHEMA_VERSION,
            "repository": str(ROOT),
        },
    }


def _finish_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    ledger = payload["ledger"]
    payload["summary"]["evaluations"] = len(ledger)
    payload["summary"]["successful_calls"] = sum(
        row["status"] == "ok" for row in ledger
    )
    payload["summary"]["failed_calls"] = sum(
        row["status"] != "ok" for row in ledger
    )
    if ledger:
        best = min(ledger, key=lambda row: row["value"])
        payload["summary"]["best_value"] = best["value"]
        payload["summary"]["best_evaluation"] = best["evaluation"]
    return payload


def _append_row(payload: Dict[str, Any], *, evaluation: int,
                point: Sequence[float], value: float, worker: int,
                submitted_at: int, completed_at: int,
                completion_order: int, region: int = 1,
                trust: Optional[Dict[str, Any]] = None) -> None:
    previous = payload["ledger"][-1]["best_so_far"] if payload["ledger"] else math.inf
    payload["ledger"].append({
        "evaluation": evaluation,
        "point": [float(item) for item in point],
        "value": float(value),
        "status": "ok",
        "worker": worker,
        "region": region,
        "submitted_at": submitted_at,
        "completed_at": completed_at,
        "completion_order": completion_order,
        "best_so_far": min(previous, float(value)),
        "trust": trust or {
            "length": None,
            "success_counter": None,
            "failure_counter": None,
            "restarts": None,
        },
    })


def run_delay_oracle(args: argparse.Namespace) -> Dict[str, Any]:
    """Replay a delay-injected evaluator with deterministic completion order."""
    payload = _base_payload(args)
    generator = np.random.default_rng(args.seed)
    pending: List[Tuple[int, int, int, Job]] = []
    available = list(range(args.workers))
    next_evaluation = 0
    clock = 0
    completion_order = 0

    def submit(now: int, worker: int, evaluation: int) -> None:
        point = generator.uniform(0.0, 1.0, size=args.dimension)
        job = Job(evaluation, point, worker, now, now + delay_for(evaluation, args.seed))
        heapq.heappush(pending, (job.completed_at, job.evaluation, worker, job))

    while next_evaluation < args.budget and available:
        submit(clock, available.pop(0), next_evaluation)
        next_evaluation += 1

    while pending:
        _, _, worker, job = heapq.heappop(pending)
        clock = max(clock, job.completed_at)
        value = quadratic(job.point)
        _append_row(
            payload, evaluation=job.evaluation, point=job.point, value=value,
            worker=worker, submitted_at=job.submitted_at,
            completed_at=clock, completion_order=completion_order,
        )
        completion_order += 1
        available.append(worker)
        available.sort()
        if next_evaluation < args.budget:
            submit(clock, available.pop(0), next_evaluation)
            next_evaluation += 1

    payload["provenance"]["oracle"] = {
        "kind": "independent_analytic_delay_replay",
        "delay_formula": "1 + ((seed + 7*evaluation) % 4)",
        "random_generator": "numpy.default_rng",
    }
    return _finish_payload(payload)


def _trust_from_fortran(values: Sequence[float], integers: Sequence[int]) -> Dict[str, Any]:
    return {
        "length": float(values[0]),
        "success_counter": integers[0],
        "failure_counter": integers[1],
        "restarts": integers[2],
    }


def run_fortbo(args: argparse.Namespace) -> Dict[str, Any]:
    if args.case != "synthetic-quadratic":
        raise BackendUnavailable(
            "the FortBO adapter currently exposes only the synthetic quadratic "
            "ABI gate; physics evaluators remain outside this repository"
        )
    extra = [args.acquisition]
    if args.frozen_candidates:
        extra.append(str(args.frozen_candidates))
    if args.frozen_initial:
        extra.append(str(args.frozen_candidates) if args.frozen_candidates else "-")
        extra.append(str(args.frozen_initial))
    fo_command = resolve_fo_command(args.fortbo_command or "fo")
    command = [
        fo_command, "exec", "fortbo_reproduction",
        str(args.dimension), str(args.budget), str(args.initial), str(args.seed),
        *extra,
    ]
    if args.fortbo_command:
        command = [
            fo_command, str(args.dimension), str(args.budget),
            str(args.initial), str(args.seed), *extra,
        ]
    started = time.perf_counter()
    result = subprocess.run(
        command, cwd=ROOT, check=False, capture_output=True, text=True,
        timeout=args.timeout,
    )
    if result.returncode:
        raise RuntimeError(
            "FortBO adapter failed:\n" + result.stdout[-2000:] + result.stderr[-2000:]
        )

    payload = _base_payload(args)
    payload["provenance"]["adapter"] = {
        "command": command,
        "wall_seconds": time.perf_counter() - started,
        "objective_owner": "Python runner and Fortran adapter use the same formula",
    }
    if args.frozen_candidates:
        raw = args.frozen_candidates.read_bytes()
        payload["provenance"]["adapter"]["frozen_candidate_file"] = {
            "path": str(args.frozen_candidates),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    if args.frozen_initial:
        raw = args.frozen_initial.read_bytes()
        payload["provenance"]["adapter"]["frozen_initial_file"] = {
            "path": str(args.frozen_initial),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    for line in result.stdout.splitlines():
        fields = line.split()
        if not fields or fields[0] != "ROW":
            continue
        expected = 3 + args.dimension + 3 + 3
        if len(fields) != expected:
            raise RuntimeError(f"malformed FortBO row with {len(fields)} fields: {line}")
        evaluation = int(fields[1])
        region = int(fields[2])
        point_start = 3
        point_end = point_start + args.dimension
        point = [float(item) for item in fields[point_start:point_end]]
        value = float(fields[point_end])
        best = float(fields[point_end + 1])
        trust_values = [float(fields[point_end + 2])]
        trust_integers = [int(item) for item in fields[point_end + 3:]]
        _append_row(
            payload, evaluation=evaluation, point=point, value=value,
            worker=evaluation % args.workers, submitted_at=evaluation,
            completed_at=evaluation, completion_order=evaluation, region=region,
            trust=_trust_from_fortran(trust_values, trust_integers),
        )
        if abs(payload["ledger"][-1]["best_so_far"] - best) > 1.0e-12:
            raise RuntimeError("FortBO adapter's best value disagrees with its ledger")
    if len(payload["ledger"]) != args.budget:
        raise RuntimeError(
            f"FortBO adapter returned {len(payload['ledger'])} rows, "
            f"expected {args.budget}"
        )
    return _finish_payload(payload)


def run_botorch(args: argparse.Namespace) -> Dict[str, Any]:
    if args.case != "synthetic-quadratic":
        raise BackendUnavailable(
            "the BoTorch adapter currently exposes only the synthetic quadratic "
            "ABI gate; physics evaluators remain outside this repository"
        )
    try:
        import torch
        from botorch.acquisition import ExpectedImprovement
        from botorch.fit import fit_gpytorch_mll
        from botorch.generation import MaxPosteriorSampling
        from botorch.models import SingleTaskGP
        from botorch.optim import optimize_acqf
        from gpytorch.mlls import ExactMarginalLogLikelihood
        from torch.quasirandom import SobolEngine
    except ImportError as error:
        raise BackendUnavailable(f"BoTorch adapter unavailable: {error}") from error

    torch.set_default_dtype(torch.float64)
    generator = torch.Generator().manual_seed(args.seed)
    sobol = SobolEngine(args.dimension, scramble=True, seed=args.seed)
    x = sobol.draw(args.initial)
    y = torch.tensor([quadratic(row.numpy()) for row in x], dtype=torch.float64).unsqueeze(-1)
    payload = _base_payload(args)
    payload["provenance"]["adapter"] = {
        "package": "botorch",
        "objective_owner": "Python runner",
    }
    for index in range(args.initial):
        _append_row(
            payload, evaluation=index, point=x[index].numpy(), value=float(y[index]),
            worker=index % args.workers, submitted_at=index,
            completed_at=index, completion_order=index,
        )
    while len(payload["ledger"]) < args.budget:
        train_y = -y
        model = SingleTaskGP(x, train_y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        if args.acquisition == "ei":
            acquisition = ExpectedImprovement(model, best_f=float(train_y.max()))
            next_x, _ = optimize_acqf(
                acquisition, bounds=torch.stack((torch.zeros(args.dimension),
                                                  torch.ones(args.dimension))),
                q=1, num_restarts=4, raw_samples=32,
            )
        else:
            candidates = torch.rand((256, args.dimension), generator=generator)
            sampler = MaxPosteriorSampling(model=model, replacement=False)
            with torch.no_grad():
                next_x = sampler(candidates, num_samples=1)
        next_y = torch.tensor([[quadratic(next_x[0].numpy())]], dtype=torch.float64)
        evaluation = len(payload["ledger"])
        x = torch.cat((x, next_x), dim=0)
        y = torch.cat((y, next_y), dim=0)
        _append_row(
            payload, evaluation=evaluation, point=next_x[0].numpy(),
            value=float(next_y[0, 0]), worker=evaluation % args.workers,
            submitted_at=evaluation, completed_at=evaluation,
            completion_order=evaluation,
        )
    return _finish_payload(payload)


def load_config(path: Optional[Path]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    if path is None:
        return None, None, None
    try:
        raw = path.read_bytes()
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read config {path}: {error}") from error
    return (data.get("config_id"), hashlib.sha256(raw).hexdigest(),
            data.get("fortbo_result_label"))


def refused_payload(args: argparse.Namespace, reason: str) -> Dict[str, Any]:
    payload = _base_payload(args, status="refused")
    payload["run"]["refusal_reason"] = reason
    payload["provenance"]["refusal"] = reason
    return payload


def execute(args: argparse.Namespace) -> Dict[str, Any]:
    config_id, config_digest, result_label = load_config(args.config)
    if args.case != "synthetic-quadratic":
        reason = (
            f"no evaluator adapter is registered for case {args.case!r}; "
            "physics evaluators remain outside this repository"
        )
        payload = refused_payload(args, reason)
    else:
        try:
            if args.implementation == "oracle":
                payload = run_delay_oracle(args)
            elif args.implementation == "fortbo":
                payload = run_fortbo(args)
            else:
                payload = run_botorch(args)
        except BackendUnavailable as error:
            payload = refused_payload(args, str(error))
    if config_id is not None:
        payload["provenance"]["config_id"] = config_id
        payload["provenance"]["config_sha256"] = config_digest
    if result_label is not None:
        payload["run"]["result_label"] = result_label
        payload["provenance"]["result_label"] = result_label
    return payload


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="synthetic-quadratic")
    parser.add_argument("--implementation", choices=("oracle", "botorch", "fortbo"),
                        required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    parser.add_argument("--initial", type=int, default=DEFAULT_INITIAL)
    parser.add_argument("--dimension", type=int, default=DEFAULT_DIMENSION)
    parser.add_argument("--acquisition", choices=("ei", "ts"), default="ts")
    parser.add_argument("--frozen-candidates", type=Path,
                        help="count/dimension candidate pool for exact replay")
    parser.add_argument("--frozen-initial", type=Path,
                        help="count/dimension initial design for exact replay")
    parser.add_argument("--scratch", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--fortbo-command",
                        help="compiled fortbo_reproduction executable instead of fpm")
    args = parser.parse_args(argv)
    if args.workers < 1 or args.budget < 1 or args.initial < 1 or args.dimension < 1:
        parser.error("workers, budget, initial, and dimension must be positive")
    if args.initial > args.budget:
        parser.error("initial must not exceed budget")
    return args


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    try:
        payload = execute(args)
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    else:
        sys.stdout.write(encoded)
    if args.scratch:
        args.scratch.mkdir(parents=True, exist_ok=True)
        (args.scratch / "run.json").write_text(encoded, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
