#!/usr/bin/env python3
"""Run FortBO against the archived Landreman objective under MPI.

Rank zero owns the FortBO ask/tell process.  The remaining ranks each build an
independent copy of the archived VMEC/PCA objective, matching the original
driver's one-manager/four-worker layout.  This script is a bridge only: it
must be launched inside the historical four-GPU Slurm allocation, with the
prepared ``software/alpha_opt`` tree and its pinned Python environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, TextIO

import numpy as np

try:
    from .fortbo_environment import resolve_fo_command
except ImportError:  # pragma: no cover - used when this file is run directly
    from fortbo_environment import resolve_fo_command


ROOT = Path(__file__).resolve().parents[1]
WORK_TAG = 1
RESULT_TAG = 2
STOP_TAG = 3
DEFAULT_DIMENSION = 20
DEFAULT_BUDGET = 10_000
DEFAULT_INITIAL = 40
DEFAULT_WORKERS = 4
FAIL_VALUE = 5.5
DMERC_FAIL_VALUE = -0.5


class LandremanFortBOError(RuntimeError):
    """The archived evaluator bridge could not complete safely."""


def _protocol_command(
    fortbo_command: str,
    dimension: int,
    budget: int,
    initial: int,
    seed: int,
    workers: int,
) -> list[str]:
    return [
        fortbo_command,
        "exec",
        "fortbo_b5_completion_protocol",
        str(dimension),
        str(budget),
        str(initial),
        str(seed),
        "1",
        str(workers),
    ]


def _git_revision(path: Path) -> Optional[str]:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _clean_revision(path: Path) -> str:
    revision = _git_revision(path)
    if not revision:
        raise LandremanFortBOError(f"cannot identify Git revision: {path}")
    dirty = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"],
        check=False,
        capture_output=True,
        text=True,
    )
    if dirty.returncode or dirty.stdout.strip():
        raise LandremanFortBOError(f"source checkout is dirty: {path}")
    return revision


def _build_objective(alpha_root: Path, worker_rank: int):
    """Build one isolated copy of the archived objective on a worker rank."""
    alpha_root = alpha_root.resolve()
    os.chdir(alpha_root)
    sys.path.insert(0, str(alpha_root))
    try:
        from vmecpp.simsopt_compat import Vmec
        from alpha_opt import SurfaceWeightedPCA, get_worst_DMerc_normalized
        from alpha_opt.gpu_tracing import compute_alpha_loss
        from alpha_opt.objective import get_objective
    except ImportError as error:
        raise LandremanFortBOError(
            f"archived Landreman Python environment is incomplete: {error}"
        ) from error

    aspect_ratio = 6.0
    minor_radius = 3.1 / (aspect_ratio**0.38)
    major_radius = minor_radius * aspect_ratio
    max_b_target = 12.0
    max_b_iterations = 1
    phiedge_estimate = np.pi * (max_b_target / np.sqrt(2.0)) * minor_radius**2
    phiedge_high = phiedge_estimate * 2.0
    pca_data_file = alpha_root / (
        "data/20260402-01_prepare_weighted_data_nfpAtLeast3_PCA.h5"
    )
    if not pca_data_file.is_file():
        raise LandremanFortBOError(f"archived PCA data is missing: {pca_data_file}")

    vmec = Vmec("input.vmec", verbose=False)
    vmec.set("phiedge", phiedge_high)
    surface = SurfaceWeightedPCA(
        vmec.indata.nfp,
        major_radius,
        minor_radius,
        DEFAULT_DIMENSION,
        filename=str(pca_data_file),
    )
    x_scale = np.ones(DEFAULT_DIMENSION)

    def raw_objective() -> float:
        wout_filename = f"wout_tmp_rank{worker_rank}.nc"
        vmec.wout.save(wout_filename)
        return float(
            compute_alpha_loss(
                wout_filename,
                n_particles=25_000,
                t_max=1.0e-1,
                tau=0.1,
                min_dt=1.0e-9,
                maxloss=0.02,
                t_block=1.0e-3,
                tol=1.0e-6,
                vacuum=False,
            )
        )

    objective = get_objective(
        vmec,
        surface,
        x_scale,
        raw_objective,
        fail_val=FAIL_VALUE,
        max_B=max_b_target,
        max_B_iterations=max_b_iterations,
        phiedge=phiedge_high,
    )
    return vmec, objective, get_worst_DMerc_normalized


def _evaluate(
    objective: Any,
    worst_dmerc: Any,
    vmec: Any,
    scratch: Path,
    worker_rank: int,
    candidate_id: int,
    point: np.ndarray,
) -> dict[str, Any]:
    original_cwd = Path.cwd()
    evaluation_dir = scratch / f"eval{candidate_id:06d}"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    try:
        os.chdir(evaluation_dir)
        value = float(objective(np.asarray(point, dtype=np.float64)))
        if not np.isfinite(value) or value == FAIL_VALUE:
            raise LandremanFortBOError(
                "archived objective returned a non-finite or failure value"
            )
        dmerc = float(worst_dmerc(vmec, 0.2, 0.95))
        return {
            "status": "ok",
            "value": value,
            "worst_dmerc": dmerc,
            "worker_rank": worker_rank,
        }
    except Exception as error:  # evaluator failures are data, not a fake value
        return {
            "status": "failed",
            "value": None,
            "worst_dmerc": DMERC_FAIL_VALUE,
            "worker_rank": worker_rank,
            "failure": f"{type(error).__name__}: {error}",
        }
    finally:
        os.chdir(original_cwd)


def _worker_loop(alpha_root: Path, scratch: Path, comm: Any, worker_rank: int) -> None:
    from mpi4py import MPI

    initialization_error: Optional[str] = None
    try:
        vmec, objective, worst_dmerc = _build_objective(alpha_root, worker_rank)
    except Exception as error:
        initialization_error = f"worker {worker_rank}: {type(error).__name__}: {error}"
    initialization_errors = comm.allgather(initialization_error)
    if any(error is not None for error in initialization_errors):
        return
    while True:
        status = MPI.Status()
        comm.Probe(source=0, tag=MPI.ANY_TAG, status=status)
        if status.Get_tag() == STOP_TAG:
            comm.recv(source=0, tag=STOP_TAG)
            return
        if status.Get_tag() != WORK_TAG:
            raise LandremanFortBOError(
                f"worker {worker_rank} received unexpected MPI tag {status.Get_tag()}"
            )
        candidate_id, point = comm.recv(source=0, tag=WORK_TAG)
        result = _evaluate(
            objective,
            worst_dmerc,
            vmec,
            scratch,
            worker_rank,
            candidate_id,
            np.asarray(point, dtype=np.float64),
        )
        comm.send((candidate_id, result), dest=0, tag=RESULT_TAG)


def _next_protocol_line(stream: TextIO) -> str:
    while True:
        line = stream.readline()
        if not line:
            raise LandremanFortBOError("FortBO protocol ended before an expected line")
        stripped = line.strip()
        if stripped.startswith(("ASK ", "DONE ")) or stripped == "WAIT":
            return stripped


def _parse_point(line: str, dimension: int) -> tuple[int, int, np.ndarray]:
    fields = line.split()
    if len(fields) != dimension + 3 or fields[0] != "POINT":
        raise LandremanFortBOError(f"malformed FortBO point: {line!r}")
    try:
        candidate_id = int(fields[1])
        region = int(fields[2])
        point = np.asarray(fields[3:], dtype=np.float64)
    except ValueError as error:
        raise LandremanFortBOError(f"malformed FortBO point: {line!r}") from error
    if not np.all(np.isfinite(point)) or np.any(point < 0.0) or np.any(point > 1.0):
        raise LandremanFortBOError("FortBO proposed a point outside the unit box")
    return candidate_id, region, point


def _stop_workers(comm: Any, workers: int) -> None:
    for worker_rank in range(1, workers + 1):
        comm.send(None, dest=worker_rank, tag=STOP_TAG)


def _complete_one(
    comm: Any,
    process: subprocess.Popen[str],
    pending: dict[int, tuple[np.ndarray, int]],
    idle: list[int],
    records: dict[int, dict[str, Any]],
    completion_order: list[int],
    best: float,
) -> float:
    from mpi4py import MPI

    status = MPI.Status()
    while not comm.Iprobe(source=MPI.ANY_SOURCE, tag=RESULT_TAG, status=status):
        time.sleep(0.01)
    worker_rank = status.Get_source()
    candidate_id, result = comm.recv(source=worker_rank, tag=RESULT_TAG)
    if candidate_id not in pending:
        raise LandremanFortBOError(f"worker returned unknown candidate {candidate_id}")
    point, region = pending.pop(candidate_id)
    idle.append(worker_rank)
    idle.sort()
    if result["status"] == "ok":
        value = float(result["value"])
        best = min(best, value)
        process.stdin.write(f"TELL {candidate_id}\nVALUE {candidate_id} {value:.17g}\n")
        row_value: Optional[float] = value
    else:
        process.stdin.write(f"TELL {candidate_id}\nFAIL {candidate_id}\n")
        row_value = None
    process.stdin.flush()
    records[candidate_id] = {
        "candidate_id": candidate_id,
        "unit_x": point.tolist(),
        "region_id": region,
        "status": result["status"],
        "value": row_value,
        "worker_rank": worker_rank,
        "worst_dmerc": result.get("worst_dmerc"),
        "failure": result.get("failure"),
    }
    completion_order.append(candidate_id)
    return best


def _run_manager(args: argparse.Namespace, comm: Any) -> dict[str, Any]:
    args.scratch.mkdir(parents=True, exist_ok=True)
    initialization_errors = comm.allgather(None)
    if any(error is not None for error in initialization_errors):
        details = "; ".join(error for error in initialization_errors if error)
        raise LandremanFortBOError(f"archived evaluator initialization failed: {details}")
    stderr_path = args.scratch / "fortbo.stderr"
    command = _protocol_command(
        args.fortbo_command,
        args.dimension,
        args.budget,
        args.initial,
        args.seed,
        args.workers,
    )
    started = time.perf_counter()
    stderr_handle = stderr_path.open("w", encoding="utf-8")
    try:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr_handle,
            text=True,
            bufsize=1,
            env={**os.environ, "OMP_NUM_THREADS": "1"},
        )
    except OSError:
        stderr_handle.close()
        _stop_workers(comm, args.workers)
        raise
    assert process.stdin is not None and process.stdout is not None
    pending: dict[int, tuple[np.ndarray, int]] = {}
    idle = list(range(1, args.workers + 1))
    records: dict[int, dict[str, Any]] = {}
    completion_order: list[int] = []
    dispatched = 0
    best = float("inf")
    try:
        while True:
            if dispatched < args.budget and len(pending) < args.workers:
                line = _next_protocol_line(process.stdout)
                if line == "WAIT":
                    if not pending:
                        raise LandremanFortBOError("FortBO waited without pending work")
                    best = _complete_one(
                        comm, process, pending, idle, records, completion_order, best
                    )
                    continue
                if line.startswith("DONE "):
                    raise LandremanFortBOError("FortBO completed before its budget")
                if line != "ASK 1":
                    raise LandremanFortBOError(f"malformed FortBO ask: {line!r}")
                candidate_id, region, point = _parse_point(
                    process.stdout.readline(), args.dimension
                )
                if candidate_id != dispatched or not idle:
                    raise LandremanFortBOError("FortBO candidate or worker ordering is invalid")
                worker_rank = idle.pop(0)
                pending[candidate_id] = (point, region)
                comm.send((candidate_id, point), dest=worker_rank, tag=WORK_TAG)
                dispatched += 1
                continue
            if pending:
                best = _complete_one(
                    comm, process, pending, idle, records, completion_order, best
                )
                continue
            line = _next_protocol_line(process.stdout)
            if not line.startswith("DONE "):
                raise LandremanFortBOError(f"malformed FortBO completion: {line!r}")
            break
    except Exception:
        process.kill()
        process.wait()
        raise
    finally:
        if process.stdin is not None:
            process.stdin.close()
        return_code = process.wait()
        stderr_handle.close()
        _stop_workers(comm, args.workers)
    if return_code:
        detail = stderr_path.read_text(encoding="utf-8")[-4000:]
        raise LandremanFortBOError(
            f"FortBO protocol failed with exit code {return_code}: {detail}"
        )
    if dispatched != args.budget or len(records) != args.budget:
        raise LandremanFortBOError(
            f"FortBO completed {len(records)} calls, expected {args.budget}"
        )
    return {
        "evaluations": [records[index] for index in range(args.budget)],
        "completion_order": completion_order,
        "best_value": None if not np.isfinite(best) else best,
        "wall_seconds": time.perf_counter() - started,
        "fortbo_command": command,
        "stderr": str(stderr_path),
    }


def _document(args: argparse.Namespace, result: Mapping[str, Any]) -> dict[str, Any]:
    alpha_revision = _git_revision(args.alpha_root)
    archive_digest = None
    if args.archive:
        digest = hashlib.sha256()
        with args.archive.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        archive_digest = digest.hexdigest()
    return {
        "schema_name": "fortbo.landreman-value-only",
        "schema_version": 1,
        "source": {
            "fortbo_commit": _clean_revision(ROOT),
            "alpha_opt_commit": alpha_revision,
            "archive_sha256": archive_digest,
            "prepared_alpha_root": str(args.alpha_root.resolve()),
        },
        "problem": {
            "case_id": "landreman-alpha-loss",
            "dimension": args.dimension,
            "bounds": [[0.0] * args.dimension, [1.0] * args.dimension],
            "failure_value": FAIL_VALUE,
            "failure_policy": "FAIL retained by FortBO and excluded from surrogate",
        },
        "configuration": {
            "method": "completion-driven FortBO TuRBO-1 with Thompson sampling",
            "seed": args.seed,
            "budget": args.budget,
            "initial_points": args.initial,
            "workers": args.workers,
            "mpi_ranks": args.workers + 1,
        },
        "evaluations": result["evaluations"],
        "result": {
            "truth_calls": len(result["evaluations"]),
            "failed_evaluations": sum(
                row["status"] != "ok" for row in result["evaluations"]
            ),
            "best_value": result["best_value"],
            "wall_seconds": result["wall_seconds"],
            "completion_order": result["completion_order"],
        },
        "passed": True,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fortbo-command", default="fo")
    parser.add_argument("--dimension", type=int, default=DEFAULT_DIMENSION)
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    parser.add_argument("--initial", type=int, default=DEFAULT_INITIAL)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parser().parse_args(argv)
    args.fortbo_command = resolve_fo_command(args.fortbo_command)
    if args.dimension != DEFAULT_DIMENSION:
        raise SystemExit("Landreman parity requires dimension 20")
    if args.initial < 1 or args.budget < args.initial or args.workers < 1:
        raise SystemExit("invalid Landreman budget/worker configuration")
    if args.output.resolve().is_relative_to(ROOT.resolve()):
        raise SystemExit("Landreman output must be outside the FortBO source tree")
    if args.scratch.resolve().is_relative_to(ROOT.resolve()):
        raise SystemExit("Landreman scratch must be outside the FortBO source tree")
    if args.output.exists():
        raise SystemExit(f"Landreman output already exists: {args.output}")
    if not args.alpha_root.is_dir():
        raise SystemExit(f"prepared alpha root is missing: {args.alpha_root}")
    if args.archive is not None and not args.archive.is_file():
        raise SystemExit(f"Landreman archive is missing: {args.archive}")
    try:
        from mpi4py import MPI
    except ImportError as error:
        raise SystemExit(f"mpi4py is required inside the Slurm environment: {error}")

    comm = MPI.COMM_WORLD
    if comm.Get_size() != args.workers + 1:
        if comm.Get_rank() == 0:
            print(
                f"expected {args.workers + 1} MPI ranks, got {comm.Get_size()}",
                file=sys.stderr,
            )
        return 2
    try:
        if comm.Get_rank() == 0:
            comm.Barrier()
            result = _run_manager(args, comm)
            document = _document(args, result)
            if args.output.exists():
                raise LandremanFortBOError(f"output already exists: {args.output}")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
            print(f"wrote {args.output}")
        else:
            _worker_loop(args.alpha_root, args.scratch, comm, comm.Get_rank())
    except (LandremanFortBOError, OSError, ValueError) as error:
        if comm.Get_rank() == 0:
            print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
