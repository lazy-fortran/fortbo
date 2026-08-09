#!/usr/bin/env python3
"""Run the pinned B5 control and FortBO concurrently.

The control is the existing ``simsopt-dfo`` BoTorch TuRBO implementation. The
FortBO process uses the same pinned ConStellaration evaluator, coordinate map,
budget, seed, and worker count. The resulting control ledger is stored as the
oracle side of a pair document; the script never treats FortBO output as its
own expected answer.

This launcher is intentionally external-run friendly. It does not submit jobs
or copy repositories. Invoke it from the requested CPU/GPU allocation with
``--run-root`` on that allocation's scratch filesystem.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

try:
    from .fortbo_environment import (
        FortBOEnvironmentError,
        configure_fo_environment,
        preflight_fo,
    )
except ImportError:  # pragma: no cover - used when this file is run directly
    from fortbo_environment import (
        FortBOEnvironmentError,
        configure_fo_environment,
        preflight_fo,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DFO_ROOT = Path(os.environ["SIMSOPT_DFO_SOURCE"]) if os.environ.get("SIMSOPT_DFO_SOURCE") else None
DEFAULT_WORKERS = 8
DEFAULT_BUDGET = 256
DEFAULT_RESERVE_GIB = 8.0


class PairError(RuntimeError):
    """The paired run cannot be started or recorded safely."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _disk(path: Path) -> dict[str, int]:
    usage = shutil.disk_usage(path)
    return {"free_bytes": usage.free, "total_bytes": usage.total, "used_bytes": usage.used}


def _ensure_external(path: Path, roots: Iterable[Path], label: str) -> None:
    resolved = path.resolve()
    for root in roots:
        try:
            resolved.relative_to(root.resolve())
        except ValueError:
            continue
        raise PairError(f"{label} must be outside source checkout {root}")


def _ensure_new(path: Path, label: str) -> None:
    if path.exists() and (path.is_file() or any(path.iterdir())):
        raise PairError(f"{label} already contains artifacts: {path}")


def _git_revision(path: Path, label: str) -> str:
    """Require a real Git checkout before a pinned evaluator can start."""

    path = path.resolve()
    if not path.is_dir():
        raise PairError(f"{label} checkout is missing: {path}")
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise PairError(f"cannot inspect {label} checkout {path}: {error}") from error
    if result.returncode or not result.stdout.strip():
        detail = (result.stderr or result.stdout).strip()[-500:]
        raise PairError(f"{label} checkout is not a Git worktree: {path}: {detail}")
    return result.stdout.strip()


def _preflight_python(
    interpreter: str, code: str, label: str, environment: Mapping[str, str]
) -> None:
    try:
        result = subprocess.run(
            [interpreter, "-c", code],
            env=dict(environment),
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise PairError(f"{label} preflight failed: {error}") from error
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()[-1000:]
        raise PairError(f"{label} preflight failed: {detail}")


def _run_process(
    label: str,
    command: list[str],
    cwd: Path,
    environment: Mapping[str, str],
    stdout_path: Path,
    stderr_path: Path,
    run_root: Path,
    reserve_bytes: int,
    abort: threading.Event,
) -> dict[str, Any]:
    started = time.perf_counter()
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=dict(environment),
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        while process.poll() is None:
            usage = shutil.disk_usage(run_root)
            if usage.free < reserve_bytes:
                abort.set()
                _terminate(process)
                raise PairError(
                    f"{label}: stopped paired run below disk reserve "
                    f"({usage.free / 1024**3:.2f} GiB free)"
                )
            if abort.is_set():
                _terminate(process)
                raise PairError(f"{label}: stopped because the paired process failed")
            time.sleep(5.0)
        if process.returncode:
            abort.set()
            raise PairError(
                f"{label}: process exited with return code {process.returncode}"
            )
        return {
            "command": command,
            "cwd": str(cwd),
            "returncode": process.returncode,
            "wall_seconds": time.perf_counter() - started,
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
        }


def _terminate(process: subprocess.Popen[Any]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def _load_result(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise PairError(f"expected run output is missing: {path}")
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PairError(f"cannot read run output {path}: {error}") from error
    if not isinstance(result, dict):
        raise PairError(f"run output is not a JSON object: {path}")
    return result


def _document_summary(document: Mapping[str, Any]) -> dict[str, Any]:
    problem = document.get("problem", {})
    result = document.get("result", {})
    return {
        "schema_name": document.get("schema_name"),
        "case_id": problem.get("case_id"),
        "mode": problem.get("mode"),
        "dimension": problem.get("dimension"),
        "constellaration_commit": document.get("source", {}).get("constellaration_commit"),
        "truth_calls": result.get("truth_calls"),
        "failed_evaluations": result.get("failed_evaluations"),
        "best_value": result.get("best_value"),
        "wall_seconds": result.get("wall_seconds"),
    }


def _ledgers_passed(
    original_document: Optional[Mapping[str, Any]],
    fortbo_document: Optional[Mapping[str, Any]],
) -> bool:
    return all(
        document is not None and document.get("passed") is True
        for document in (original_document, fortbo_document)
    )


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("raw", "data-informed"), required=True)
    parser.add_argument("--regions", type=int, choices=(1, 4), default=1)
    parser.add_argument("--seed", type=int, choices=(1, 2, 3, 4, 5), required=True)
    parser.add_argument("--dfo-root", type=Path, default=DEFAULT_DFO_ROOT,
                        help="cluster checkout of simsopt-dfo; also set SIMSOPT_DFO_SOURCE")
    parser.add_argument("--fortbo-root", type=Path, default=ROOT)
    parser.add_argument("--fortbo-python", default=os.environ.get("FORTBO_PYTHON", sys.executable),
                        help="Python interpreter for the FortBO bridge")
    parser.add_argument("--original-python", default=os.environ.get("SIMSOPT_DFO_PYTHON", sys.executable))
    parser.add_argument("--fortbo-command", default="fo")
    parser.add_argument("--constellaration-root", type=Path)
    parser.add_argument("--constellaration-python", type=Path)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--disk-reserve-gib", type=float, default=DEFAULT_RESERVE_GIB)
    args = parser.parse_args(argv)
    try:
        if args.dfo_root is None:
            raise PairError("set SIMSOPT_DFO_SOURCE or pass --dfo-root on the cluster")
        if args.budget != DEFAULT_BUDGET or args.workers != DEFAULT_WORKERS:
            raise PairError("exact B5 oracle pairing requires budget 256 and eight workers")
        if not args.dfo_root.is_dir() or not (args.dfo_root / "scripts/run_b5_async_turbo.py").is_file():
            raise PairError(f"simsopt-dfo control checkout is missing: {args.dfo_root}")
        if args.constellaration_root is None:
            raise PairError(
                "exact B5 oracle pairing requires --constellaration-root"
            )
        constellaration_root = args.constellaration_root.resolve()
        _git_revision(constellaration_root, "ConStellaration")
        _ensure_external(args.run_root, (args.dfo_root, args.fortbo_root), "run-root")
        _ensure_external(args.output, (args.dfo_root, args.fortbo_root), "output")
        _ensure_new(args.run_root, "run-root")
        _ensure_new(args.output, "pair output")
        args.run_root.mkdir(parents=True, exist_ok=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        disk_before = _disk(args.run_root)
        environment = configure_fo_environment(os.environ, args.run_root / "fo-cache")
        environment["SIMSOPT_DFO_SOURCE"] = str(args.dfo_root.resolve())
        dfo_source = str((args.dfo_root / "src").resolve())
        old_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = os.pathsep.join(
            item for item in (dfo_source, old_pythonpath) if item
        )
        environment["SIMSOPT_DFO_CONSTELLARATION_SRC"] = str(constellaration_root)
        if args.constellaration_python:
            environment["SIMSOPT_DFO_CONSTELLARATION_PYTHON"] = str(args.constellaration_python)
        _preflight_python(
            args.original_python,
            "import simsopt_dfo, torch, botorch",
            "original control",
            environment,
        )
        constellaration_python = environment.get("SIMSOPT_DFO_CONSTELLARATION_PYTHON")
        if constellaration_python:
            _preflight_python(
                constellaration_python,
                "import constellaration",
                "ConStellaration evaluator",
                environment,
            )
        fo_environment = preflight_fo(
            args.fortbo_command,
            args.fortbo_root,
            environment,
        )
        fortbo_fo_command = str(fo_environment["command"])
        reserve_bytes = int(args.disk_reserve_gib * 1024**3)
        original_output = args.run_root / "original.json"
        fortbo_output = args.run_root / "fortbo.json"
        original_scratch = args.run_root / "original-scratch"
        fortbo_scratch = args.run_root / "fortbo-scratch"
        original_command = [
            args.original_python,
            "scripts/run_b5_async_turbo.py",
            "--mode",
            args.mode,
            "--regions",
            str(args.regions),
            "--seed",
            str(args.seed),
            "--scratch",
            str(original_scratch),
            "--output",
            str(original_output),
        ]
        fortbo_command = [
            args.fortbo_python,
            str(args.fortbo_root / "scripts/run_fortbo_b5.py"),
            "--mode",
            args.mode,
            "--regions",
            str(args.regions),
            "--seed",
            str(args.seed),
            "--scratch",
            str(fortbo_scratch),
            "--output",
            str(fortbo_output),
            "--budget",
            str(args.budget),
            "--workers",
            str(args.workers),
            "--completion-driven",
            "--fortbo-command",
            fortbo_fo_command,
        ]
        abort = threading.Event()
        with ThreadPoolExecutor(max_workers=2) as executor:
            original_future = executor.submit(
                _run_process,
                "original control",
                original_command,
                args.dfo_root,
                environment,
                args.run_root / "original.stdout",
                args.run_root / "original.stderr",
                args.run_root,
                reserve_bytes,
                abort,
            )
            fortbo_future = executor.submit(
                _run_process,
                "FortBO",
                fortbo_command,
                args.fortbo_root,
                environment,
                args.run_root / "fortbo.stdout",
                args.run_root / "fortbo.stderr",
                args.run_root,
                reserve_bytes,
                abort,
            )
            process_records: dict[str, Any] = {}
            errors: list[str] = []
            for label, future in (("original", original_future), ("fortbo", fortbo_future)):
                try:
                    process_records[label] = future.result()
                except Exception as error:  # preserve both sides of an aborted pair
                    abort.set()
                    errors.append(f"{label}: {error}")
                    process_records[label] = {"error": str(error)}
        original_document = _load_result(original_output) if original_output.is_file() else None
        fortbo_document = _load_result(fortbo_output) if fortbo_output.is_file() else None
        if not _ledgers_passed(original_document, fortbo_document):
            errors.append("one or both child ledgers are missing passed=true")
        pair = {
            "schema_name": "fortbo.oracle-pair",
            "schema_version": 1,
            "case": "b5-constellaration-turbo",
            "configuration": {
                "mode": args.mode,
                "regions": args.regions,
                "seed": args.seed,
                "budget": args.budget,
                "workers": args.workers,
                "oracle_implementation": "simsopt-dfo BoTorch control",
                "fortbo_implementation": "FortBO completion-driven TuRBO",
            },
            "oracle": {
                "output": str(original_output),
                "sha256": _sha256(original_output) if original_output.is_file() else None,
                "summary": _document_summary(original_document) if original_document else None,
            },
            "fortbo": {
                "output": str(fortbo_output),
                "sha256": _sha256(fortbo_output) if fortbo_output.is_file() else None,
                "summary": _document_summary(fortbo_document) if fortbo_document else None,
            },
            "fortbo_environment": fo_environment,
            "processes": process_records,
            "disk": {
                "path": str(args.run_root),
                "reserve_gib": args.disk_reserve_gib,
                "before": disk_before,
                "after": _disk(args.run_root),
            },
            "status": "complete" if not errors and all(
                record.get("returncode") == 0 for record in process_records.values()
            ) and _ledgers_passed(original_document, fortbo_document) else "failed",
            "errors": errors,
        }
        args.output.write_text(json.dumps(pair, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
        return 0 if pair["status"] == "complete" else 1
    except (OSError, PairError, FortBOEnvironmentError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
