#!/usr/bin/env python3
"""Replay a FortBO smoke trace against the independent Landreman trust state."""

from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

try:
    from .landreman_reference import TurboState
except ImportError:  # direct execution from the scripts directory
    from landreman_reference import TurboState


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Row:
    evaluation: int
    point: np.ndarray
    value: float
    best: float
    length: float
    success_counter: int
    failure_counter: int
    restarts: int


def parse_rows(output: str, dimension: int) -> list[Row]:
    rows = []
    expected = 3 + dimension + 3 + 3
    for line in output.splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] != "ROW":
            continue
        if len(fields) != expected:
            raise ValueError(f"malformed trace row with {len(fields)} fields")
        point_end = 3 + dimension
        rows.append(Row(
            evaluation=int(fields[1]),
            point=np.array([float(item) for item in fields[3:point_end]],
                           dtype=np.float64),
            value=float(fields[point_end]),
            best=float(fields[point_end + 1]),
            length=float(fields[point_end + 2]),
            success_counter=int(fields[point_end + 3]),
            failure_counter=int(fields[point_end + 4]),
            restarts=int(fields[point_end + 5]),
        ))
    if [row.evaluation for row in rows] != list(range(len(rows))):
        raise ValueError("trace evaluations are not contiguous")
    if not rows:
        raise ValueError("trace contains no ROW records")
    return rows


def check_frozen_proposals(
        rows: list[Row], initial: int,
        frozen_initial: Optional[np.ndarray] = None,
        frozen_candidates: Optional[np.ndarray] = None,
        atol: float = 1.0e-12) -> None:
    """Check proposal coordinates against caller-owned replay fixtures.

    The fixtures are independent of FortBO and are the escape hatch for
    cross-language Sobol/Owen streams that cannot be reconstructed bit-for-bit
    locally.  Initial rows are ordered; later rows only need to be members of
    the frozen candidate pool because the posterior decides which candidate
    is selected.
    """
    if frozen_initial is not None:
        if len(frozen_initial) < initial:
            raise ValueError("frozen initial design is shorter than the trace")
        observed = np.array([row.point for row in rows[:initial]])
        np.testing.assert_allclose(observed, frozen_initial[:initial],
                                   rtol=0.0, atol=atol)

    if frozen_candidates is not None:
        if len(rows) == initial:
            return
        for row in rows[initial:]:
            if not np.any(np.all(np.isclose(frozen_candidates, row.point,
                                             rtol=0.0, atol=atol), axis=1)):
                raise AssertionError(
                    f"proposal at evaluation {row.evaluation} is not in "
                    "the frozen candidate pool"
                )


def check_rows(rows: list[Row], dimension: int, initial: int, atol: float = 1.0e-12) -> None:
    if initial < 1 or initial > len(rows):
        raise ValueError("initial count must be within the trace")
    initial_best = min(row.value for row in rows[:initial])
    state = TurboState(dimension, 1)
    state.best_value = initial_best
    expected_best = float("inf")
    for row in rows[:initial]:
        if abs(row.best - min(expected_best, row.value)) > atol:
            raise AssertionError("initial best trace disagrees with the values")
        expected_best = min(expected_best, row.value)
        if row.success_counter != 0 or row.failure_counter != 0:
            raise AssertionError("initial design changed trust counters")
    for row in rows[initial:]:
        state.update(np.array([row.value], dtype=np.float64))
        expected_best = min(expected_best, row.value)
        if abs(row.best - expected_best) > atol:
            raise AssertionError(f"best trace mismatch at evaluation {row.evaluation}")
        if abs(row.length - state.length) > atol:
            raise AssertionError(f"trust radius mismatch at evaluation {row.evaluation}")
        if row.success_counter != state.success_counter:
            raise AssertionError(f"success counter mismatch at evaluation {row.evaluation}")
        if row.failure_counter != state.failure_counter:
            raise AssertionError(f"failure counter mismatch at evaluation {row.evaluation}")
        if row.restarts != 0:
            raise AssertionError("the bounded smoke trace unexpectedly restarted")


def check(output: str, dimension: int, initial: int, atol: float = 1.0e-12,
          frozen_initial: Optional[np.ndarray] = None,
          frozen_candidates: Optional[np.ndarray] = None) -> None:
    rows = parse_rows(output, dimension)
    check_rows(rows, dimension, initial, atol=atol)
    check_frozen_proposals(rows, initial, frozen_initial=frozen_initial,
                           frozen_candidates=frozen_candidates, atol=atol)


def read_point_pool(path: Path, dimension: int) -> np.ndarray:
    """Read the small count/width/text format used by the replay adapter."""
    lines = [line.split() for line in path.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    if not lines or len(lines[0]) != 2:
        raise ValueError(f"invalid point-pool header in {path}")
    count, width = (int(item) for item in lines[0])
    if count < 1 or width != dimension or len(lines[1:]) != count:
        raise ValueError(f"invalid point-pool shape in {path}")
    values = np.array([[float(item) for item in row] for row in lines[1:]],
                      dtype=np.float64)
    if values.shape != (count, dimension):
        raise ValueError(f"invalid point-pool rows in {path}")
    return values


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command", default="fo")
    parser.add_argument("--dimension", type=int, default=2)
    parser.add_argument("--budget", type=int, default=32)
    parser.add_argument("--initial", type=int, default=3)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--acquisition", choices=("ei", "ts"), default="ei")
    parser.add_argument("--frozen-initial", type=Path)
    parser.add_argument("--frozen-candidates", type=Path)
    parser.add_argument("--atol", type=float, default=1.0e-12)
    args = parser.parse_args(argv)
    extra = [args.acquisition]
    if args.frozen_candidates is not None:
        extra.append(str(args.frozen_candidates))
    if args.frozen_initial is not None:
        if args.frozen_candidates is None:
            extra.append("-")
        extra.extend([str(args.frozen_initial)])
    command = [args.command, "exec", "fortbo_reproduction", str(args.dimension),
               str(args.budget), str(args.initial), str(args.seed), *extra]
    if args.command != "fo":
        command = [args.command, str(args.dimension), str(args.budget),
                   str(args.initial), str(args.seed), *extra]
    result = subprocess.run(
        command,
        cwd=ROOT, check=False, capture_output=True, text=True,
    )
    if result.returncode:
        raise SystemExit(result.stdout + result.stderr)
    initial = (read_point_pool(args.frozen_initial, args.dimension)
               if args.frozen_initial is not None else None)
    candidates = (read_point_pool(args.frozen_candidates, args.dimension)
                  if args.frozen_candidates is not None else None)
    check(result.stdout, args.dimension, args.initial, atol=args.atol,
          frozen_initial=initial, frozen_candidates=candidates)
    print("TuRBO trust trace: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
