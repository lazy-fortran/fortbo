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


def check(output: str, dimension: int, initial: int, atol: float = 1.0e-12) -> None:
    check_rows(parse_rows(output, dimension), dimension, initial, atol=atol)


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command", default="fo")
    parser.add_argument("--dimension", type=int, default=2)
    parser.add_argument("--budget", type=int, default=32)
    parser.add_argument("--initial", type=int, default=3)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--acquisition", choices=("ei", "ts"), default="ei")
    parser.add_argument("--atol", type=float, default=1.0e-12)
    args = parser.parse_args(argv)
    result = subprocess.run(
        [args.command, "exec", "fortbo_reproduction", str(args.dimension),
         str(args.budget), str(args.initial), str(args.seed), args.acquisition],
        cwd=ROOT, check=False, capture_output=True, text=True,
    )
    if result.returncode:
        raise SystemExit(result.stdout + result.stderr)
    check(result.stdout, args.dimension, args.initial, atol=args.atol)
    print("TuRBO trust trace: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
