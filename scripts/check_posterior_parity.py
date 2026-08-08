#!/usr/bin/env python3
"""Compare a FortML posterior fixture with the independent NumPy oracle."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from landreman_reference import gp_posterior


ROOT = Path(__file__).resolve().parents[1]


def expected_moments() -> tuple[np.ndarray, np.ndarray]:
    train_x = np.array([[0.0, 0.0], [0.4, 0.2], [0.8, 0.9], [0.2, 0.7]])
    train_y = np.array([0.5, -0.2, 0.7, 0.1])
    query_x = np.array([[0.1, 0.3], [0.7, 0.4], [0.25, 0.75]])
    return gp_posterior(train_x, train_y, query_x, np.array([0.3, 0.3]),
                        variance=1.2, noise_variance=0.04)


def parse_moments(output: str) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    for line in output.splitlines():
        fields = line.split()
        if fields and fields[0] == "MOMENT":
            if len(fields) != 4:
                raise ValueError(f"malformed moment row: {line}")
            rows.append((int(fields[1]), float(fields[2]), float(fields[3])))
    rows.sort()
    if [row[0] for row in rows] != list(range(3)):
        raise ValueError(f"expected three moment rows, got {rows}")
    return (np.array([row[1] for row in rows]),
            np.array([row[2] for row in rows]))


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fortbo-command", default="fo",
                        help="fo or a compiled fortbo_posterior_reproduction executable")
    parser.add_argument("--atol", type=float, default=1.0e-12)
    args = parser.parse_args(argv)
    if args.fortbo_command == "fo":
        command = ["fo", "exec", "fortbo_posterior_reproduction"]
    else:
        command = [args.fortbo_command]
    result = subprocess.run(command, cwd=ROOT, check=False,
                            capture_output=True, text=True)
    if result.returncode:
        raise SystemExit(result.stdout + result.stderr)
    actual_mean, actual_variance = parse_moments(result.stdout)
    expected_mean, expected_variance = expected_moments()
    np.testing.assert_allclose(actual_mean, expected_mean, rtol=0.0, atol=args.atol)
    np.testing.assert_allclose(actual_variance, expected_variance, rtol=0.0, atol=args.atol)
    print(f"posterior parity PASS (atol={args.atol:g})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
