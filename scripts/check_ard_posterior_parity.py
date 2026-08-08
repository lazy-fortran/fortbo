#!/usr/bin/env python3
"""Compare the FortBO ARD posterior fixture with an independent NumPy oracle."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

try:
    from .landreman_reference import gp_posterior
except ImportError:  # direct execution from the scripts directory
    from landreman_reference import gp_posterior


ROOT = Path(__file__).resolve().parents[1]
TRAIN_X = np.array([[0.10, 0.20], [0.40, 0.80], [0.70, 0.30], [0.90, 0.95]])
TRAIN_Y = np.array([0.7, -0.2, 0.4, 1.1])
QUERY_X = np.array([[0.15, 0.25], [0.55, 0.65], [0.85, 0.60]])
LENGTHSCALES = np.array([0.30, 0.65])


def expected() -> tuple[np.ndarray, np.ndarray]:
    return gp_posterior(
        TRAIN_X, TRAIN_Y, QUERY_X, LENGTHSCALES,
        variance=1.2, noise_variance=0.04,
    )


def parse_moments(output: str) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    for line in output.splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] != "MOMENT" or len(fields) != 4:
            continue
        rows.append((int(fields[1]), float(fields[2]), float(fields[3])))
    rows.sort()
    if [row[0] for row in rows] != list(range(1, len(rows) + 1)):
        raise ValueError("ARD posterior output has missing or duplicate moment indices")
    if len(rows) != len(QUERY_X):
        raise ValueError(f"expected {len(QUERY_X)} moments, got {len(rows)}")
    return (np.array([row[1] for row in rows]),
            np.array([row[2] for row in rows]))


def check(output: str, atol: float = 1.0e-12) -> None:
    observed_mean, observed_variance = parse_moments(output)
    expected_mean, expected_variance = expected()
    np.testing.assert_allclose(observed_mean, expected_mean, atol=atol, rtol=0.0)
    np.testing.assert_allclose(observed_variance, expected_variance, atol=atol, rtol=0.0)


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command", default="fo")
    parser.add_argument("--atol", type=float, default=1.0e-12)
    args = parser.parse_args(argv)
    result = subprocess.run(
        [args.command, "exec", "fortbo_ard_posterior_reproduction"],
        cwd=ROOT, check=False, capture_output=True, text=True,
    )
    if result.returncode:
        raise SystemExit(result.stdout + result.stderr)
    check(result.stdout, atol=args.atol)
    print("ARD posterior parity: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
