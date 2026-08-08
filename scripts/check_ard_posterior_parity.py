#!/usr/bin/env python3
"""Compare the FortBO ARD posterior fixture with an independent NumPy oracle."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

try:
    from .landreman_reference import gp_posterior, matern52
except ImportError:  # direct execution from the scripts directory
    from landreman_reference import gp_posterior, matern52


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


def expected_gradients() -> tuple[np.ndarray, np.ndarray]:
    gram = matern52(
        TRAIN_X, TRAIN_X, LENGTHSCALES, variance=1.2
    )
    gram += 0.04*np.eye(len(TRAIN_X))
    cross = matern52(
        TRAIN_X, QUERY_X, LENGTHSCALES, variance=1.2
    )
    alpha = np.linalg.solve(gram, TRAIN_Y)
    solved = np.linalg.solve(gram, cross)
    mean_gradient = np.empty((len(QUERY_X), len(LENGTHSCALES)))
    sd_gradient = np.empty_like(mean_gradient)
    for i, query in enumerate(QUERY_X):
        difference = query[None, :] - TRAIN_X
        scaled = np.sqrt(5.0*np.sum((difference/LENGTHSCALES)**2, axis=1))
        kernel_gradient = (
            (-5.0*1.2*(1.0 + scaled)*np.exp(-scaled)/3.0)[:, None]
            * difference/LENGTHSCALES**2
        )
        mean_gradient[i] = kernel_gradient.T @ alpha
        variance_gradient = -2.0*(kernel_gradient.T @ solved[:, i])
        _, variance = expected()
        sd_gradient[i] = variance_gradient/(2.0*np.sqrt(variance[i]))
    return mean_gradient, sd_gradient


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


def parse_gradients(output: str) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    for line in output.splitlines():
        fields = line.split()
        if fields and fields[0] == "GRADIENT":
            if len(fields) != 5:
                raise ValueError("malformed ARD gradient record")
            rows.append((int(fields[1]), int(fields[2]), float(fields[3]), float(fields[4])))
    rows.sort()
    expected_rows = [(i, j) for i in range(1, len(QUERY_X) + 1) for j in range(1, 3)]
    if [(row[0], row[1]) for row in rows] != expected_rows:
        raise ValueError("ARD posterior output has missing or duplicate gradients")
    return (np.array([[row[2] for row in rows[2*i:2*i + 2]] for i in range(len(QUERY_X))]),
            np.array([[row[3] for row in rows[2*i:2*i + 2]] for i in range(len(QUERY_X))]))


def check(output: str, atol: float = 1.0e-12) -> None:
    observed_mean, observed_variance = parse_moments(output)
    expected_mean, expected_variance = expected()
    np.testing.assert_allclose(observed_mean, expected_mean, atol=atol, rtol=0.0)
    np.testing.assert_allclose(observed_variance, expected_variance, atol=atol, rtol=0.0)
    observed_mean_gradient, observed_sd_gradient = parse_gradients(output)
    expected_mean_gradient, expected_sd_gradient = expected_gradients()
    np.testing.assert_allclose(observed_mean_gradient, expected_mean_gradient,
                               atol=atol, rtol=0.0)
    np.testing.assert_allclose(observed_sd_gradient, expected_sd_gradient,
                               atol=atol, rtol=0.0)


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
