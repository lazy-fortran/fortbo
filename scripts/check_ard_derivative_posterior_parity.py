#!/usr/bin/env python3
"""Compare the value-plus-gradient ARD fixture with an independent oracle."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TRAIN_X = np.array([[0.10, 0.20], [0.40, 0.80], [0.70, 0.30], [0.90, 0.95]])
TRAIN_Y = np.array([0.7, -0.2, 0.4, 1.1])
TRAIN_GRADIENT = np.array([
    [0.3, -0.1], [0.2, 0.4], [-0.2, 0.5], [0.1, -0.3],
])
QUERY_X = np.array([[0.15, 0.25], [0.55, 0.65], [0.85, 0.60]])
LENGTHSCALES = np.array([0.30, 0.65])
SIGNAL_VARIANCE = 1.2
NOISE_VARIANCE = 0.04


def value_gradient_mixed(x1: np.ndarray, x2: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Return k, d k/d x1, and d2 k/(d x1 d x2) independently."""
    delta = x1 - x2
    scaled = np.sqrt(5.0*np.sum((delta/LENGTHSCALES)**2))
    exponential = np.exp(-scaled)
    value = SIGNAL_VARIANCE*(1.0 + scaled + scaled*scaled/3.0)*exponential
    coefficient = -5.0*SIGNAL_VARIANCE*(1.0 + scaled)*exponential/3.0
    gradient = coefficient*delta/LENGTHSCALES**2
    mixed = (-25.0*SIGNAL_VARIANCE*exponential/3.0
             * np.outer(delta/LENGTHSCALES**2, delta/LENGTHSCALES**2))
    mixed += np.diag(-coefficient/LENGTHSCALES**2)
    return value, gradient, mixed


def observation_covariance(x1: np.ndarray, component1: int,
                           x2: np.ndarray, component2: int) -> float:
    value, gradient, mixed = value_gradient_mixed(x1, x2)
    if component1 == 0 and component2 == 0:
        return value
    if component1 > 0 and component2 == 0:
        return gradient[component1 - 1]
    if component1 == 0 and component2 > 0:
        return -gradient[component2 - 1]
    return mixed[component1 - 1, component2 - 1]


def oracle() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    observation_x = np.repeat(TRAIN_X, 3, axis=0)
    components = np.tile([0, 1, 2], len(TRAIN_X))
    observation_y = np.column_stack((TRAIN_Y, TRAIN_GRADIENT)).reshape(-1)
    gram = np.array([
        [observation_covariance(x1, c1, x2, c2)
         for x2, c2 in zip(observation_x, components)]
        for x1, c1 in zip(observation_x, components)
    ])
    gram += NOISE_VARIANCE*np.eye(len(gram))
    alpha = np.linalg.solve(gram, observation_y)
    cross = np.array([
        [observation_covariance(query, 0, train, component)
         for train, component in zip(observation_x, components)]
        for query in QUERY_X
    ]).T
    mean = cross.T @ alpha
    solved = np.linalg.solve(gram, cross)
    variance = np.maximum(
        SIGNAL_VARIANCE - np.sum(cross*solved, axis=0), 0.0,
    )
    mean_gradient = np.empty((len(QUERY_X), 2))
    sd_gradient = np.empty_like(mean_gradient)
    for query_index, query in enumerate(QUERY_X):
        cross_gradient = np.array([
            (value_gradient_mixed(query, train)[1]
             if component == 0 else value_gradient_mixed(query, train)[2][:, component - 1])
            for train, component in zip(observation_x, components)
        ])
        mean_gradient[query_index] = cross_gradient.T @ alpha
        variance_gradient = -2.0*cross_gradient.T @ solved[:, query_index]
        sd_gradient[query_index] = variance_gradient/(2.0*np.sqrt(variance[query_index]))
    return mean, variance, mean_gradient, sd_gradient


def parse(output: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    moments = []
    gradients = []
    for line in output.splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "MOMENT":
            if len(fields) != 4:
                raise ValueError("malformed derivative ARD moment record")
            moments.append((int(fields[1]), float(fields[2]), float(fields[3])))
        elif fields[0] == "GRADIENT":
            if len(fields) != 5:
                raise ValueError("malformed derivative ARD gradient record")
            gradients.append((int(fields[1]), int(fields[2]), float(fields[3]), float(fields[4])))
    moments.sort()
    gradients.sort()
    if [(row[0], row[1]) for row in gradients] != [
        (i, j) for i in range(1, len(QUERY_X) + 1) for j in range(1, 3)
    ]:
        raise ValueError("derivative ARD output has missing or duplicate gradients")
    if [row[0] for row in moments] != list(range(1, len(QUERY_X) + 1)):
        raise ValueError("derivative ARD output has missing or duplicate moments")
    return (
        np.array([row[1] for row in moments]),
        np.array([row[2] for row in moments]),
        np.array([[row[2] for row in gradients[2*i:2*i + 2]]
                  for i in range(len(QUERY_X))]),
        np.array([[row[3] for row in gradients[2*i:2*i + 2]]
                  for i in range(len(QUERY_X))]),
    )


def check(output: str, atol: float = 1.0e-12) -> None:
    observed = parse(output)
    expected = oracle()
    for left, right in zip(observed, expected):
        np.testing.assert_allclose(left, right, atol=atol, rtol=0.0)


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command", default="fo")
    parser.add_argument("--atol", type=float, default=1.0e-12)
    args = parser.parse_args(argv)
    result = subprocess.run(
        [args.command, "exec", "fortbo_ard_derivative_posterior_reproduction"],
        cwd=ROOT, check=False, capture_output=True, text=True,
    )
    if result.returncode:
        raise SystemExit(result.stdout + result.stderr)
    check(result.stdout, atol=args.atol)
    print("ARD derivative posterior parity: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
