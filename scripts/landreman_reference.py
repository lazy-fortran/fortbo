"""Independent NumPy definitions for the Landreman value-only parity gates.

This module does not import FortBO, FortML, BoTorch, or any generated kernel.
It is deliberately written from the ARD Matérn-5/2 and TuRBO definitions so a
trace comparison can fail in either direction.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Tuple

import numpy as np


def matern52(x: np.ndarray, y: np.ndarray, lengthscales: np.ndarray,
             variance: float = 1.0) -> np.ndarray:
    """Return the ARD Matérn-5/2 covariance matrix."""
    left = np.asarray(x, dtype=np.float64)
    right = np.asarray(y, dtype=np.float64)
    scales = np.asarray(lengthscales, dtype=np.float64)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("x and y must be two-dimensional with equal width")
    if scales.shape != (left.shape[1],) or np.any(scales <= 0.0):
        raise ValueError("lengthscales must be positive and match the dimension")
    distance = np.sqrt(np.sum(((left[:, None, :] - right[None, :, :]) / scales)**2,
                              axis=2))
    root5 = math.sqrt(5.0)*distance
    return variance*(1.0 + root5 + 5.0*distance**2/3.0)*np.exp(-root5)


def gp_posterior(train_x: np.ndarray, train_y: np.ndarray,
                 query_x: np.ndarray, lengthscales: np.ndarray,
                 *, variance: float = 1.0, noise_variance: float = 1.0e-6
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """Return exact latent posterior mean and marginal variance."""
    x = np.asarray(train_x, dtype=np.float64)
    y = np.asarray(train_y, dtype=np.float64).reshape(-1)
    query = np.asarray(query_x, dtype=np.float64)
    if x.ndim != 2 or query.ndim != 2 or y.shape != (len(x),):
        raise ValueError("training and query shapes are inconsistent")
    gram = matern52(x, x, lengthscales, variance)
    gram.flat[::len(gram) + 1] += noise_variance
    cross = matern52(x, query, lengthscales, variance)
    alpha = np.linalg.solve(gram, y)
    mean = cross.T @ alpha
    solved = np.linalg.solve(gram, cross)
    prior = np.full(len(query), variance, dtype=np.float64)
    marginal_variance = np.maximum(prior - np.sum(cross*solved, axis=0), 0.0)
    return mean, marginal_variance


def expected_improvement(mean: np.ndarray, standard_deviation: np.ndarray,
                         best: float, xi: float = 0.0) -> np.ndarray:
    """Analytic EI for minimization, using only the scalar normal definition."""
    mean = np.asarray(mean, dtype=np.float64)
    sd = np.maximum(np.asarray(standard_deviation, dtype=np.float64), 0.0)
    gap = best - xi - mean
    result = np.maximum(gap, 0.0)
    positive = sd > 0.0
    if np.any(positive):
        z = gap[positive]/sd[positive]
        cdf = 0.5*(1.0 + np.vectorize(math.erf)(z/math.sqrt(2.0)))
        pdf = np.exp(-0.5*z*z)/math.sqrt(2.0*math.pi)
        result[positive] = gap[positive]*cdf + sd[positive]*pdf
    return result


def trust_bounds(center: np.ndarray, lengthscales: np.ndarray,
                 length: float) -> Tuple[np.ndarray, np.ndarray]:
    """Shape and clip a unit-cube TuRBO region by ARD lengthscales."""
    center = np.asarray(center, dtype=np.float64)
    scales = np.asarray(lengthscales, dtype=np.float64)
    if center.ndim != 1 or scales.shape != center.shape:
        raise ValueError("center and lengthscale shapes are inconsistent")
    weights = scales/scales.mean()
    weights /= np.prod(weights**(1.0/len(weights)))
    lower = np.maximum(center - weights*length/2.0, 0.0)
    upper = np.minimum(center + weights*length/2.0, 1.0)
    return lower, upper


@dataclass
class TurboState:
    """Landreman's scalar trust-state replay."""

    dimension: int
    batch_size: int = 1
    length: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    success_tolerance: int = 10
    best_value: float = math.inf
    success_counter: int = 0
    failure_counter: int = 0
    restart_triggered: bool = False

    def __post_init__(self) -> None:
        if self.dimension < 1 or self.batch_size < 1:
            raise ValueError("dimension and batch size must be positive")
        self.failure_tolerance = math.ceil(
            max(4.0/self.batch_size, self.dimension/self.batch_size)
        )

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if len(values) < 1 or not np.all(np.isfinite(values)):
            raise ValueError("a trust-state batch needs finite values")
        current = float(np.min(values))
        improved = current < self.best_value - 1.0e-3*abs(self.best_value)
        if improved:
            self.success_counter += 1
            self.failure_counter = 0
        else:
            self.success_counter = 0
            self.failure_counter += 1
        if self.success_counter == self.success_tolerance:
            self.length = min(2.0*self.length, self.length_max)
            self.success_counter = 0
        elif self.failure_counter == self.failure_tolerance:
            self.length /= 2.0
            self.failure_counter = 0
        self.best_value = min(self.best_value, current)
        if self.length < self.length_min:
            self.restart_triggered = True


def candidate_count(dimension: int) -> int:
    if dimension < 1:
        raise ValueError("dimension must be positive")
    return min(5000, max(2000, 200*dimension))
