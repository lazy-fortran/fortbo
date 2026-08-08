#!/usr/bin/env python3
"""Independent behavioral tests for the Landreman parity reference."""

import unittest

import numpy as np

from scripts.landreman_reference import (
    TurboState,
    candidate_count,
    expected_improvement,
    gp_posterior,
    matern52,
    trust_bounds,
)


class LandremanReferenceTests(unittest.TestCase):
    def test_ard_matern_is_symmetric_and_has_the_declared_diagonal(self):
        points = np.array([[0.0, 0.0], [0.2, 0.8], [0.9, 0.1]])
        covariance = matern52(points, points, np.array([0.2, 0.7]), variance=1.3)
        np.testing.assert_allclose(covariance, covariance.T, rtol=0.0, atol=1.0e-15)
        np.testing.assert_allclose(np.diag(covariance), 1.3, rtol=0.0, atol=1.0e-15)

    def test_posterior_uses_the_independent_dense_gp_definition(self):
        x = np.array([[0.0, 0.0], [0.4, 0.2], [0.8, 0.9]])
        y = np.array([0.5, -0.2, 0.7])
        query = np.array([[0.1, 0.3], [0.7, 0.4]])
        mean, variance = gp_posterior(x, y, query, np.array([0.3, 0.6]),
                                       variance=1.2, noise_variance=0.04)
        gram = matern52(x, x, np.array([0.3, 0.6]), variance=1.2)
        gram += 0.04*np.eye(len(x))
        cross = matern52(x, query, np.array([0.3, 0.6]), variance=1.2)
        alpha = np.linalg.solve(gram, y)
        expected_mean = cross.T @ alpha
        expected_variance = 1.2 - np.sum(cross*np.linalg.solve(gram, cross), axis=0)
        np.testing.assert_allclose(mean, expected_mean, rtol=0.0, atol=1.0e-13)
        np.testing.assert_allclose(variance, expected_variance, rtol=0.0, atol=1.0e-13)

    def test_ei_has_the_zero_variance_limit_and_normal_value(self):
        deterministic = expected_improvement(np.array([-1.0, 1.0]),
                                             np.array([0.0, 0.0]), 0.0)
        np.testing.assert_allclose(deterministic, [1.0, 0.0])
        stochastic = expected_improvement(np.array([0.0]), np.array([1.0]), 0.0)
        np.testing.assert_allclose(stochastic, [1.0/np.sqrt(2.0*np.pi)], atol=1.0e-15)

    def test_trust_state_matches_landreman_counter_events(self):
        state = TurboState(20, 1)
        state.best_value = 11.0
        for value in range(10, 0, -1):
            state.update(np.array([float(value)]))
        self.assertEqual(state.length, 1.6)
        self.assertEqual(state.success_counter, 0)
        for _ in range(20):
            state.update(np.array([1.0]))
        self.assertEqual(state.length, 0.8)
        self.assertEqual(state.failure_counter, 0)

    def test_candidate_and_bounds_contract(self):
        self.assertEqual(candidate_count(1), 2000)
        self.assertEqual(candidate_count(20), 4000)
        self.assertEqual(candidate_count(200), 5000)
        lower, upper = trust_bounds(np.array([0.5, 0.5]), np.array([1.0, 2.0]), 0.2)
        self.assertAlmostEqual(np.prod(upper-lower), 0.2**2)


if __name__ == "__main__":
    unittest.main()
