import unittest

import numpy as np

from scripts.check_ard_posterior_parity import check, expected, parse_moments
from scripts.landreman_reference import gp_posterior


class ArdPosteriorParityTests(unittest.TestCase):
    def test_independent_dense_oracle_has_anisotropic_moments(self):
        mean, variance = expected()
        self.assertEqual(mean.shape, (3,))
        self.assertEqual(variance.shape, (3,))
        self.assertTrue(np.all(variance > 0.0))

        swapped_mean, _ = gp_posterior(
            np.array([[0.10, 0.20], [0.40, 0.80], [0.70, 0.30], [0.90, 0.95]]),
            np.array([0.7, -0.2, 0.4, 1.1]),
            np.array([[0.15, 0.25], [0.55, 0.65], [0.85, 0.60]]),
            np.array([0.65, 0.30]), variance=1.2, noise_variance=0.04,
        )
        self.assertGreater(np.max(np.abs(mean - swapped_mean)), 1.0e-4)

    def test_parser_and_oracle_reject_a_changed_moment(self):
        mean, variance = expected()
        output = "\n".join(
            f"MOMENT {i + 1} {mean[i]:.16e} {variance[i]:.16e}"
            for i in range(3)
        )
        parsed_mean, parsed_variance = parse_moments(output)
        np.testing.assert_allclose(parsed_mean, mean, atol=1.0e-15, rtol=0.0)
        np.testing.assert_allclose(parsed_variance, variance, atol=1.0e-15, rtol=0.0)
        changed = output.replace(f"{mean[1]:.16e}", f"{mean[1] + 1.0e-3:.16e}")
        with self.assertRaises(AssertionError):
            check(changed)


if __name__ == "__main__":
    unittest.main()
