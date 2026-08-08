import unittest

import numpy as np

from scripts.glas_covariance import periodic_fourier_covariance


class GlasCovarianceTests(unittest.TestCase):
    def test_published_fourier_covariance_matches_integrated_kernel(self):
        modes = 3
        amplitude = 0.02
        lengthscale = 0.5
        covariance = periodic_fourier_covariance(modes, amplitude, lengthscale)
        grid = np.arange(1024, dtype=np.float64) * (2.0 * np.pi / 1024.0)
        kernel = amplitude * amplitude / 3.0 * np.exp(
            -2.0 * np.sin((grid[:, None] - grid[None, :]) / 2.0) ** 2
            / lengthscale**2)
        basis = np.column_stack([
            np.ones_like(grid),
            *(np.cos(order * grid) for order in range(1, modes + 1)),
            *(np.sin(order * grid) for order in range(1, modes + 1)),
        ])
        weights = (2.0 * np.pi / 1024.0) / np.pi
        integrated = weights**2 * basis.T @ kernel @ basis
        np.testing.assert_allclose(covariance[:7, :7], integrated, rtol=2.0e-12,
                                   atol=2.0e-15)

    def test_five_coil_dimension_and_pointwise_rms(self):
        covariance = periodic_fourier_covariance(6, 0.005, 0.5, coils=5)
        self.assertEqual(covariance.shape, (195, 195))
        coordinate = covariance[:13, :13]
        variances = []
        for theta in [0.0, 0.37, 1.2, 2.7]:
            basis = np.asarray(
                [0.5, *(np.cos(order * theta) for order in range(1, 7)),
                 *(np.sin(order * theta) for order in range(1, 7))])
            variances.append(basis @ coordinate @ basis)
        np.testing.assert_allclose(variances, variances[0], rtol=1.0e-14,
                                   atol=1.0e-18)
        self.assertLess(3.0 * variances[0], 0.005**2)
        self.assertGreater(3.0 * variances[0], 0.99 * 0.005**2)


if __name__ == "__main__":
    unittest.main()
