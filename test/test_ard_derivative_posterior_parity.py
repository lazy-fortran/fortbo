import unittest

import numpy as np

from scripts.check_ard_derivative_posterior_parity import (
    QUERY_X,
    check,
    observation_covariance,
    oracle,
)


class ArdDerivativePosteriorParityTests(unittest.TestCase):
    def test_observation_blocks_are_symmetric(self):
        left = np.array([0.17, 0.41])
        right = np.array([0.73, 0.82])
        for component_left in range(3):
            for component_right in range(3):
                self.assertAlmostEqual(
                    observation_covariance(left, component_left, right, component_right),
                    observation_covariance(right, component_right, left, component_left),
                    places=13,
                )

    def test_parser_and_oracle_reject_a_changed_derivative_record(self):
        expected = oracle()
        lines = []
        for i in range(len(QUERY_X)):
            lines.append(f"MOMENT {i + 1} {expected[0][i]:.16e} {expected[1][i]:.16e}")
            for j in range(2):
                lines.append(
                    f"GRADIENT {i + 1} {j + 1} "
                    f"{expected[2][i, j]:.16e} {expected[3][i, j]:.16e}"
                )
        output = "\n".join(lines)
        check(output)
        changed = output.replace(
            f"{expected[2][1, 0]:.16e}",
            f"{expected[2][1, 0] + 1.0e-3:.16e}",
        )
        with self.assertRaises(AssertionError):
            check(changed)


if __name__ == "__main__":
    unittest.main()
