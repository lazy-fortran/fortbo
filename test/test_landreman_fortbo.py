"""Behavioral checks for the Landreman FortBO protocol boundary."""

import unittest

import numpy as np

from scripts.run_landreman_fortbo import (
    LandremanFortBOError,
    _parse_point,
    _protocol_command,
)


class LandremanFortBOTests(unittest.TestCase):
    def test_protocol_command_matches_completion_bridge_abi(self):
        self.assertEqual(
            _protocol_command("/opt/bin/fo", 20, 10000, 40, 0, 4),
            [
                "/opt/bin/fo",
                "exec",
                "fortbo_b5_completion_protocol",
                "20",
                "10000",
                "40",
                "0",
                "1",
                "4",
            ],
        )

    def test_parse_point_is_an_independent_unit_box_gate(self):
        candidate, region, point = _parse_point(
            "POINT 17 1 0.0 0.25 1.0", 3
        )
        self.assertEqual((candidate, region), (17, 1))
        np.testing.assert_allclose(point, [0.0, 0.25, 1.0])

    def test_parse_point_rejects_a_point_outside_the_archived_bounds(self):
        with self.assertRaises(LandremanFortBOError):
            _parse_point("POINT 0 1 0.1 1.000001", 2)

    def test_parse_point_rejects_malformed_dimension(self):
        with self.assertRaises(LandremanFortBOError):
            _parse_point("POINT 0 1 0.1", 2)


if __name__ == "__main__":
    unittest.main()
