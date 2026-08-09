"""Behavioral checks for the Landreman FortBO protocol boundary."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.run_landreman_fortbo import (
    LandremanFortBOError,
    _parse_point,
    _protocol_command,
)
from scripts.check_landreman_fortbo import (
    LandremanFortBOCheckError,
    check,
)


def ledger(best_value=1.0):
    return {
        "schema_name": "fortbo.landreman-value-only",
        "schema_version": 1,
        "passed": True,
        "source": {
            "fortbo_commit": "a" * 40,
            "alpha_opt_commit": "b" * 40,
            "archive_sha256": "c" * 64,
        },
        "problem": {
            "case_id": "landreman-alpha-loss",
            "dimension": 20,
            "failure_policy": "FAIL retained by FortBO and excluded from surrogate",
        },
        "configuration": {"budget": 3, "workers": 2, "mpi_ranks": 3},
        "evaluations": [
            {"candidate_id": 0, "unit_x": [0.1] * 20, "region_id": 1,
             "worker_rank": 1, "status": "ok", "value": 1.0},
            {"candidate_id": 1, "unit_x": [0.2] * 20, "region_id": 1,
             "worker_rank": 2, "status": "ok", "value": 2.0},
            {"candidate_id": 2, "unit_x": [0.3] * 20, "region_id": 1,
             "worker_rank": 1, "status": "failed", "value": None},
        ],
        "result": {
            "truth_calls": 3,
            "failed_evaluations": 1,
            "best_value": best_value,
            "completion_order": [1, 0, 2],
        },
    }


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

    def test_ledger_checker_uses_an_independent_minimum_and_order_oracle(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fortbo.json"
            path.write_text(json.dumps(ledger()), encoding="utf-8")
            result = check(path, expected_budget=3, expected_workers=2,
                           expected_archive_sha256="c" * 64)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["successful"], 2)

    def test_ledger_checker_rejects_a_fabricated_best_value(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fortbo.json"
            path.write_text(json.dumps(ledger(best_value=0.5)), encoding="utf-8")
            with self.assertRaises(LandremanFortBOCheckError):
                check(path)


if __name__ == "__main__":
    unittest.main()
