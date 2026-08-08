#!/usr/bin/env python3
"""Independent behavioral checks for the common reproduction runner."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
RUNNER = ROOT / "scripts" / "run_fortbo_reproduction.py"


class ReproductionRunnerTests(unittest.TestCase):
    def run_runner(self, *arguments):
        return subprocess.run(
            [sys.executable, str(RUNNER), *arguments],
            check=False, capture_output=True, text=True,
        )

    def test_delay_oracle_has_deterministic_completion_and_best_trace(self):
        arguments = (
            "--implementation", "oracle", "--seed", "13", "--workers", "2",
            "--budget", "9", "--initial", "3",
        )
        first = self.run_runner(*arguments)
        second = self.run_runner(*arguments)
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(second.returncode, 0, second.stderr)
        left, right = json.loads(first.stdout), json.loads(second.stdout)
        self.assertEqual(left, right)
        self.assertEqual(left["schema_version"], 1)
        self.assertEqual(len(left["ledger"]), 9)
        self.assertNotEqual(
            [row["evaluation"] for row in left["ledger"]], list(range(9)),
            "the delay oracle must exercise completion order rather than submission order",
        )
        best = float("inf")
        for row in left["ledger"]:
            best = min(best, row["value"])
            self.assertEqual(row["best_so_far"], best)
        self.assertEqual(left["summary"]["best_value"], min(row["value"] for row in left["ledger"]))

    def test_botorch_absence_is_a_structured_refusal(self):
        result = self.run_runner("--implementation", "botorch", "--budget", "3", "--initial", "2")
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        if payload["run"]["status"] == "refused":
            self.assertIn("refusal_reason", payload["run"])
        else:
            self.assertEqual(payload["summary"]["evaluations"], 3)

    def test_physics_case_is_not_silently_replaced_by_synthetic_data(self):
        result = self.run_runner(
            "--implementation", "fortbo", "--case", "glas-bindel-dturbo-adamcv",
            "--budget", "2", "--initial", "1",
            "--config", str(ROOT / "configs/reproduction/glas-bindel-dturbo.json"),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["run"]["status"], "refused")
        self.assertEqual(payload["ledger"], [])
        self.assertIn("physics evaluators", payload["run"]["refusal_reason"])
        self.assertEqual(payload["run"]["result_label"], "fortbo-exact-derivative")

    def test_output_and_scratch_are_the_same_common_document(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            scratch = Path(directory) / "scratch"
            result = self.run_runner(
                "--implementation", "oracle", "--budget", "3", "--initial", "2",
                "--output", str(output), "--scratch", str(scratch),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(output.read_text()),
                             json.loads((scratch / "run.json").read_text()))


if __name__ == "__main__":
    unittest.main()
