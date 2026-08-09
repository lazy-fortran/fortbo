"""Independent behavioral checks for the original-control oracle pair."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
CHECKER = ROOT / "scripts/check_oracle_pair.py"


def run_document(best_value, commit="upstream-commit"):
    values = [3.0, 1.0, None]
    return {
        "schema_name": "test.b5",
        "schema_version": 1,
        "source": {"constellaration_commit": commit},
        "problem": {"case_id": "b5", "mode": "data-informed", "dimension": 20},
        "configuration": {
            "mode": "data-informed",
            "regions": 1,
            "seed": 1,
            "budget": 3,
            "workers": 8,
        },
        "evaluations": [
            {"status": "ok", "value": values[0]},
            {"status": "ok", "value": values[1]},
            {"status": "failed", "value": values[2]},
        ],
        "result": {
            "truth_calls": 3,
            "failed_evaluations": 1,
            "best_value": best_value,
            "wall_seconds": 1.0,
        },
        "passed": True,
    }


class OraclePairTests(unittest.TestCase):
    def run_checker(self, pair):
        return subprocess.run(
            [sys.executable, str(CHECKER), str(pair)],
            check=False,
            capture_output=True,
            text=True,
        )

    def make_pair(self, directory, fortbo_best=1.0, fortbo_commit="upstream-commit"):
        oracle = directory / "oracle.json"
        fortbo = directory / "fortbo.json"
        oracle.write_text(json.dumps(run_document(1.0)), encoding="utf-8")
        fortbo.write_text(json.dumps(run_document(fortbo_best, fortbo_commit)), encoding="utf-8")
        pair = directory / "pair.json"
        pair.write_text(json.dumps({
            "schema_name": "fortbo.oracle-pair",
            "schema_version": 1,
            "status": "complete",
            "configuration": {
                "mode": "data-informed",
                "regions": 1,
                "seed": 1,
                "budget": 3,
                "workers": 8,
            },
            "oracle": {"output": str(oracle)},
            "fortbo": {"output": str(fortbo)},
        }), encoding="utf-8")
        return pair

    def test_original_ledger_is_used_as_behavioral_oracle(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = self.run_checker(self.make_pair(Path(temporary)))
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            self.assertIn('"fortbo_minus_oracle": 0.0', result.stdout)

    def test_inconsistent_best_value_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = self.run_checker(self.make_pair(Path(temporary), fortbo_best=2.0))
            self.assertEqual(result.returncode, 1)
            self.assertIn("best value is not the minimum", result.stderr)

    def test_different_upstream_commit_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = self.run_checker(
                self.make_pair(Path(temporary), fortbo_commit="different-commit")
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("different ConStellaration commits", result.stderr)


if __name__ == "__main__":
    unittest.main()
