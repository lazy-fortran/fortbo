"""Independent behavioral checks for the original-control oracle pair."""

import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path

from scripts.run_b5_oracle_pair import (
    PairError,
    _ledgers_passed,
    _run_process,
    main as run_pair,
)


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
    def run_checker(self, pair, *extra):
        return subprocess.run(
            [sys.executable, str(CHECKER), str(pair), *extra],
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

    def test_different_coordinate_transform_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pair = self.make_pair(root)
            fortbo = root / "fortbo.json"
            document = json.loads(fortbo.read_text(encoding="utf-8"))
            document["problem"]["transform_sha256"] = "changed-transform"
            fortbo.write_text(json.dumps(document), encoding="utf-8")
            result = self.run_checker(pair)
            self.assertEqual(result.returncode, 1)
            self.assertIn("transform digests differ", result.stderr)

    def test_checker_rebases_archived_remote_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pair = self.make_pair(root, fortbo_best=1.0)
            archive = root / "archive" / "run"
            archive.mkdir(parents=True)
            for name in ("oracle.json", "fortbo.json"):
                (archive / name).write_text(
                    (root / name).read_text(encoding="utf-8"), encoding="utf-8"
                )
            document = json.loads(pair.read_text(encoding="utf-8"))
            document["oracle"]["output"] = "/remote/run/oracle.json"
            document["fortbo"]["output"] = "/remote/run/fortbo.json"
            pair.write_text(json.dumps(document), encoding="utf-8")
            result = self.run_checker(
                pair,
                "--rebase-from", "/remote/run",
                "--rebase-to", str(archive),
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)

    def test_pair_runner_refuses_to_reuse_nonempty_run_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dfo = root / "dfo"
            (dfo / "scripts").mkdir(parents=True)
            (dfo / "scripts/run_b5_async_turbo.py").write_text("# fixture\n",
                                                                  encoding="utf-8")
            run_root = root / "run"
            run_root.mkdir()
            (run_root / "old.json").write_text("{}\n", encoding="utf-8")
            result = run_pair([
                "--mode", "data-informed", "--seed", "1",
                "--dfo-root", str(dfo), "--run-root", str(run_root),
                "--output", str(root / "pair.json"),
            ])
            self.assertEqual(result, 2)

    def test_failed_child_aborts_the_pair(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_root = root / "run"
            run_root.mkdir()
            abort = threading.Event()
            with self.assertRaises(PairError):
                _run_process(
                    "oracle",
                    [sys.executable, "-c", "import sys; sys.exit(7)"],
                    root,
                    os.environ.copy(),
                    root / "stdout.log",
                    root / "stderr.log",
                    run_root,
                    0,
                    abort,
                )
            self.assertTrue(abort.is_set())

    def test_pair_requires_both_child_ledgers_to_pass(self):
        passed = {"passed": True}
        failed = {"passed": False}
        self.assertTrue(_ledgers_passed(passed, passed))
        self.assertFalse(_ledgers_passed(passed, failed))
        self.assertFalse(_ledgers_passed(passed, None))

    def test_pair_runner_preflights_the_original_environment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dfo = root / "dfo"
            (dfo / "scripts").mkdir(parents=True)
            (dfo / "scripts/run_b5_async_turbo.py").write_text("# fixture\n",
                                                                  encoding="utf-8")
            result = run_pair([
                "--mode", "data-informed", "--seed", "1",
                "--dfo-root", str(dfo), "--original-python", sys.executable,
                "--run-root", str(root / "run"),
                "--output", str(root / "pair.json"),
            ])
            self.assertEqual(result, 2)


if __name__ == "__main__":
    unittest.main()
