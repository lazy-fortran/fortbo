import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.check_b5_campaign import B5AuditError, _check_ledger, check


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ledger(path: Path, *, seed: int, regions: int = 1, mode: str = "data-informed"):
    dimension = 20 if mode == "data-informed" else 80
    evaluations = []
    for candidate_id in range(256):
        row = {
            "candidate_id": candidate_id,
            "unit_x": [0.5] * dimension,
            "status": "ok",
            "value": float(candidate_id),
        }
        if regions > 1:
            row["region_id"] = candidate_id % regions
        evaluations.append(row)
    return {
        "schema_name": ("simsopt-dfo.b5-async-turbo-m"
                        if regions > 1 else "simsopt-dfo.b5-async-turbo"),
        "schema_version": 1,
        "source": {"git_commit": "a" * 40, "git_dirty": False,
                    "constellaration_commit": "b" * 40},
        "passed": True,
        "problem": {"case_id": "b5_simple_to_build_qi", "mode": mode,
                     "dimension": dimension},
        "configuration": {
            "method": ("completion-driven TuRBO-m with Thompson sampling"
                        if regions > 1 else
                        "completion-driven TuRBO-1 with Thompson sampling"),
            "seed": seed, "budget": 256, "workers": 8,
            "regions": regions, "initial_points_per_region": 40,
        },
        "evaluations": evaluations,
        "result": {"truth_calls": 256, "peak_concurrency": 8,
                   "failed_evaluations": 0},
    }


class B5CampaignTests(unittest.TestCase):
    def test_failed_row_cannot_invent_an_objective(self):
        raw = _ledger(Path("unused"), seed=1)
        raw["evaluations"][3]["status"] = "failed"
        raw["evaluations"][3]["value"] = None
        raw["result"]["failed_evaluations"] = 1
        _check_ledger(raw, Path("fixture.json"), mode="data-informed", regions=1)
        raw["evaluations"][3]["value"] = 4.0
        with self.assertRaises(B5AuditError):
            _check_ledger(raw, Path("fixture.json"), mode="data-informed", regions=1)

    def test_comparison_audit_rejects_changed_raw_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            turbo = root / "turbo"
            turbo.mkdir()
            turbo_m = root / "turbo-m"
            turbo_m.mkdir()

            raw_documents = []
            for mode in ("raw", "data-informed"):
                for seed in range(1, 6):
                    name = f"{mode}-seed-{seed}.json"
                    path = turbo / name
                    path.write_text(json.dumps(_ledger(path, seed=seed,
                                                       mode=mode)))
                    raw_documents.append({"file": name, "mode": mode,
                                          "seed": seed, "sha256": _digest(path)})
            comparison = {
                "schema_name": "simsopt-dfo.b5-turbo-comparison",
                "schema_version": 1, "passed": True,
                "source": {"git_commit": "a" * 40, "git_dirty": False,
                            "constellaration_commit": "b" * 40},
                "verification_source": {"git_commit": "c" * 40,
                                        "git_dirty": False,
                                        "constellaration_commit": "d" * 40},
                "configuration": {"case_id": "b5_simple_to_build_qi",
                                   "budget": 256, "workers": 8,
                                   "paired_seeds": [1, 2, 3, 4, 5]},
                "raw_documents": raw_documents,
                "accounting": {"truth_calls_per_run": 256,
                               "total_truth_calls": 2560},
            }
            turbo_comparison = turbo / "comparison.json"
            turbo_comparison.write_text(json.dumps(comparison))
            # A second valid-shaped comparison is unnecessary for this test;
            # the first audit must fail before it reaches the TuRBO-m input.
            with self.assertRaises(B5AuditError):
                check(turbo_comparison, turbo_comparison, root)
            raw_documents[0]["sha256"] = "0" * 64
            turbo_comparison.write_text(json.dumps(comparison))
            with self.assertRaises(B5AuditError):
                check(turbo_comparison, turbo_comparison, root)


if __name__ == "__main__":
    unittest.main()
