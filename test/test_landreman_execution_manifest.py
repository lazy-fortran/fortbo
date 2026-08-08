import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from scripts.check_landreman_execution_manifest import (
    ExecutionManifestError,
    check,
)


class LandremanExecutionManifestTests(unittest.TestCase):
    def make_fixture(self, directory):
        archive_path = directory / "landreman.tar"
        job_name = "source/job"
        driver_name = "source/driver.py"
        job = b"#SBATCH --ntasks=5\n#SBATCH --cpus-per-task=13\n#SBATCH --gpus-per-node=4\n"
        driver = (
            b'pca_data_file = "/pscratch/archive/data.h5"\n'
        )
        job_path = directory / "job"
        driver_path = directory / "driver.py"
        job_path.write_bytes(job)
        driver_path.write_bytes(driver)
        with tarfile.open(archive_path, "w") as archive:
            archive.add(job_path, arcname=job_name)
            archive.add(driver_path, arcname=driver_name)
        manifest = {
            "source": [
                {"id": "landreman_archive", "kind": "file",
                 "path": str(archive_path)},
                {"id": "landreman_execution_job", "kind": "archive_member",
                 "archive": "landreman_archive", "member": job_name},
                {"id": "landreman_driver", "kind": "archive_member",
                 "archive": "landreman_archive", "member": driver_name},
            ],
            "parameters": {
                "run": {
                    "mpi": {
                        "historical_mpi_ranks": 5,
                        "historical_worker_ranks": 4,
                        "historical_gpus": 4,
                        "historical_cpus_per_task": 13,
                    },
                    "path_remapping": {
                        "source_absolute_path_in_archive": "/pscratch/archive/data.h5",
                        "portable_source": "landreman_pca_data",
                    },
                },
            },
        }
        manifest_path = directory / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return manifest_path

    def test_archived_scheduler_and_path_evidence_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            check(self.make_fixture(Path(temporary)))

    def test_changed_rank_evidence_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest_path = self.make_fixture(directory)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["parameters"]["run"]["mpi"]["historical_gpus"] = 3
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(ExecutionManifestError):
                check(manifest_path)


if __name__ == "__main__":
    unittest.main()
