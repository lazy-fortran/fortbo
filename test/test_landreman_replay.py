import hashlib
import json
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path

from scripts.run_landreman_original import ReplayError, prepare


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class LandremanReplayTests(unittest.TestCase):
    def test_slurm_wrapper_has_valid_shell_and_historical_resources(self):
        wrapper = Path(__file__).parents[1] / "slurm/landreman_exact_replay.sbatch"
        result = subprocess.run(["bash", "-n", str(wrapper)], check=False,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        source = wrapper.read_text(encoding="utf-8")
        for directive in (
            "#SBATCH --ntasks=5",
            "#SBATCH --cpus-per-task=13",
            "#SBATCH --gpus-per-node=4",
        ):
            self.assertIn(directive, source)
        self.assertIn("--execute", source)

    def test_fortbo_slurm_wrapper_uses_the_same_historical_allocation(self):
        wrapper = Path(__file__).parents[1] / "slurm/landreman_fortbo.sbatch"
        result = subprocess.run(["bash", "-n", str(wrapper)], check=False,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        source = wrapper.read_text(encoding="utf-8")
        for directive in (
            "#SBATCH --ntasks=5",
            "#SBATCH --cpus-per-task=13",
            "#SBATCH --gpus-per-node=4",
        ):
            self.assertIn(directive, source)
        self.assertIn("run_landreman_fortbo.py", source)
        self.assertIn("--check-only", source)
        self.assertIn("--gpu-bind=map_gpu:0,1,2,3", source)

    def test_prepare_extracts_and_records_the_declared_path_remap(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            archive = directory / "landreman.tar"
            prefix = "release/software/alpha_opt/"
            driver_name = prefix + "scripts/driver.py"
            pca_name = prefix + "data/pca.h5"
            vmec_name = prefix + "data/input.vmec"
            driver = b'pca_data_file = "/pscratch/pca.h5"\n'
            pca = b"pca bytes\n"
            vmec = b"vmec bytes\n"
            files = {
                driver_name: driver,
                pca_name: pca,
                vmec_name: vmec,
            }
            with tarfile.open(archive, "w") as output:
                for name, contents in files.items():
                    source = directory / Path(name).name
                    source.write_bytes(contents)
                    output.add(source, arcname=name)
            manifest = {
                "source": [
                    {"id": "landreman_archive", "kind": "file",
                     "path": str(archive), "sha256": digest(archive.read_bytes())},
                    {"id": "landreman_driver", "kind": "archive_member",
                     "archive": "landreman_archive", "member": driver_name,
                     "sha256": digest(driver)},
                    {"id": "landreman_pca_data", "kind": "archive_member",
                     "archive": "landreman_archive", "member": pca_name,
                     "sha256": digest(pca)},
                    {"id": "landreman_vmec_input", "kind": "archive_member",
                     "archive": "landreman_archive", "member": vmec_name,
                     "sha256": digest(vmec)},
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
                            "source_absolute_path_in_archive": "/pscratch/pca.h5",
                            "reason": "fixture remap",
                        },
                    },
                },
            }
            config = directory / "manifest.json"
            config.write_text(json.dumps(manifest), encoding="utf-8")
            run_root = directory / "run"

            replay = prepare(config, None, run_root, "python3", 0.0)

            package = run_root / "software" / "alpha_opt"
            self.assertEqual(
                (package / "scripts/driver.py").read_text(encoding="utf-8"),
                f'pca_data_file = "{package / "data/pca.h5"}"\n',
            )
            self.assertEqual((package / "data/pca.h5").read_bytes(), pca)
            self.assertEqual(replay["allocation"]["ranks"], 5)
            self.assertEqual(replay["command"][:5], [
                "srun", "--ntasks=5", "--cpus-per-task=13",
                "--cpu-bind=cores", "--gpu-bind=map_gpu:0,1,2,3",
            ])
            saved = json.loads((run_root / "replay.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["status"], "prepared")
            self.assertEqual(saved["driver"]["original_sha256"], digest(driver))

    def test_prepare_refuses_nonhistorical_allocation(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config = directory / "manifest.json"
            config.write_text(json.dumps({
                "source": [{"id": "landreman_archive", "path": "missing"}],
                "parameters": {"run": {"mpi": {"historical_mpi_ranks": 1}}},
            }), encoding="utf-8")
            with self.assertRaises(ReplayError):
                prepare(config, None, directory / "run", "python3", 0.0)


if __name__ == "__main__":
    unittest.main()
