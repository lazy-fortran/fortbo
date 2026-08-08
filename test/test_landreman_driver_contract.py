import hashlib
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from scripts.check_landreman_driver_contract import (
    DriverContractError,
    check,
)


CONTRACT_SOURCE = """
n_particles = 25000
t_max = 1e-1
tau = 0.1
maxloss = 0.02
t_block = 1e-3
tol = 1e-6
min_dt = 1e-9
max_B_target = 12.0
max_B_iterations = 1
vacuum = False
dim_x = 20
treat_failures_as_big_number = True
fail_val = 5.5  # failure
DMerc_fail_val = -0.5
num_evals = 10000
batch_size = 1

class TurboState:
    length: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    success_tolerance: int = 10

def generate_turbo_batch(X_turbo, Y_turbo, batch_size,
                         n_candidates: int = None,
                         num_restarts: int = 10,
                         raw_samples: int = 512):
    n_candidates = min(5000, max(2000, 200 * X_turbo.shape[-1]))
    d = X_turbo.shape[-1]
    sobol = SobolEngine(d, scramble=True)
    prob_perturb = min(20.0 / d, 1.0)
    ei = qExpectedImprovement(model, Y_turbo.max())
    X_next, acq_value = optimize_acqf(
        ei, q=batch_size,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
    )

likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
MaternKernel(nu=2.5, ard_num_dims=X.shape[-1],
             lengthscale_constraint=Interval(0.005, 4.0))
with gpytorch.settings.max_cholesky_size(float("inf")):
    pass
sobol = SobolEngine(dim, scramble=True, seed=0)
n_init = max(2 * dim, (size - 1) * 2)
comm.Iprobe(source=MPI.ANY_SOURCE, tag=RESULT_TAG)
trial_index, result_value = comm.recv(source=worker_rank, tag=RESULT_TAG)
Y_turbo_list.append(result_value)
"""

ENVIRONMENT = b"""ax-platform               1.1.2
botorch                   0.15.1
gpytorch                  1.14
torch                     2.8.0
"""


class LandremanDriverContractTests(unittest.TestCase):
    def make_manifest(self, directory: Path, source: bytes = None) -> Path:
        source = source or CONTRACT_SOURCE.encode()
        archive_path = directory / "landreman.tar"
        driver_name = "source/driver.py"
        environment_name = "source/Ax_modules_pip_list"
        driver_path = directory / "driver.py"
        environment_path = directory / "Ax_modules_pip_list"
        driver_path.write_bytes(source)
        environment_path.write_bytes(ENVIRONMENT)
        with tarfile.open(archive_path, "w") as archive:
            archive.add(driver_path, arcname=driver_name)
            archive.add(environment_path, arcname=environment_name)
        manifest = {
            "source": [
                {"id": "landreman_archive", "kind": "file",
                 "path": str(archive_path)},
                {"id": "landreman_driver", "kind": "archive_member",
                 "archive": "landreman_archive", "member": driver_name,
                 "sha256": hashlib.sha256(source).hexdigest()},
                {"id": "landreman_ax_environment", "kind": "archive_member",
                 "archive": "landreman_archive", "member": environment_name,
                 "sha256": hashlib.sha256(ENVIRONMENT).hexdigest()},
            ],
            "parameters": {
                "run": {
                    "environment_source": "landreman_ax_environment",
                    "environment_packages": {
                        "torch": "2.8.0",
                        "botorch": "0.15.1",
                        "gpytorch": "1.14",
                        "ax-platform": "1.1.2",
                    },
                },
            },
        }
        path = directory / "manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path

    def test_archived_algorithm_contract_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            check(self.make_manifest(Path(temporary)))

    def test_changed_qei_contract_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = CONTRACT_SOURCE.replace("raw_samples: int = 512",
                                             "raw_samples: int = 256")
            with self.assertRaises(DriverContractError):
                check(self.make_manifest(Path(temporary), source.encode()))


if __name__ == "__main__":
    unittest.main()
