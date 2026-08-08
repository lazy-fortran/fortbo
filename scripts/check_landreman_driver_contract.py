#!/usr/bin/env python3
"""Check the pinned Landreman driver against its recorded algorithm contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tarfile
from pathlib import Path
from typing import Any, Mapping, Optional


class DriverContractError(ValueError):
    """The archived driver does not implement the recorded replay contract."""


def _source(manifest: Mapping[str, Any], artifact_id: str) -> Mapping[str, Any]:
    for artifact in manifest.get("source", []):
        if artifact.get("id") == artifact_id:
            return artifact
    raise DriverContractError(f"missing source pin: {artifact_id}")


def _member_bytes(archive: tarfile.TarFile, member: str) -> bytes:
    extracted = archive.extractfile(member)
    if extracted is None:
        raise DriverContractError(f"archive member is not a regular file: {member}")
    with extracted:
        return extracted.read()


def _require(source: str, pattern: str, label: str) -> None:
    if re.search(pattern, source, flags=re.MULTILINE) is None:
        raise DriverContractError(f"cannot find {label} in archived driver")


def _check_digest(data: bytes, expected: Any, label: str) -> None:
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise DriverContractError(
            f"{label} digest mismatch: expected {expected}, got {actual}"
        )


def check_driver(source: str) -> None:
    """Check independent source-level invariants from the archived driver."""
    literals = {
        "particle count": r"^n_particles = 25000\s*$",
        "alpha trace horizon": r"^t_max = 1e-1\s*$",
        "alpha pitch": r"^tau = 0\.1\s*$",
        "alpha loss threshold": r"^maxloss = 0\.02\s*$",
        "alpha block": r"^t_block = 1e-3\s*$",
        "alpha tolerance": r"^tol = 1e-6\s*$",
        "alpha minimum step": r"^min_dt = 1e-9\s*$",
        "PCA dimension": r"^dim_x = 20\s*$",
        "evaluation budget": r"^num_evals = 10000\s*$",
        "batch size": r"^batch_size = 1\s*$",
        "max-B target": r"^max_B_target = 12\.0\s*$",
        "max-B iteration count": r"^max_B_iterations = 1\s*$",
        "vacuum setting": r"^vacuum = False\s*$",
        "VMEC failure value": r"^fail_val = 5\.5\s+",
        "DMerc failure value": r"^DMerc_fail_val = -0\.5\s*$",
        "failure-as-value policy": r"^treat_failures_as_big_number = True\s*$",
        "trust initial length": r"length: float = 0\.8",
        "trust minimum length": r"length_min: float = 0\.5\*\*7",
        "trust maximum length": r"length_max: float = 1\.6",
        "trust success tolerance": r"success_tolerance: int = 10",
        "candidate count": r"min\(5000, max\(2000, 200 \* X_turbo\.shape\[-1\]\)\)",
        "candidate perturbation probability": r"min\(20\.0 / d, 1\.0\)",
        "qEI restart count": r"num_restarts: int = 10",
        "qEI raw samples": r"raw_samples: int = 512",
        "noise interval": r"Interval\(1e-8, 1e-3\)",
        "ARD Matern kernel": r"MaternKernel\(nu=2\.5, ard_num_dims=X\.shape\[-1\]",
        "lengthscale interval": r"Interval\(0\.005, 4\.0\)",
    }
    for label, pattern in literals.items():
        _require(source, pattern, label)

    expressions = {
        "seeded scrambled initial Sobol":
            r"SobolEngine\(dim, scramble=True, seed=0\)",
        "worker-scaled initial design":
            r"n_init = max\(2 \* dim, \(size - 1\) \* 2\)",
        "analytic qEI construction":
            r"qExpectedImprovement\(model, Y_turbo\.max\(\)\)",
        "qEI batch size": r"q=batch_size,",
        "qEI restart argument": r"num_restarts=num_restarts,",
        "qEI raw-sample argument": r"raw_samples=raw_samples,",
        "exact Cholesky policy":
            r"gpytorch\.settings\.max_cholesky_size\(float\(\"inf\"\)\)",
        "asynchronous result polling":
            r"comm\.Iprobe\(source=MPI\.ANY_SOURCE, tag=RESULT_TAG",
        "asynchronous result receive":
            r"comm\.recv\(source=worker_rank, tag=RESULT_TAG\)",
        "completion-ordered objective history": r"Y_turbo_list\.append\(",
    }
    for label, pattern in expressions.items():
        _require(source, pattern, label)


def check(manifest_path: Path) -> None:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DriverContractError(f"cannot read manifest: {error}") from error

    archive_pin = _source(manifest, "landreman_archive")
    driver_pin = _source(manifest, "landreman_driver")
    environment_pin = _source(manifest, "landreman_ax_environment")
    if archive_pin.get("kind") != "file":
        raise DriverContractError("Landreman archive pin must be a file")
    if driver_pin.get("kind") != "archive_member":
        raise DriverContractError("Landreman driver pin must be an archive member")
    if driver_pin.get("archive") != archive_pin.get("id"):
        raise DriverContractError("Landreman driver does not point to the archive")
    if environment_pin.get("archive") != archive_pin.get("id"):
        raise DriverContractError("Landreman environment does not point to the archive")

    archive_path = Path(archive_pin["path"])
    if not archive_path.is_file():
        raise DriverContractError(f"archive is missing: {archive_path}")
    try:
        with tarfile.open(archive_path, mode="r:") as archive:
            driver_bytes = _member_bytes(archive, driver_pin["member"])
            environment_bytes = _member_bytes(archive, environment_pin["member"])
    except (OSError, tarfile.TarError, KeyError) as error:
        raise DriverContractError(f"cannot read archived driver: {error}") from error

    _check_digest(driver_bytes, driver_pin.get("sha256"), "driver")
    _check_digest(environment_bytes, environment_pin.get("sha256"), "environment")
    try:
        source = driver_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise DriverContractError("archived driver is not UTF-8") from error
    check_driver(source)
    try:
        environment = environment_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise DriverContractError("archived environment list is not UTF-8") from error
    expected_packages = {
        "torch": "2.8.0",
        "botorch": "0.15.1",
        "gpytorch": "1.14",
        "ax-platform": "1.1.2",
    }
    for package, version in expected_packages.items():
        _require(environment, rf"^{re.escape(package)}\s+{re.escape(version)}\s*$",
                 f"{package} {version} environment pin")
    run = manifest.get("parameters", {}).get("run", {})
    if run.get("environment_source") != environment_pin.get("id"):
        raise DriverContractError("manifest environment source is not the pinned list")
    if run.get("environment_packages") != expected_packages:
        raise DriverContractError("manifest environment package versions are incomplete")
    print("Landreman driver contract: PASS (scrambled Sobol, ARD GP, qEI, async completion)")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args(argv)
    try:
        check(args.manifest)
    except DriverContractError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
