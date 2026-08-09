#!/usr/bin/env python3
"""Prepare or execute the pinned Landreman BoTorch driver on Slurm.

The archived driver is kept intact except for its recorded absolute PCA path,
which is replaced in a generated copy by the extracted PCA member.  The
launcher extracts only ``software/alpha_opt`` from the archive, records both
driver digests and the remap, and refuses to execute without a Slurm
allocation.  Preparation is safe to run on a login node; physics execution is
intentionally an explicit ``--execute`` action inside the allocation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/reproduction/landreman-data-informed-turbo.json"
DEFAULT_RESERVE_GIB = 8.0
EXPECTED_RANKS = 5
EXPECTED_WORKERS = 4
EXPECTED_GPUS = 4
EXPECTED_CPUS = 13


class ReplayError(RuntimeError):
    """The exact-tool replay cannot be prepared or safely started."""


def _path(text: str) -> Path:
    expanded = os.path.expandvars(text)
    if "$" in expanded:
        raise ReplayError(f"an environment variable in source path is unset: {text}")
    return Path(expanded).expanduser().resolve()


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source(manifest: Mapping[str, Any], artifact_id: str) -> Mapping[str, Any]:
    for artifact in manifest.get("source", []):
        if artifact.get("id") == artifact_id:
            return artifact
    raise ReplayError(f"manifest is missing source pin: {artifact_id}")


def _member_digest(archive: tarfile.TarFile, name: str) -> str:
    try:
        member = archive.getmember(name)
    except KeyError as error:
        raise ReplayError(f"archive member is absent: {name}") from error
    extracted = archive.extractfile(member)
    if extracted is None:
        raise ReplayError(f"archive member is not a regular file: {name}")
    digest = hashlib.sha256()
    with extracted:
        for chunk in iter(lambda: extracted.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_member_target(root: Path, relative: str) -> Path:
    target = (root / relative).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as error:
        raise ReplayError(f"archive member escapes replay root: {relative}") from error
    return target


def _extract_alpha_opt(
    archive: tarfile.TarFile,
    prefix: str,
    target: Path,
    reserve_gib: float,
) -> int:
    members = [
        member for member in archive.getmembers()
        if member.name.startswith(prefix)
    ]
    if not members:
        raise ReplayError(f"archive has no alpha_opt tree at {prefix}")
    if any(member.issym() or member.islnk() for member in members):
        raise ReplayError("refusing symlink or hardlink in archived alpha_opt tree")
    required = sum(member.size for member in members if member.isfile())
    usage_path = target
    while not usage_path.exists():
        usage_path = usage_path.parent
    usage = shutil.disk_usage(usage_path)
    reserve = int(reserve_gib * 1024**3)
    if usage.free < required + reserve:
        raise ReplayError(
            f"refusing extraction with {usage.free / 1024**3:.2f} GiB free; "
            f"need {required / 1024**3:.2f} GiB plus {reserve_gib:.2f} GiB reserve"
        )
    target.mkdir(parents=True, exist_ok=True)
    for member in members:
        relative = member.name[len(prefix):]
        destination = _safe_member_target(target, relative)
        if member.isdir():
            destination.mkdir(parents=True, exist_ok=True)
            continue
        if not member.isfile():
            raise ReplayError(f"unsupported archive member type: {member.name}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        extracted = archive.extractfile(member)
        if extracted is None:
            raise ReplayError(f"archive member is not readable: {member.name}")
        with extracted, destination.open("wb") as output:
            shutil.copyfileobj(extracted, output, length=1024 * 1024)
        destination.chmod(member.mode & 0o777)
    return required


def _manifest_path(config: Mapping[str, Any], archive: Optional[Path]) -> Path:
    if archive is not None:
        return archive.resolve()
    pin = _source(config, "landreman_archive")
    return _path(str(pin["path"]))


def prepare(
    config_path: Path,
    archive_path: Optional[Path],
    run_root: Path,
    python: str,
    reserve_gib: float,
) -> dict[str, Any]:
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReplayError(f"cannot read replay manifest: {error}") from error
    archive_pin = _source(config, "landreman_archive")
    driver_pin = _source(config, "landreman_driver")
    pca_pin = _source(config, "landreman_pca_data")
    vmec_pin = _source(config, "landreman_vmec_input")
    run = config.get("parameters", {}).get("run", {})
    mpi = run.get("mpi", {})
    actual = {
        "ranks": mpi.get("historical_mpi_ranks"),
        "workers": mpi.get("historical_worker_ranks"),
        "gpus": mpi.get("historical_gpus"),
        "cpus_per_task": mpi.get("historical_cpus_per_task"),
    }
    expected = {
        "ranks": EXPECTED_RANKS,
        "workers": EXPECTED_WORKERS,
        "gpus": EXPECTED_GPUS,
        "cpus_per_task": EXPECTED_CPUS,
    }
    if actual != expected:
        raise ReplayError(f"manifest allocation is not the historical one: {actual}")

    archive = _manifest_path(config, archive_path)
    if not archive.is_file():
        raise ReplayError(f"Landreman archive is missing: {archive}")
    expected_archive = archive_pin.get("sha256")
    actual_archive = _digest(archive)
    if expected_archive and actual_archive != expected_archive:
        raise ReplayError(
            f"Landreman archive digest mismatch: {actual_archive} != {expected_archive}"
        )

    run_root = run_root.resolve()
    if run_root == ROOT or ROOT in run_root.parents:
        raise ReplayError("run-root must be outside the FortBO source checkout")
    if run_root.exists() and any(run_root.iterdir()):
        raise ReplayError(f"run-root must be new or empty: {run_root}")
    run_root.mkdir(parents=True, exist_ok=True)

    prefix = str(driver_pin["member"]).split("software/alpha_opt/", 1)[0] + "software/alpha_opt/"

    def relative_member(pin: Mapping[str, Any]) -> str:
        name = str(pin["member"])
        if not name.startswith(prefix):
            raise ReplayError(f"{pin['id']} is outside the archived alpha_opt tree")
        return name[len(prefix):]

    driver_relative = relative_member(driver_pin)
    pca_relative = relative_member(pca_pin)
    relative_member(vmec_pin)
    package_root = run_root / "software" / "alpha_opt"
    with tarfile.open(archive, mode="r:*") as tar:
        for pin in (driver_pin, pca_pin, vmec_pin):
            actual_member = _member_digest(tar, str(pin["member"]))
            if pin.get("sha256") and actual_member != pin["sha256"]:
                raise ReplayError(
                    f"{pin['id']} digest mismatch: {actual_member} != {pin['sha256']}"
                )
        extracted_bytes = _extract_alpha_opt(tar, prefix, package_root, reserve_gib)

    driver_path = package_root / driver_relative
    original_driver = driver_path.with_name(driver_path.name + ".archived")
    driver_source = driver_path.read_text(encoding="utf-8")
    remapping = run.get("path_remapping", {})
    source_absolute = remapping.get("source_absolute_path_in_archive")
    if not isinstance(source_absolute, str) or source_absolute not in driver_source:
        raise ReplayError("archived driver does not contain the declared PCA path remap")
    pca_path = package_root / pca_relative
    original_driver.write_text(driver_source, encoding="utf-8")
    driver_path.write_text(driver_source.replace(source_absolute, str(pca_path)), encoding="utf-8")
    driver_path.chmod(original_driver.stat().st_mode & 0o777)

    command = [
        "srun",
        "--ntasks=5",
        "--cpus-per-task=13",
        "--cpu-bind=cores",
        "--gpu-bind=map_gpu:0,1,2,3",
        python,
        driver_relative,
    ]
    replay = {
        "schema_name": "fortbo.landreman-exact-replay",
        "schema_version": 1,
        "config": str(config_path.resolve()),
        "archive": str(archive),
        "archive_sha256": actual_archive,
        "source_prefix": prefix,
        "extracted_bytes": extracted_bytes,
        "working_directory": str(package_root),
        "command": command,
        "allocation": expected,
        "driver": {
            "archived_member": str(driver_pin["member"]),
            "original_sha256": _digest(original_driver),
            "patched_sha256": _digest(driver_path),
        },
        "path_remapping": {
            "from": source_absolute,
            "to": str(pca_path),
            "reason": remapping.get("reason"),
        },
        "status": "prepared",
    }
    replay_path = run_root / "replay.json"
    replay_path.write_text(json.dumps(replay, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return replay


def execute(replay: Mapping[str, Any], run_root: Path) -> int:
    if not os.environ.get("SLURM_JOB_ID"):
        raise ReplayError("--execute is allowed only inside a Slurm allocation")
    command = list(replay["command"])
    package_root = Path(str(replay["working_directory"]))
    with (run_root / "stdout.log").open("w", encoding="utf-8") as stdout, \
            (run_root / "stderr.log").open("w", encoding="utf-8") as stderr:
        result = subprocess.run(command, cwd=package_root, stdout=stdout, stderr=stderr,
                                env={**os.environ, "OMP_PLACES": "cores",
                                     "OMP_PROC_BIND": "spread", "PYTHONUNBUFFERED": "1"},
                                check=False)
    updated = dict(replay)
    updated["status"] = "complete" if result.returncode == 0 else "failed"
    updated["returncode"] = result.returncode
    (run_root / "replay.json").write_text(
        json.dumps(updated, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result.returncode


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--archive", type=Path, help="override LANDREMAN_ARCHIVE")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--python", default=os.environ.get("LANDREMAN_PYTHON", sys.executable))
    parser.add_argument("--disk-reserve-gib", type=float, default=DEFAULT_RESERVE_GIB)
    parser.add_argument("--execute", action="store_true",
                        help="run the prepared command; requires SLURM_JOB_ID")
    args = parser.parse_args(argv)
    try:
        replay = prepare(args.config, args.archive, args.run_root, args.python,
                         args.disk_reserve_gib)
        if args.execute:
            return execute(replay, args.run_root.resolve())
        print(f"prepared {args.run_root.resolve() / 'replay.json'}")
        print("command: " + " ".join(replay["command"]))
        return 0
    except (OSError, ReplayError, tarfile.TarError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
