#!/usr/bin/env python3
"""Check the Landreman archive's recorded execution evidence semantically."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tarfile
from pathlib import Path
from typing import Any, Mapping, Optional


class ExecutionManifestError(ValueError):
    """The archived job/source and the manifest disagree."""


def _path(text: str) -> Path:
    expanded = os.path.expandvars(text)
    if re.search(r"\$(?:[A-Za-z_][A-Za-z0-9_]*|\{[^}]+\})", expanded):
        raise ExecutionManifestError(f"an environment variable in source path is unset: {text}")
    return Path(expanded).expanduser()


def _source(manifest: Mapping[str, Any], artifact_id: str) -> Mapping[str, Any]:
    for artifact in manifest["source"]:
        if artifact.get("id") == artifact_id:
            return artifact
    raise ExecutionManifestError(f"missing source pin: {artifact_id}")


def _member_text(archive: tarfile.TarFile, member: str) -> str:
    extracted = archive.extractfile(member)
    if extracted is None:
        raise ExecutionManifestError(f"archive member is not a regular file: {member}")
    with extracted:
        return extracted.read().decode("utf-8")


def _required_int(text: str, pattern: str, label: str) -> int:
    match = re.search(pattern, text, flags=re.MULTILINE)
    if match is None:
        raise ExecutionManifestError(f"cannot find {label} in archived job")
    return int(match.group(1))


def check(manifest_path: Path) -> None:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExecutionManifestError(f"cannot read manifest: {error}") from error

    archive_pin = _source(manifest, "landreman_archive")
    job_pin = _source(manifest, "landreman_execution_job")
    driver_pin = _source(manifest, "landreman_driver")
    parameters = manifest.get("parameters", {})
    run = parameters.get("run", {})
    mpi = run.get("mpi", {})
    remapping = run.get("path_remapping", {})
    archive_path = _path(archive_pin["path"])
    if not archive_path.is_file():
        raise ExecutionManifestError(f"archive is missing: {archive_path}")
    if job_pin.get("archive") != archive_pin.get("id"):
        raise ExecutionManifestError("execution job does not point to the archive")
    if driver_pin.get("archive") != archive_pin.get("id"):
        raise ExecutionManifestError("driver does not point to the archive")

    with tarfile.open(archive_path, mode="r:") as archive:
        job = _member_text(archive, job_pin["member"])
        driver = _member_text(archive, driver_pin["member"])

    ranks = _required_int(job, r"^#SBATCH --ntasks=(\d+)\s*$", "MPI rank count")
    workers = _required_int(job, r"^#SBATCH --cpus-per-task=(\d+)\s*$", "CPU count")
    gpus = _required_int(job, r"^#SBATCH --gpus-per-node=(\d+)\s*$", "GPU count")
    source_match = re.search(r'^pca_data_file\s*=\s*[\"\']([^\"\']+)', driver,
                             flags=re.MULTILINE)
    if source_match is None:
        raise ExecutionManifestError("cannot find archived PCA source path")

    expected = {
        "historical_mpi_ranks": ranks,
        "historical_worker_ranks": ranks - 1,
        "historical_gpus": gpus,
        "historical_cpus_per_task": workers,
    }
    for key, actual in expected.items():
        if mpi.get(key) != actual:
            raise ExecutionManifestError(
                f"{key}: manifest has {mpi.get(key)!r}, archive has {actual!r}")
    if remapping.get("source_absolute_path_in_archive") != source_match.group(1):
        raise ExecutionManifestError("manifest path remap does not match archived driver")
    if remapping.get("portable_source") != "landreman_pca_data":
        raise ExecutionManifestError("portable PCA source is not the pinned data artifact")

    print(f"Landreman execution manifest: PASS ({ranks} ranks, {gpus} GPUs)")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args(argv)
    try:
        check(args.manifest)
    except ExecutionManifestError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
