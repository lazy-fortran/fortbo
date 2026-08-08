#!/usr/bin/env python3
"""Audit the Glas/Bindel harvest for executable reproduction inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tarfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional


class GlasHarvestError(ValueError):
    """The harvest pin or its negative evidence is invalid."""


def _source(manifest: Mapping[str, Any], artifact_id: str) -> Mapping[str, Any]:
    for artifact in manifest.get("source", []):
        if artifact.get("id") == artifact_id:
            return artifact
    raise GlasHarvestError(f"missing source pin: {artifact_id}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GlasHarvestError(message)


def _member_names(archive: tarfile.TarFile) -> list[str]:
    return [member.name for member in archive.getmembers()]


def _executable_input_names(names: Iterable[str]) -> list[str]:
    """Return likely source/data inputs, excluding paper and figure artifacts."""
    source_suffixes = {
        ".c", ".cc", ".cpp", ".f", ".f90", ".f95", ".jl", ".m", ".py",
        ".ipynb", ".h5", ".hdf5", ".nc", ".json", ".csv", ".yaml", ".yml",
    }
    return [
        name for name in names
        if Path(name).suffix.lower() in source_suffixes
    ]


def check(manifest_path: Path) -> None:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GlasHarvestError(f"cannot read manifest: {error}") from error

    archive_pin = _source(manifest, "glas_bindel_harvest")
    _require(archive_pin.get("kind") == "file", "Glas harvest pin must be a file")
    archive_path = Path(archive_pin["path"])
    _require(archive_path.is_file(), f"harvest is missing: {archive_path}")

    try:
        with archive_path.open("rb") as stream:
            actual_digest = hashlib.sha256(stream.read()).hexdigest()
        with tarfile.open(archive_path, mode="r:*") as archive:
            names = _member_names(archive)
    except (OSError, tarfile.TarError) as error:
        raise GlasHarvestError(f"cannot inspect harvest: {error}") from error
    _require(actual_digest == archive_pin.get("sha256"),
             f"harvest digest mismatch: expected {archive_pin.get('sha256')}, "
             f"got {actual_digest}")

    missing_inputs = manifest.get("missing_inputs", [])
    _require(len(missing_inputs) == 4,
             "manifest must keep the four declared Glas missing-input classes")
    source_names = _executable_input_names(names)
    _require(not source_names,
             "harvest unexpectedly contains executable or numerical input files: "
             + ", ".join(source_names))

    probes = {
        "FOCUS source": r"(^|/)(focus|run_.*dturbo|.*dturbo.*)\.(py|f90|jl|m)$",
        "W7-X numerical input": r"(^|/).*w7.?x.*\.(h5|hdf5|nc|dat|input|json)$",
        "perturbation covariance": r"(^|/).*covariance.*\.(h5|hdf5|nc|json|csv|npz)$",
        "optimizer ledger": r"(^|/).*(ledger|experiment|optimizer).*\.(json|csv|yaml|yml)$",
    }
    for label, pattern in probes.items():
        matches = [name for name in names if re.search(pattern, name, re.IGNORECASE)]
        _require(not matches, f"{label} unexpectedly present: {', '.join(matches)}")

    print(f"Glas harvest: PASS ({len(names)} members; no executable/data inputs)")


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args(argv)
    try:
        check(args.manifest)
    except GlasHarvestError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
