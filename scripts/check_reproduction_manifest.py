#!/usr/bin/env python3
"""Validate a versioned FortBO reproduction manifest and its source pins."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tarfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


SHA256 = re.compile(r"^[0-9a-f]{64}$")
KINDS = {"file", "archive_member"}
STATUSES = {"ready-for-replay", "ready-for-control-audit", "literature-only"}
RESULT_LABELS = {
    "fortbo-value-only",
    "fortbo-exact-derivative",
    "fortbo-variational-derivative",
    "published-parity-dturbo",
}


class ManifestError(ValueError):
    """The manifest is malformed or refers to an invalid source pin."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ManifestError(message)


def load_manifest(path: Path) -> Mapping[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            manifest = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestError(f"cannot read {path}: {error}") from error

    _require(isinstance(manifest, dict), "manifest root must be an object")
    _require(manifest.get("schema_version") == 1, "schema_version must be 1")
    _require(isinstance(manifest.get("config_id"), str), "config_id is required")
    _require(isinstance(manifest.get("config_revision"), str),
             "config_revision is required")
    if "status" in manifest:
        _require(manifest["status"] in STATUSES,
                 "status must be ready-for-replay, ready-for-control-audit, "
                 "or literature-only")
    if "fortbo_result_label" in manifest:
        _require(manifest["fortbo_result_label"] in RESULT_LABELS,
                 "fortbo_result_label is not a recognized evidence label")
    if manifest.get("published_derivative_parity") == "blocked":
        _require(manifest.get("fortbo_result_label") != "published-parity-dturbo",
                 "blocked published derivative parity cannot be labeled published-parity-dturbo")
    _require(isinstance(manifest.get("source"), list) and manifest["source"],
             "source must be a non-empty array")
    _require(isinstance(manifest.get("parameters"), dict),
             "parameters must be an object")

    source_ids = set()
    for index, artifact in enumerate(manifest["source"]):
        prefix = f"source[{index}]"
        _require(isinstance(artifact, dict), f"{prefix} must be an object")
        artifact_id = artifact.get("id")
        _require(isinstance(artifact_id, str) and artifact_id,
                 f"{prefix}.id is required")
        _require(artifact_id not in source_ids,
                 f"duplicate source id: {artifact_id}")
        source_ids.add(artifact_id)
        kind = artifact.get("kind")
        _require(kind in KINDS, f"{prefix}.kind must be file or archive_member")
        digest = artifact.get("sha256")
        _require(isinstance(digest, str) and SHA256.fullmatch(digest),
                 f"{prefix}.sha256 must be a lowercase SHA-256 digest")
        if kind == "file":
            _require(isinstance(artifact.get("path"), str) and artifact["path"],
                     f"{prefix}.path is required")
        else:
            _require(isinstance(artifact.get("archive"), str) and artifact["archive"],
                     f"{prefix}.archive is required")
            _require(isinstance(artifact.get("member"), str) and artifact["member"],
                     f"{prefix}.member is required")

    for index, artifact in enumerate(manifest["source"]):
        if artifact["kind"] == "archive_member":
            _require(artifact["archive"] in source_ids,
                     f"source[{index}].archive does not name a source artifact")
            archive = next(item for item in manifest["source"]
                           if item["id"] == artifact["archive"])
            _require(archive["kind"] == "file",
                     f"source[{index}].archive must name a file artifact")

    return manifest


def _resolve(path_text: str, root: Optional[Path]) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() or root is None else root / path


def _hash_stream(stream: Any) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def _source_path(artifact: Mapping[str, Any], by_id: Mapping[str, Mapping[str, Any]],
                 root: Optional[Path]) -> Path:
    if artifact["kind"] == "file":
        return _resolve(artifact["path"], root)
    archive = by_id[artifact["archive"]]
    return _resolve(archive["path"], root)


def verify_sources(manifest: Mapping[str, Any], root: Optional[Path],
                   allow_missing: bool, metadata_only: bool) -> Tuple[int, int]:
    by_id = {artifact["id"]: artifact for artifact in manifest["source"]}
    missing = 0
    failures = 0
    if metadata_only:
        print(f"metadata OK: {manifest['config_id']} ({len(by_id)} source pins)")
        return missing, failures

    for artifact in manifest["source"]:
        path = _source_path(artifact, by_id, root)
        if not path.is_file():
            print(f"MISSING {artifact['id']}: {path}")
            missing += 1
            continue
        try:
            if artifact["kind"] == "file":
                with path.open("rb") as stream:
                    actual = _hash_stream(stream)
            else:
                with tarfile.open(path, mode="r:") as archive:
                    try:
                        member = archive.getmember(artifact["member"])
                    except KeyError as error:
                        raise ManifestError(
                            f"{artifact['id']}: archive member is absent: "
                            f"{artifact['member']}") from error
                    extracted = archive.extractfile(member)
                    if extracted is None:
                        raise ManifestError(
                            f"{artifact['id']}: archive member is not a regular file")
                    with extracted:
                        actual = _hash_stream(extracted)
        except (OSError, tarfile.TarError, ManifestError) as error:
            print(f"ERROR {artifact['id']}: {error}")
            failures += 1
            continue
        if actual != artifact["sha256"]:
            print(f"MISMATCH {artifact['id']}: expected {artifact['sha256']}, "
                  f"got {actual}")
            failures += 1
        else:
            print(f"OK {artifact['id']}: {path}")

    if missing and not allow_missing:
        failures += missing
    return missing, failures


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--root", type=Path,
                        help="root for relative source paths")
    parser.add_argument("--allow-missing", action="store_true",
                        help="report unavailable external sources without failing")
    parser.add_argument("--metadata-only", action="store_true",
                        help="validate the manifest without reading external sources")
    args = parser.parse_args(argv)

    try:
        manifest = load_manifest(args.manifest)
        missing, failures = verify_sources(
            manifest, args.root, args.allow_missing, args.metadata_only)
    except ManifestError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    if failures:
        return 1
    if missing and not args.allow_missing:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
