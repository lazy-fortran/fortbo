#!/usr/bin/env python3
"""Download and verify the public artifacts used by reproduction campaigns.

Archives and repositories live outside the source checkout. Downloads are
streamed to a temporary sibling and are moved into place only after a pinned
SHA-256 digest (when one is declared) matches. Git sources are checked out at
the declared immutable revision and are never overwritten if an existing
checkout is at a different revision.
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
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "configs/reproduction/source-downloads.json"
DEFAULT_ROOT = Path(
    os.environ.get("FORTBO_REPRODUCTION_SOURCE_ROOT", "/var/tmp/ert/fortbo-reproduction-sources")
)
DEFAULT_RESERVE_GIB = 8.0
SHA256_LENGTH = 64


class SourceError(RuntimeError):
    """A public source could not be downloaded or verified."""


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> list[Mapping[str, Any]]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SourceError(f"cannot read manifest {path}: {error}") from error
    if document.get("schema_version") != 1 or not isinstance(document.get("artifacts"), list):
        raise SourceError("source manifest must have schema_version 1 and an artifacts list")
    artifacts = document["artifacts"]
    for artifact in artifacts:
        if not isinstance(artifact, dict) or not artifact.get("id"):
            raise SourceError("each source artifact must be an object with an id")
        if artifact.get("kind") not in {"file", "git"}:
            raise SourceError(f"unsupported source kind for {artifact.get('id')!r}")
        if not artifact.get("url") or not artifact.get("relative_path"):
            raise SourceError(f"{artifact['id']}: url and relative_path are required")
        required_digests = artifact.get("required_digests", {})
        if not isinstance(required_digests, dict):
            raise SourceError(f"{artifact['id']}: required_digests must be an object")
        for required_path, digest in required_digests.items():
            if (not isinstance(required_path, str) or not required_path
                    or not isinstance(digest, str)
                    or len(digest) != SHA256_LENGTH
                    or any(character not in "0123456789abcdef" for character in digest)):
                raise SourceError(
                    f"{artifact['id']}: required_digests must map paths to lowercase SHA-256 digests"
                )
    return artifacts


def _safe_destination(root: Path, relative: str) -> Path:
    destination = (root / relative).resolve()
    try:
        destination.relative_to(root.resolve())
    except ValueError as error:
        raise SourceError(f"source path escapes destination root: {relative}") from error
    return destination


def _destination(artifact: Mapping[str, Any], root: Path) -> Path:
    local_env = artifact.get("local_path_env")
    local = os.environ.get(local_env, "") if local_env else ""
    if local:
        return Path(local).expanduser().resolve()
    return _safe_destination(root, artifact["relative_path"])


def _require_free_space(path: Path, required_bytes: int, reserve_gib: float) -> None:
    usage = shutil.disk_usage(path if path.exists() else path.parent)
    reserve = int(reserve_gib * 1024**3)
    if usage.free < required_bytes + reserve:
        raise SourceError(
            f"refusing download near full filesystem {usage.free / 1024**3:.2f} GiB free; "
            f"need {required_bytes / 1024**3:.2f} GiB plus {reserve_gib:.2f} GiB reserve"
        )


def _download(
    artifact: Mapping[str, Any], destination: Path, check_only: bool, reserve_gib: float
) -> None:
    expected = artifact.get("sha256")
    if destination.is_file():
        actual = _digest(destination)
        if expected and actual != expected:
            raise SourceError(
                f"{artifact['id']}: existing file digest mismatch: {actual} != {expected}"
            )
        print(f"OK {artifact['id']}: {destination} ({actual})")
        return
    if check_only:
        raise SourceError(f"MISSING {artifact['id']}: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    _require_free_space(destination.parent, int(artifact.get("size_bytes", 0)), reserve_gib)
    partial = destination.with_name(destination.name + ".part")
    if partial.exists():
        partial.unlink()
    request = urllib.request.Request(
        artifact["url"], headers={"User-Agent": "fortbo-reproduction-fetch/1"}
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
    except (OSError, urllib.error.URLError) as error:
        partial.unlink(missing_ok=True)
        raise SourceError(f"{artifact['id']}: download failed: {error}") from error
    actual = _digest(partial)
    if expected and actual != expected:
        partial.unlink(missing_ok=True)
        raise SourceError(f"{artifact['id']}: downloaded digest mismatch: {actual} != {expected}")
    partial.replace(destination)
    print(f"DOWNLOADED {artifact['id']}: {destination} ({actual})")


def _git_output(path: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise SourceError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _required_file(root: Path, relative: str, artifact_id: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise SourceError(
            f"{artifact_id}: required path escapes the checkout: {relative}"
        ) from error
    if not candidate.is_file():
        raise SourceError(f"{artifact_id}: required path is absent: {relative}")
    return candidate


def _checkout(
    artifact: Mapping[str, Any], destination: Path, check_only: bool, require_clean: bool
) -> None:
    revision = artifact.get("revision")
    if not revision:
        raise SourceError(f"{artifact['id']}: git artifacts require a revision")
    if destination.exists() and not (destination / ".git").exists():
        raise SourceError(f"{artifact['id']}: destination exists but is not a Git checkout: {destination}")
    if not destination.exists():
        if check_only:
            raise SourceError(f"MISSING {artifact['id']}: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            ["git", "clone", "--no-checkout", artifact["url"], str(destination)],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode:
            raise SourceError(f"{artifact['id']}: clone failed: {result.stderr.strip()}")
        _git_output(destination, "fetch", "--depth", "1", "origin", revision)
        _git_output(destination, "checkout", "--detach", revision)
    actual = _git_output(destination, "rev-parse", "HEAD")
    if actual != revision:
        raise SourceError(f"{artifact['id']}: revision {actual} != {revision}")
    dirty = _git_output(destination, "status", "--porcelain", "--untracked-files=all")
    if dirty and require_clean:
        raise SourceError(f"{artifact['id']}: checkout is dirty: {destination}")
    for required in artifact.get("required_paths", []):
        _required_file(destination, required, artifact["id"])
    for required, expected in artifact.get("required_digests", {}).items():
        actual_digest = _digest(_required_file(destination, required, artifact["id"]))
        if actual_digest != expected:
            raise SourceError(
                f"{artifact['id']}: required file digest mismatch for {required}: "
                f"{actual_digest} != {expected}"
            )
    tree_digest = hashlib.sha256(_git_output(destination, "ls-tree", "-r", "--full-tree", "HEAD").encode()).hexdigest()
    cleanliness = "clean" if not dirty else "DIRTY; reuse only, not an exact source"
    print(f"OK {artifact['id']}: {destination} ({revision}, tree {tree_digest}, {cleanliness})")


def _unpack(
    artifact: Mapping[str, Any], source: Path, root: Path, reserve_gib: float
) -> None:
    target_text = artifact.get("unpack_to")
    if not target_text:
        return
    target = _safe_destination(root, target_text)
    marker = target / ".fortbo-source-unpacked"
    if marker.is_file():
        print(f"OK {artifact['id']} unpacked: {target}")
        return
    target.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(source, mode="r:*") as archive:
            root_resolved = target.resolve()
            members = archive.getmembers()
            required_bytes = sum(member.size for member in members if member.isfile())
            _require_free_space(target.parent, required_bytes, reserve_gib)
            for member in members:
                member_path = (target / member.name).resolve()
                try:
                    member_path.relative_to(root_resolved)
                except ValueError as error:
                    raise SourceError(f"{artifact['id']}: archive path escapes target: {member.name}") from error
            archive.extractall(target)
    except (OSError, tarfile.TarError) as error:
        raise SourceError(f"{artifact['id']}: extraction failed: {error}") from error
    marker.write_text("verified by fetch_reproduction_sources.py\n", encoding="utf-8")
    print(f"UNPACKED {artifact['id']}: {target}")


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--artifact", action="append", dest="artifacts")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--unpack", action="store_true", help="extract archives declaring unpack_to")
    parser.add_argument("--min-free-gib", type=float, default=DEFAULT_RESERVE_GIB)
    parser.add_argument("--require-clean", action="store_true")
    args = parser.parse_args(argv)
    try:
        artifacts = _load(args.manifest)
        selected = set(args.artifacts or (artifact["id"] for artifact in artifacts))
        known = {artifact["id"] for artifact in artifacts}
        unknown = selected - known
        if unknown:
            raise SourceError("unknown artifact(s): " + ", ".join(sorted(unknown)))
        for artifact in artifacts:
            if artifact["id"] not in selected:
                continue
            destination = _destination(artifact, args.root)
            if artifact["kind"] == "file":
                _download(artifact, destination, args.check_only, args.min_free_gib)
                if args.unpack and destination.is_file():
                    _require_free_space(destination.parent, 0, args.min_free_gib)
                    _unpack(artifact, destination, args.root, args.min_free_gib)
            else:
                _checkout(artifact, destination, args.check_only, args.require_clean)
    except (OSError, SourceError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
