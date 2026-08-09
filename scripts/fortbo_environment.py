"""Preflight the source and ``fo`` environment used by FortBO runners."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Iterable, Mapping, Optional


class FortBOEnvironmentError(RuntimeError):
    """The selected FortBO source or build environment is not runnable."""


def resolve_fo_command(command: str) -> str:
    """Resolve a command name or path to the executable that will be used."""

    if not command:
        raise FortBOEnvironmentError("the fo command must not be empty")
    if os.sep in command or (os.altsep and os.altsep in command):
        candidate = Path(command).expanduser()
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        resolved = candidate.resolve()
    else:
        located = shutil.which(command)
        if located is None:
            raise FortBOEnvironmentError(
                f"cannot resolve fo command {command!r} through PATH"
            )
        resolved = Path(located).resolve()
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise FortBOEnvironmentError(f"fo command is not executable: {resolved}")
    return str(resolved)


def configure_fo_environment(
    environment: Optional[Mapping[str, str]], cache_root: Path
) -> dict[str, str]:
    """Use a run-local fo cache unless the caller selected one explicitly."""

    selected = dict(environment or os.environ)
    selected.setdefault("FO_CACHE_DIR", str(cache_root.resolve()))
    return selected


def fortbo_path_dependencies(root: Path) -> tuple[Path, ...]:
    """Return every recursive relative path dependency in fpm manifests."""

    pending = [root.resolve()]
    visited: set[Path] = set()
    missing: list[str] = []
    while pending:
        package = pending.pop()
        if package in visited:
            continue
        visited.add(package)
        manifest = package / "fpm.toml"
        if not manifest.is_file():
            missing.append(f"{package}/fpm.toml")
            continue
        try:
            document = tomllib.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as error:
            raise FortBOEnvironmentError(
                f"cannot read FortBO dependency manifest {manifest}: {error}"
            ) from error
        dependencies = document.get("dependencies", {})
        if not isinstance(dependencies, dict):
            continue
        for name, specification in dependencies.items():
            if not isinstance(specification, dict) or "path" not in specification:
                continue
            dependency_path = (package / specification["path"]).resolve()
            if not dependency_path.is_dir():
                missing.append(f"{name}={dependency_path}")
            else:
                pending.append(dependency_path)
    if missing:
        details = ", ".join(sorted(missing))
        raise FortBOEnvironmentError(
            "FortBO path dependencies are missing; prepare clean sibling "
            f"checkouts before launching: {details}"
        )
    return tuple(sorted(visited))


def _run_checked(
    command: list[str],
    cwd: Path,
    environment: Mapping[str, str],
    label: str,
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            env=dict(environment),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise FortBOEnvironmentError(f"{label} failed: {error}") from error
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()[-2000:]
        raise FortBOEnvironmentError(
            f"{label} exited with return code {result.returncode}: {detail}"
        )
    return result


def preflight_fo(
    command: str,
    root: Path,
    environment: Optional[Mapping[str, str]] = None,
    *,
    run_tests: bool = False,
    test_timeout: int = 60,
    timeout: float = 1200.0,
) -> dict[str, object]:
    """Resolve and exercise ``fo`` before a FortBO run is allowed to start."""

    root = root.resolve()
    if not root.is_dir() or not (root / "fpm.toml").is_file():
        raise FortBOEnvironmentError(f"FortBO source checkout is missing: {root}")
    dependencies = fortbo_path_dependencies(root)
    resolved = resolve_fo_command(command)
    selected_environment = dict(environment or os.environ)
    version = _run_checked(
        [resolved, "--version"], root, selected_environment, "fo --version", 60.0
    )
    _run_checked([resolved, "build"], root, selected_environment, "fo build", timeout)
    if run_tests:
        selected_environment["FO_TEST_TIMEOUT"] = str(test_timeout)
        _run_checked(
            [resolved, "test"], root, selected_environment,
            "fo test", timeout,
        )
    return {
        "command": resolved,
        "version": version.stdout.strip().splitlines()[0] if version.stdout.strip() else "",
        "source": str(root),
        "path_dependencies": [str(path) for path in dependencies],
        "fo_cache_dir": selected_environment.get("FO_CACHE_DIR"),
        "tests_run": run_tests,
        "test_timeout": test_timeout if run_tests else None,
    }


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fortbo-root", type=Path, required=True)
    parser.add_argument("--fortbo-command", default="fo")
    parser.add_argument("--test", action="store_true", help="run the full fo test suite")
    parser.add_argument("--test-timeout", type=int, default=60)
    args = parser.parse_args(argv)
    try:
        result = preflight_fo(
            args.fortbo_command,
            args.fortbo_root,
            run_tests=args.test,
            test_timeout=args.test_timeout,
        )
    except FortBOEnvironmentError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
