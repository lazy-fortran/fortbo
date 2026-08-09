"""Behavioral checks for source reuse and safe missing-source handling."""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
FETCHER = ROOT / "scripts/fetch_reproduction_sources.py"


class SourceFetchTests(unittest.TestCase):
    def run_fetch(self, manifest, root, *extra, environment=None):
        env = os.environ.copy()
        if environment:
            env.update(environment)
        return subprocess.run(
            [sys.executable, str(FETCHER), "--manifest", str(manifest), "--root", str(root), *extra],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )

    def make_manifest(self, directory, digest):
        manifest = directory / "sources.json"
        manifest.write_text(json.dumps({
            "schema_version": 1,
            "artifacts": [{
                "id": "fixture",
                "kind": "file",
                "url": "https://invalid.example/fixture",
                "relative_path": "download/fixture.bin",
                "local_path_env": "FIXTURE_SOURCE",
                "sha256": digest,
            }],
        }), encoding="utf-8")
        return manifest

    def test_existing_environment_source_is_verified_without_downloading(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / "existing.bin"
            source.write_bytes(b"existing source\n")
            manifest = self.make_manifest(directory, hashlib.sha256(source.read_bytes()).hexdigest())
            result = self.run_fetch(
                manifest,
                directory / "root",
                "--check-only",
                environment={"FIXTURE_SOURCE": str(source)},
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            self.assertIn("OK fixture", result.stdout)
            self.assertFalse((directory / "root" / "download").exists())

    def test_missing_check_only_source_fails_before_network_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest = self.make_manifest(directory, "0" * 64)
            result = self.run_fetch(manifest, directory / "root", "--check-only")
            self.assertEqual(result.returncode, 1)
            self.assertIn("MISSING fixture", result.stderr)


if __name__ == "__main__":
    unittest.main()
