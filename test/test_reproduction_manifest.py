#!/usr/bin/env python3
"""Behavioral tests for the reproduction manifest checker."""

import hashlib
import json
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "check_reproduction_manifest.py"


def digest(data):
    return hashlib.sha256(data).hexdigest()


class ReproductionManifestTests(unittest.TestCase):
    def make_manifest(self, directory, expected):
        source = directory / "source.bin"
        source.write_bytes(b"independent source bytes\n")
        manifest = {
            "schema_version": 1,
            "config_id": "fixture",
            "config_revision": "test",
            "source": [{
                "id": "source",
                "kind": "file",
                "path": "source.bin",
                "sha256": expected,
            }],
            "parameters": {},
        }
        path = directory / "fixture.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path

    def run_checker(self, manifest, *extra):
        return subprocess.run(
            [sys.executable, str(SCRIPT), str(manifest), "--root",
             str(manifest.parent), *extra],
            check=False, capture_output=True, text=True)

    def test_matching_file_pin_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            expected = digest(b"independent source bytes\n")
            result = self.run_checker(self.make_manifest(directory, expected))
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            self.assertIn("OK source", result.stdout)

    def test_changed_file_pin_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            expected = digest(b"independent source bytes\n")
            manifest = self.make_manifest(directory, expected)
            (directory / "source.bin").write_bytes(b"changed bytes\n")
            result = self.run_checker(manifest)
            self.assertEqual(result.returncode, 1)
            self.assertIn("MISMATCH source", result.stdout)

    def test_missing_external_source_can_be_audited_explicitly(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest = self.make_manifest(
                directory, digest(b"independent source bytes\n"))
            (directory / "source.bin").unlink()
            result = self.run_checker(manifest)
            self.assertEqual(result.returncode, 1)
            allowed = self.run_checker(manifest, "--allow-missing")
            self.assertEqual(allowed.returncode, 0)
            self.assertIn("MISSING source", allowed.stdout)

    def test_archive_member_pin_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            member_data = b"archive member bytes\n"
            archive_path = directory / "source.tar"
            member_path = directory / "member.txt"
            member_path.write_bytes(member_data)
            with tarfile.open(archive_path, "w") as archive:
                archive.add(member_path, arcname="member.txt")
            manifest = {
                "schema_version": 1,
                "config_id": "archive-fixture",
                "config_revision": "test",
                "source": [
                    {
                        "id": "archive",
                        "kind": "file",
                        "path": "source.tar",
                        "sha256": digest(archive_path.read_bytes()),
                    },
                    {
                        "id": "member",
                        "kind": "archive_member",
                        "archive": "archive",
                        "member": "member.txt",
                        "sha256": digest(member_data),
                    },
                ],
                "parameters": {},
            }
            manifest_path = directory / "archive-fixture.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            result = self.run_checker(manifest_path)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            self.assertIn("OK member", result.stdout)

    def test_blocked_published_derivative_lane_cannot_claim_parity(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest = self.make_manifest(
                directory, digest(b"independent source bytes\n"))
            data = json.loads(manifest.read_text(encoding="utf-8"))
            data.update({
                "fortbo_result_label": "published-parity-dturbo",
                "published_derivative_parity": "blocked",
            })
            manifest.write_text(json.dumps(data), encoding="utf-8")
            result = self.run_checker(manifest, "--metadata-only")
            self.assertEqual(result.returncode, 2)
            self.assertIn("cannot be labeled published-parity-dturbo",
                          result.stderr)

    def test_control_audit_status_is_valid(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest = self.make_manifest(
                directory, digest(b"independent source bytes\n"))
            data = json.loads(manifest.read_text(encoding="utf-8"))
            data["status"] = "ready-for-control-audit"
            manifest.write_text(json.dumps(data), encoding="utf-8")
            result = self.run_checker(manifest, "--metadata-only")
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_local_variational_derivative_label_is_valid(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest = self.make_manifest(
                directory, digest(b"independent source bytes\n"))
            data = json.loads(manifest.read_text(encoding="utf-8"))
            data["fortbo_result_label"] = "fortbo-variational-derivative"
            manifest.write_text(json.dumps(data), encoding="utf-8")
            result = self.run_checker(manifest, "--metadata-only")
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
