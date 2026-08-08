import hashlib
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from scripts.check_glas_harvest import GlasHarvestError, check


class GlasHarvestTests(unittest.TestCase):
    def make_manifest(self, directory: Path, extra_name: str = None) -> Path:
        archive_path = directory / "harvest.tar"
        files = {"main.tex": b"paper\n", "figures/front.png": b"figure\n"}
        if extra_name:
            files[extra_name] = b"unexpected\n"
        paths = []
        for name, data in files.items():
            path = directory / Path(name).name
            path.write_bytes(data)
            paths.append((path, name))
        with tarfile.open(archive_path, "w") as archive:
            for path, name in paths:
                archive.add(path, arcname=name)
        manifest = {
            "source": [{
                "id": "glas_bindel_harvest",
                "kind": "file",
                "path": str(archive_path),
                "sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
            }],
            "missing_inputs": ["FOCUS source", "W7-X input",
                               "periodic GP/Fourier perturbation covariance",
                               "optimizer source and experiment seed ledger"],
        }
        manifest_path = directory / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return manifest_path

    def test_literature_only_harvest_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            check(self.make_manifest(Path(temporary)))

    def test_executable_source_is_not_misclassified_as_literature_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(GlasHarvestError):
                check(self.make_manifest(Path(temporary), "reproduce.py"))


if __name__ == "__main__":
    unittest.main()
