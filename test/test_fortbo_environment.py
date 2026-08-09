"""Independent behavioral checks for the FortBO/fo environment preflight."""

import os
import stat
import tempfile
import unittest
from pathlib import Path

from scripts.fortbo_environment import (
    FortBOEnvironmentError,
    fortbo_path_dependencies,
    preflight_fo,
)


class FortBOEnvironmentTests(unittest.TestCase):
    def test_preflight_resolves_and_exercises_the_selected_fo(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "fortbo"
            source.mkdir()
            (source / "fpm.toml").write_text("name = 'fortbo'\n", encoding="utf-8")
            sentinel = root / "built"
            fo = root / "fo"
            fo.write_text(
                "#!/bin/sh\n"
                "if [ \"$1\" = \"--version\" ]; then\n"
                "  echo 'fo 0.3.2'\n"
                "elif [ \"$1\" = \"build\" ]; then\n"
                "  : > \"$FO_SENTINEL\"\n"
                "else\n"
                "  exit 3\n"
                "fi\n",
                encoding="utf-8",
            )
            fo.chmod(fo.stat().st_mode | stat.S_IXUSR)
            environment = dict(os.environ, FO_SENTINEL=str(sentinel))

            result = preflight_fo(str(fo), source, environment)

            self.assertEqual(result["command"], str(fo.resolve()))
            self.assertEqual(result["version"], "fo 0.3.2")
            self.assertTrue(sentinel.is_file())
            self.assertFalse(result["tests_run"])

    def test_recursive_path_dependency_inventory_is_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fortbo = root / "fortbo"
            fortnum = root / "fortnum"
            fortbo.mkdir()
            fortnum.mkdir()
            (fortbo / "fpm.toml").write_text(
                "[dependencies]\nfortnum = { path = '../fortnum' }\n",
                encoding="utf-8",
            )
            (fortnum / "fpm.toml").write_text("name = 'fortnum'\n", encoding="utf-8")

            self.assertEqual(
                fortbo_path_dependencies(fortbo),
                (fortbo.resolve(), fortnum.resolve()),
            )

    def test_missing_path_dependency_is_refused_before_build(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "fortbo"
            source.mkdir()
            (source / "fpm.toml").write_text(
                "[dependencies]\nfortnum = { path = '../fortnum' }\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(FortBOEnvironmentError, "fortnum"):
                fortbo_path_dependencies(source)


if __name__ == "__main__":
    unittest.main()
