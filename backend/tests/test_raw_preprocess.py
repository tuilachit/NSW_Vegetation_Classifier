from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nsw_veg_inference.raw_preprocess import (
    REQUIRED_SOIL_EXTS,
    SOIL_STEM,
    _safe_name,
    missing_soil_assets,
    soil_assets_ready,
)


class RawPreprocessTests(unittest.TestCase):
    def test_safe_name_removes_parent_directories_and_backslashes(self) -> None:
        self.assertEqual(_safe_name("../../uploads/aerial.tif", "fallback.tif"), "aerial.tif")
        self.assertEqual(_safe_name(r"folder\elvis.zip", "fallback.zip"), "folder_elvis.zip")
        self.assertEqual(_safe_name("", "fallback.zip"), "fallback.zip")

    def test_reports_missing_soil_bundle_until_all_required_files_exist(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            asc_dir = Path(temp_dir)
            with patch.dict(os.environ, {"VEGEMAP_ASC_DIR": str(asc_dir)}):
                self.assertEqual(len(missing_soil_assets()), len(REQUIRED_SOIL_EXTS))
                self.assertFalse(soil_assets_ready())

                for extension in REQUIRED_SOIL_EXTS:
                    (asc_dir / f"{SOIL_STEM}{extension}").touch()

                self.assertEqual(missing_soil_assets(), [])
                self.assertTrue(soil_assets_ready())


if __name__ == "__main__":
    unittest.main()
