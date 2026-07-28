from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from nsw_veg_inference.config import (
    BINARY_IDXS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    INPUT_CHANNELS,
    SOIL_IDX,
)
from nsw_veg_inference.preprocess import (
    load_normalization_metadata,
    preprocess_channels,
)


class PreprocessChannelsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.metadata = {
            "soil_scale": 17,
            "channel_stats": {
                "4": {"mean": 10, "std": 2},
                "10": {"mean": 5, "std": 4},
            },
        }

    def test_normalizes_every_channel_family_without_mutating_input(self) -> None:
        source = np.zeros((1, 2, INPUT_CHANNELS), dtype=np.float64)
        source[0, 0, :3] = [255, 128, 0]
        source[0, 0, SOIL_IDX] = 34
        source[0, 1, SOIL_IDX] = np.nan
        source[0, 0, 4] = 12
        source[0, 1, 4] = np.nan
        source[0, 0, 10] = 9
        source[0, 1, 10] = 1
        source[0, 0, BINARY_IDXS] = [-1, 0.25, 2, np.nan, np.inf]
        original = source.copy()

        result = preprocess_channels(source, self.metadata)

        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(np.array_equal(source, original, equal_nan=True))
        expected_rgb = (np.array([1, 128 / 255, 0]) - IMAGENET_MEAN) / IMAGENET_STD
        np.testing.assert_allclose(result[0, 0, :3], expected_rgb, rtol=1e-6)
        np.testing.assert_allclose(result[0, :, SOIL_IDX], [1, 0])
        np.testing.assert_allclose(result[0, :, 4], [1, 0])
        np.testing.assert_allclose(result[0, :, 10], [1, -1])
        np.testing.assert_allclose(result[0, 0, BINARY_IDXS], [0, 0.25, 1, 0, 1])
        self.assertTrue(np.isfinite(result).all())

    def test_rejects_tiles_with_the_wrong_channel_count(self) -> None:
        tile = np.zeros((4, 4, INPUT_CHANNELS - 1), dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "Expected 11 channels"):
            preprocess_channels(tile, self.metadata)

    def test_load_metadata_requires_channel_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            metadata_path = Path(temp_dir) / "metadata.json"
            metadata_path.write_text(json.dumps({"soil_scale": 17}), encoding="utf-8")

            with self.assertRaisesRegex(KeyError, "does not contain channel_stats"):
                load_normalization_metadata(metadata_path)


if __name__ == "__main__":
    unittest.main()
