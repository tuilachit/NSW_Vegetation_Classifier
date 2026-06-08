"""One-call inference wrapper for the two-model vegetation pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio

from .config import FINAL_CLASS_VALUES, PATCH_SIZE
from .models import load_weighted_models
from .preprocess import load_normalization_metadata, preprocess_channels, read_11_channel_geotiff


class VegetationPipeline:
    """Binary vegetation/Other gate + 9-group vegetation classifier."""

    def __init__(self, binary_weights_path: str, group_weights_path: str, metadata_path: str):
        self.metadata = load_normalization_metadata(metadata_path)
        self.binary_model, self.group_model = load_weighted_models(binary_weights_path, group_weights_path)

    def predict_tile_array(self, x_full: np.ndarray) -> np.ndarray:
        """Predict one 512x512x11 tile.

        Returns a uint8 raster using:
        - 0 = nodata, only if caller applies an external nodata mask
        - 1-9 = vegetation groups
        - 17 = Other / non-target
        """
        if x_full.shape[:2] != (PATCH_SIZE, PATCH_SIZE):
            raise ValueError(f"Expected {PATCH_SIZE}x{PATCH_SIZE} tile, got {x_full.shape[:2]}")

        x = preprocess_channels(x_full, self.metadata)
        batch = x[None, ...]

        binary_probs = self.binary_model.predict(batch, verbose=0)[0]
        group_probs = self.group_model.predict(batch, verbose=0)[0]

        binary_pred = np.argmax(binary_probs, axis=-1).astype(np.uint8)
        group_pred = np.argmax(group_probs, axis=-1).astype(np.uint8) + 1

        out = np.full(binary_pred.shape, FINAL_CLASS_VALUES["other"], dtype=np.uint8)
        out[binary_pred == 1] = group_pred[binary_pred == 1]
        return out

    def predict_geotiff_tile(self, input_path: str | Path, output_path: str | Path) -> np.ndarray:
        x_full, profile = read_11_channel_geotiff(input_path)
        pred = self.predict_tile_array(x_full)
        profile.update(count=1, dtype="uint8", nodata=FINAL_CLASS_VALUES["nodata"])
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(pred, 1)
        return pred
