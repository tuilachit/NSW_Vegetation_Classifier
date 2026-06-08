"""Preprocessing for 11-channel NSW vegetation tiles."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio

from .config import (
    BINARY_IDXS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    INPUT_CHANNELS,
    RGB_IDXS,
    SOIL_IDX,
    ZSCORE_IDXS,
)


def load_normalization_metadata(path: str | Path) -> dict:
    with Path(path).open() as f:
        meta = json.load(f)
    if "channel_stats" not in meta:
        raise KeyError(f"{path} does not contain channel_stats")
    return meta


def preprocess_channels(x_full: np.ndarray, metadata: dict) -> np.ndarray:
    if x_full.shape[-1] != INPUT_CHANNELS:
        raise ValueError(f"Expected {INPUT_CHANNELS} channels, got {x_full.shape[-1]}")

    channel_stats = {int(k): v for k, v in metadata["channel_stats"].items()}
    soil_scale = float(metadata.get("soil_scale", 17.0))

    x = x_full.astype(np.float32, copy=True)
    imagenet_mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    imagenet_std = np.array(IMAGENET_STD, dtype=np.float32)

    rgb = x[..., RGB_IDXS] / 255.0
    x[..., RGB_IDXS] = (rgb - imagenet_mean) / imagenet_std

    soil = np.nan_to_num(x[..., SOIL_IDX], nan=0.0, posinf=0.0, neginf=0.0)
    x[..., SOIL_IDX] = np.clip(soil / max(soil_scale, 1.0), 0.0, 1.0)

    for idx in ZSCORE_IDXS:
        stats = channel_stats[idx]
        mean = float(stats["mean"])
        std = float(stats["std"])
        band = np.nan_to_num(x[..., idx], nan=mean, posinf=mean, neginf=mean)
        x[..., idx] = (band - mean) / max(std, 1e-6)

    for idx in BINARY_IDXS:
        x[..., idx] = np.clip(np.nan_to_num(x[..., idx], nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def read_11_channel_geotiff(path: str | Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        if src.count != INPUT_CHANNELS:
            raise ValueError(f"{path}: expected {INPUT_CHANNELS} bands, got {src.count}")
        arr = np.transpose(src.read().astype(np.float32), (1, 2, 0))
        profile = src.profile.copy()
    return arr, profile
