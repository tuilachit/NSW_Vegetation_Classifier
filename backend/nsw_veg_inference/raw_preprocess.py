"""Raw aerial + ELVIS preprocessing wrapper.

This module runs the standalone single-tile preprocessing script supplied with
the project assets. It creates the same 11-band GeoTIFF format used by training,
then returns the generated file bytes for model inference.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]
TILE_BUILDER = Path(__file__).resolve().with_name("tile_builder.py")
DEFAULT_ASC_DIR = APP_ROOT / "soil_assets" / "asc"
SOIL_STEM = "SoilType_ASC_NSW_v4_5_210429"
REQUIRED_SOIL_EXTS = (".shp", ".dbf", ".shx", ".prj")


def soil_assets_dir() -> Path:
    return Path(os.environ.get("VEGEMAP_ASC_DIR", DEFAULT_ASC_DIR)).expanduser()


def missing_soil_assets() -> list[str]:
    asc_dir = soil_assets_dir()
    return [str(asc_dir / f"{SOIL_STEM}{ext}") for ext in REQUIRED_SOIL_EXTS if not (asc_dir / f"{SOIL_STEM}{ext}").exists()]


def soil_assets_ready() -> bool:
    return not missing_soil_assets()


def _safe_name(name: str, fallback: str) -> str:
    base = Path(name or fallback).name
    return base.replace("/", "_").replace("\\", "_") or fallback


def _link_soil_assets(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    matches = sorted(src_dir.glob(f"{SOIL_STEM}.*"))
    if not matches:
        raise FileNotFoundError(f"No ASC soil files found in {src_dir}")

    for src in matches:
        dst = dst_dir / src.name
        try:
            dst.symlink_to(src)
        except OSError:
            shutil.copy2(src, dst)


def build_11_band_tile(
    aerial_bytes: bytes,
    lidar_zip_bytes: bytes,
    aerial_filename: str = "aerial.tif",
    lidar_filename: str = "elvis.zip",
) -> tuple[bytes, str, str]:
    """Build a model-ready 11-band GeoTIFF from raw uploaded inputs.

    Returns:
        output_bytes: generated 11-band GeoTIFF bytes.
        output_name: generated filename.
        log_text: stdout/stderr from the preprocessing run.
    """
    missing = missing_soil_assets()
    if missing:
        raise FileNotFoundError("Missing ASC soil assets: " + ", ".join(missing))
    if not TILE_BUILDER.exists():
        raise FileNotFoundError(f"Missing tile builder script: {TILE_BUILDER}")

    aerial_name = _safe_name(aerial_filename, "aerial.tif")
    lidar_name = _safe_name(lidar_filename, "elvis.zip")
    if not aerial_name.lower().endswith((".tif", ".tiff")):
        aerial_name = f"{Path(aerial_name).stem}.tif"
    if not lidar_name.lower().endswith(".zip"):
        lidar_name = f"{Path(lidar_name).stem}.zip"

    timeout = int(os.environ.get("VEGEMAP_PREPROCESS_TIMEOUT", "420"))
    with tempfile.TemporaryDirectory(prefix="vegemap_raw_") as tmp:
        workdir = Path(tmp)
        inputs_dir = workdir / "inputs"
        outputs_dir = workdir / "outputs"
        inputs_dir.mkdir()
        outputs_dir.mkdir()
        _link_soil_assets(soil_assets_dir(), workdir / "asc")

        (inputs_dir / aerial_name).write_bytes(aerial_bytes)
        (inputs_dir / lidar_name).write_bytes(lidar_zip_bytes)

        proc = subprocess.run(
            [sys.executable, str(TILE_BUILDER)],
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        log_text = "\n".join(part for part in [proc.stdout, proc.stderr] if part)
        if proc.returncode != 0:
            raise RuntimeError(f"Preprocessing failed with exit code {proc.returncode}\n{log_text[-4000:]}")

        output_name = f"{Path(aerial_name).stem}.tif"
        output_path = outputs_dir / output_name
        if not output_path.exists():
            outputs = sorted(outputs_dir.glob("*.tif"))
            if not outputs:
                raise RuntimeError(f"Preprocessing completed but no output GeoTIFF was written.\n{log_text[-4000:]}")
            output_path = outputs[0]
            output_name = output_path.name

        return output_path.read_bytes(), output_name, log_text
