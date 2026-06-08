from __future__ import annotations

import base64
import io
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from rasterio.io import MemoryFile
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from nsw_veg_inference.config import FINAL_CLASS_VALUES
from nsw_veg_inference.preprocess import read_11_channel_geotiff
from nsw_veg_inference.raw_preprocess import build_11_band_tile, missing_soil_assets, soil_assets_ready


APP_ROOT = Path(__file__).resolve().parent
ASSET_DIR = APP_ROOT / "model_assets"
BINARY_WEIGHTS = ASSET_DIR / "binary_veg_other_p2_best.weights.h5"
GROUP_WEIGHTS = ASSET_DIR / "group9_after_binary_p2_best.weights.h5"
METADATA = ASSET_DIR / "binary_run_metadata.json"

CLASS_COLOURS = {
    0: "#000000",
    1: "#b0d8e8",
    2: "#c97a3e",
    3: "#4a7c32",
    4: "#2a9d8f",
    5: "#a8c046",
    6: "#8b5cf6",
    7: "#1a5c2e",
    8: "#d4845a",
    9: "#52b788",
    17: "#4b5563",
}

CLASS_NAMES = {
    0: "No data",
    1: "Alpine",
    2: "Arid Shrublands",
    3: "Dry Sclerophyll Forests",
    4: "Wetlands",
    5: "Grass / Grassy Woodlands",
    6: "Heathlands",
    7: "Rainforests",
    8: "Semi-arid Woodlands",
    9: "Wet Sclerophyll Forests",
    17: "Other / non-target land cover",
}

app = FastAPI(title="VegeMap NSW Vegetation Classifier API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline: Any | None = None
model_load_error: str | None = None


def missing_assets() -> list[str]:
    return [str(p.relative_to(APP_ROOT)) for p in [BINARY_WEIGHTS, GROUP_WEIGHTS, METADATA] if not p.exists()]


@app.on_event("startup")
def load_models() -> None:
    global pipeline, model_load_error
    missing = missing_assets()
    if missing:
        model_load_error = "Missing model assets: " + ", ".join(missing)
        print(model_load_error)
        return

    try:
        from nsw_veg_inference.pipeline import VegetationPipeline

        pipeline = VegetationPipeline(
            binary_weights_path=str(BINARY_WEIGHTS),
            group_weights_path=str(GROUP_WEIGHTS),
            metadata_path=str(METADATA),
        )
        model_load_error = None
        print("Model loaded OK: binary gate + 9-group classifier")
    except Exception as exc:  # noqa: BLE001
        pipeline = None
        model_load_error = repr(exc)
        print("Model load failed:", model_load_error)


@app.get("/health")
def health() -> dict:
    return {
        "ok": pipeline is not None,
        "models_loaded": pipeline is not None,
        "missing_assets": missing_assets(),
        "model_load_error": model_load_error,
        "soil_assets_ready": soil_assets_ready(),
        "missing_soil_assets": missing_soil_assets(),
    }


@app.get("/model-info")
def model_info() -> dict:
    return {
        "pipeline": "binary Vegetation/Other gate + 9-group vegetation classifier",
        "input": {
            "tile_size": [512, 512],
            "channels": [
                "Red",
                "Green",
                "Blue",
                "SoilCode",
                "CHM",
                "CanopyCover",
                "StrataGround",
                "StrataUnderstorey",
                "StrataMidstorey",
                "StrataCanopy",
                "TWI",
            ],
        },
        "output_classes": FINAL_CLASS_VALUES,
        "metrics": {
            "binary_test_miou": 0.6535,
            "group9_test_miou": 0.2035,
        },
    }


def prediction_preview_png(pred: np.ndarray) -> str:
    rgb = np.zeros((*pred.shape, 3), dtype=np.uint8)
    for class_value, colour in CLASS_COLOURS.items():
        colour = colour.lstrip("#")
        rgb[pred == class_value] = [int(colour[i : i + 2], 16) for i in (0, 2, 4)]
    image = Image.fromarray(rgb, mode="RGB")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def prediction_geotiff(pred: np.ndarray, profile: dict) -> str:
    out_profile = profile.copy()
    out_profile.update(count=1, dtype="uint8", nodata=0)
    with MemoryFile() as memfile:
        with memfile.open(**out_profile) as dst:
            dst.write(pred.astype(np.uint8), 1)
        return base64.b64encode(memfile.read()).decode("ascii")


def class_summary(pred: np.ndarray) -> list[dict]:
    total = int(pred.size)
    values, counts = np.unique(pred, return_counts=True)
    rows = []
    for value, count in zip(values.tolist(), counts.tolist()):
        rows.append(
            {
                "class_value": int(value),
                "class_name": CLASS_NAMES.get(int(value), f"Class {value}"),
                "pixels": int(count),
                "fraction": float(count / max(total, 1)),
                "colour": CLASS_COLOURS.get(int(value), "#999999"),
            }
        )
    return rows


@app.post("/predict-tile")
async def predict_tile(file: UploadFile = File(...)) -> JSONResponse:
    if pipeline is None:
        raise HTTPException(status_code=503, detail=model_load_error or "Model is not loaded")
    if not file.filename.lower().endswith((".tif", ".tiff")):
        raise HTTPException(status_code=400, detail="Upload a 512x512 11-band GeoTIFF tile")

    data = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".tif") as tmp:
        tmp.write(data)
        tmp.flush()
        try:
            x_full, profile = read_11_channel_geotiff(tmp.name)
            pred = pipeline.predict_tile_array(x_full)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc

    return JSONResponse(
        {
            "ok": True,
            "filename": file.filename,
            "shape": list(pred.shape),
            "crs": str(profile.get("crs")) if profile.get("crs") is not None else None,
            "transform": list(profile.get("transform")) if profile.get("transform") is not None else None,
            "classes": class_summary(pred),
            "preview_png_base64": prediction_preview_png(pred),
            "prediction_geotiff_base64": prediction_geotiff(pred, profile),
        }
    )


@app.post("/predict-raw")
async def predict_raw(
    aerial: UploadFile = File(...),
    lidar_zip: UploadFile = File(...),
) -> JSONResponse:
    if pipeline is None:
        raise HTTPException(status_code=503, detail=model_load_error or "Model is not loaded")
    if not soil_assets_ready():
        raise HTTPException(status_code=503, detail={"message": "Missing ASC soil assets", "missing_soil_assets": missing_soil_assets()})
    if not aerial.filename.lower().endswith((".tif", ".tiff")):
        raise HTTPException(status_code=400, detail="Aerial input must be a 512x512 RGB GeoTIFF in EPSG:3857")
    if not lidar_zip.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="LiDAR input must be an ELVIS .zip archive")

    try:
        tile_bytes, tile_name, preprocess_log = build_11_band_tile(
            await aerial.read(),
            await lidar_zip.read(),
            aerial.filename,
            lidar_zip.filename,
        )
        with tempfile.NamedTemporaryFile(suffix=".tif") as tmp:
            tmp.write(tile_bytes)
            tmp.flush()
            x_full, profile = read_11_channel_geotiff(tmp.name)
            pred = pipeline.predict_tile_array(x_full)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Raw preprocessing/prediction failed: {exc}") from exc

    return JSONResponse(
        {
            "ok": True,
            "filename": tile_name,
            "preprocessed_filename": tile_name,
            "shape": list(pred.shape),
            "crs": str(profile.get("crs")) if profile.get("crs") is not None else None,
            "transform": list(profile.get("transform")) if profile.get("transform") is not None else None,
            "classes": class_summary(pred),
            "preview_png_base64": prediction_preview_png(pred),
            "prediction_geotiff_base64": prediction_geotiff(pred, profile),
            "preprocessing_log_tail": preprocess_log[-3000:],
        }
    )


@app.get("/")
def root() -> dict:
    return {"service": "VegeMap API", "docs": "/docs", "health": "/health"}
