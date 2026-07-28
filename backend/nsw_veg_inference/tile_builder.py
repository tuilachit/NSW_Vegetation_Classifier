"""
inference.py — Single-tile inference preprocessing pipeline for the NSW Vegetation Classifier.

PURPOSE
-------
The training pipeline (data_preprocessing/run_all.py) processes many geographic regions,
each a grid of 512×512 px tiles, and produces multi-modal input stacks for model
training. At inference time, the user supplies a single aerial photograph and a
matching LiDAR point cloud; this script applies the same preprocessing steps to
produce a single 11-channel .tif that can be fed directly to the trained U-Net model.

INPUTS  (place all files before running)
-----------------------------------------
1. ./inputs/<name>.tif
       A single 512×512 px RGB aerial photograph as a GeoTIFF.
       MUST be in EPSG:3857 (Web Mercator) — the same projection used during training.
       The script derives the geographic bounding box from the file's embedded transform;
       no manual coordinate entry is required.

2. ./inputs/<name>.zip
       A single ELVIS download package (.zip) containing LAZ or LAS point cloud files
       for the same area as the aerial image.
       Download via: https://elevation.fsdf.org.au/

3. ./asc/SoilType_ASC_NSW_v4_5_210429.shp  (plus companion .dbf, .shx, .prj files)
       The NSW ASC soil type shapefile used during training (~384 MB).
       Only the polygons that overlap the aerial image's bounding box are loaded,
       so the full file does not need to be in memory.

OUTPUT
------
./outputs/<aerial_stem>.tif
       11-band float32 GeoTIFF, LZW-compressed, EPSG:3857.
       Same spatial extent and transform as the input aerial.

       Band layout (matches data_preprocessing/merge.py exactly — do NOT reorder):
           Band  1: Red                 float32, range 0–255
           Band  2: Green               float32, range 0–255
           Band  3: Blue                float32, range 0–255
           Band  4: Soil code           float32 (categorical); 0=nodata, 1–17=ASC class
           Band  5: CHM                 float32, metres above ground; NaN where no LiDAR
           Band  6: Canopy Cover        float32, fraction 0–1
           Band  7: Strata Ground       float32, fraction 0–1 (returns 0.0–0.5 m)
           Band  8: Strata Understorey  float32, fraction 0–1 (returns 0.5–2.0 m)
           Band  9: Strata Midstorey    float32, fraction 0–1 (returns 2.0–10.0 m)
           Band 10: Strata Canopy       float32, fraction 0–1 (returns 10.0 m+)
           Band 11: TWI                 float32, Topographic Wetness Index

USAGE
-----
    cd inference
    python inference.py

DEPENDENCIES
------------
    numpy, rasterio, geopandas, shapely, pyproj, laspy, scipy, numba

NOTES
-----
- EPSG:3857 introduces ~20% scale distortion at NSW latitudes. A "1 m cell" in the
  DEM corresponds to ~0.8 m on the ground. This matches the training pipeline and
  must be preserved here so the model sees consistent feature scales.

- CHM values of NaN indicate pixels with no LiDAR returns. The model's normalisation
  layer handles NaN internally; do not replace them with 0 before inference.

- The first run will take several extra seconds because numba compiles the D8 flow
  accumulation function on first call. Subsequent runs within the same Python process
  are faster (compiled code is cached in memory).
"""

import math
import os
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import geopandas as gpd
import laspy
import numba
import numpy as np
import rasterio
import rasterio.transform
import scipy.ndimage
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.features import rasterize
from rasterio.transform import from_bounds, from_origin
from shapely.geometry import box

# =============================================================================
# CONSTANTS
# Mirrored from data_preprocessing/config.py and the individual preprocessing
# scripts.  Kept inline here so this script is fully standalone and can be
# dropped into a fresh environment without any imports from the training code.
# =============================================================================

# The model was trained on 512×512 px tiles; this must not change at inference time.
TILE_SIZE = 512

# Single CRS used throughout the pipeline: Web Mercator.
CRS_EPSG = 3857

# Australian Soil Classification order → integer code.
# 0 is reserved for nodata (pixels not covered by any soil polygon).
# These codes must match what was used during training (soil.py lines 20–38).
ASC_ORDER_MAP = {
    "Calcarosols":          1,
    "Chromosols":           2,
    "Dermosols":            3,
    "Ferrosols":            4,
    "Hydrosols":            5,
    "Kandosols":            6,
    "Kurosols":             7,
    "Kurosols (natric)":    8,
    "Organosols":           9,
    "Podosols":            10,
    "Rudosols":            11,
    "Rudosols (alluvial)": 12,
    "Sodosols":            13,
    "Tenosols":            14,
    "Vertosols":           15,
    "Not assessed":        16,
    "Water":               17,
}

# Minimum return height (metres above ground) for a point to count as "canopy".
CANOPY_THRESHOLD = 2.0

# Height bands used for the four strata density features.
# Each entry is (lower_bound_inclusive, upper_bound_exclusive) in metres above ground.
STRATA_BINS = [(0.0, 0.5), (0.5, 2.0), (2.0, 10.0), (10.0, 1e9)]

# The DEM is built on a grid that extends DEM_BUFFER metres beyond the AOI on
# every side.  This gives the D8 flow accumulation algorithm upstream-catchment
# context for cells near the tile boundary, preventing edge artefacts in the TWI.
DEM_BUFFER = 500   # metres (EPSG:3857 units; ~20% larger than true metres at NSW latitudes)

# Spatial resolution of the DEM grid.  Using 1 m keeps the DEM detailed enough
# for accurate height normalisation without excessive memory usage.
DEM_RESOLUTION = 1.0   # metres per cell (EPSG:3857 units)

# ASPRS point classification codes treated as ground returns for DEM generation.
# Class 2 = Ground (standard), Class 8 = Model Key-point (used by some NSW datasets).
GROUND_CLASSES = (2, 8)

# Band metadata tag written to the output GeoTIFF.  Matches merge.py exactly.
BAND_TAGS = (
    "1=Red,2=Green,3=Blue,4=SoilCode,"
    "5=CHM,6=CanopyCover,"
    "7=StrataGround,8=StrataUnderstorey,"
    "9=StrataMidstorey,10=StrataCanopy,"
    "11=TWI"
)
N_BANDS = 11

# Fixed filesystem layout, relative to the directory where this script lives.
# All paths assume the script is executed from inside the inference/ directory.
INPUTS_DIR     = "inputs"
ASC_DIR        = "asc"
OUTPUTS_DIR    = "outputs"
SOIL_SHAPEFILE = os.path.join(ASC_DIR, "SoilType_ASC_NSW_v4_5_210429.shp")
EXTRACT_DIR    = os.path.join(INPUTS_DIR, "_extracted")  # LAZ/LAS files unpacked here


# =============================================================================
# DATA CONTAINER
# =============================================================================

@dataclass
class ElvisData:
    """Holds the list of extracted LAZ/LAS point cloud file paths from an ELVIS package."""
    las_files: List[Path] = field(default_factory=list)


# =============================================================================
# HELPER — MGA ZONE FALLBACK EPSG
# =============================================================================

def _mga_fallback_epsg(west: float, east: float, south: float, north: float) -> int:
    """Return the GDA2020 MGA zone EPSG code for the centre of the given AOI.

    NSW ELVIS LiDAR data is distributed in GDA2020 MGA (EPSG:785x) coordinates.
    The correct zone depends on longitude:
        Zone 54 → EPSG:7854  (138–144°E)
        Zone 55 → EPSG:7855  (144–150°E)
        Zone 56 → EPSG:7856  (150–156°E)

    This function derives the zone from the AOI centre rather than hardcoding one,
    which prevents mismatches when a file has no embedded CRS and the AOI falls
    in a different zone than a hardcoded default.

    Args:
        west, east:   Easting bounds of the AOI in EPSG:3857.
        south, north: Northing bounds of the AOI in EPSG:3857.

    Returns:
        EPSG integer (e.g., 7855 for Zone 55).
    """
    to_wgs84   = Transformer.from_crs(CRS_EPSG, 4326, always_xy=True)
    lon_centre = to_wgs84.transform((west + east) / 2, (south + north) / 2)[0]
    zone       = math.floor((lon_centre + 180) / 6) + 1
    return 7800 + zone   # GDA2020 MGA zone EPSGs follow this formula


# =============================================================================
# STAGE 1 — READ AERIAL IMAGE, DERIVE BOUNDING BOX
# =============================================================================

def read_aerial(
    inputs_dir: str,
) -> Tuple[np.ndarray, object, float, float, float, float]:
    """Read the aerial GeoTIFF and extract RGB data plus the geographic bounding box.

    Validates that:
        - Exactly one .tif file is present in inputs_dir.
        - The image is 512×512 px (model input size).
        - The image is in EPSG:3857 (the training CRS).

    The bounding box (WEST, SOUTH, EAST, NORTH) is read from the file's embedded
    affine transform — no manual coordinate entry is needed.

    Args:
        inputs_dir: Directory containing the aerial .tif file (i.e., ./inputs/).

    Returns:
        aerial    — (3, 512, 512) float32 NumPy array; bands are [Red, Green, Blue]
        transform — rasterio Affine transform for the tile (EPSG:3857)
        west      — leftmost easting of the tile in EPSG:3857 metres
        south     — bottommost northing of the tile in EPSG:3857 metres
        east      — rightmost easting
        north     — topmost northing
    """
    tif_files = list(Path(inputs_dir).glob("*.tif"))

    if len(tif_files) == 0:
        raise FileNotFoundError(
            f"No .tif file found in ./{inputs_dir}/. "
            "Place a single 512×512 RGB GeoTIFF in EPSG:3857 there and re-run."
        )
    if len(tif_files) > 1:
        names = [f.name for f in tif_files]
        raise ValueError(
            f"Multiple .tif files found in ./{inputs_dir}/: {names}. "
            "Place exactly one aerial image there."
        )

    aerial_path = tif_files[0]
    print(f"  Aerial file: {aerial_path.name}")

    with rasterio.open(aerial_path) as src:
        # Enforce EPSG:3857 — everything downstream (soil rasterisation, LiDAR
        # reprojection, DEM grid) is computed in this CRS.
        if src.crs is None or src.crs.to_epsg() != CRS_EPSG:
            raise ValueError(
                f"Aerial CRS is '{src.crs}' but must be EPSG:{CRS_EPSG} (Web Mercator). "
                "Reproject the image to EPSG:3857 before running inference."
            )

        # The model accepts exactly 512×512 px patches.
        if src.width != TILE_SIZE or src.height != TILE_SIZE:
            raise ValueError(
                f"Aerial image is {src.width}×{src.height} px but must be "
                f"{TILE_SIZE}×{TILE_SIZE} px."
            )

        # Read bands 1, 2, 3 as float32.  Explicitly requesting [1,2,3] handles
        # RGBA images (4 bands) gracefully by ignoring the alpha channel.
        if src.count < 3:
            raise ValueError(
                f"Aerial image has {src.count} band(s); need at least 3 (RGB)."
            )
        aerial    = src.read([1, 2, 3]).astype(np.float32)   # (3, 512, 512)
        transform = src.transform
        bounds    = src.bounds   # rasterio BoundingBox: (left, bottom, right, top)

    west  = bounds.left
    south = bounds.bottom
    east  = bounds.right
    north = bounds.top

    print(f"  Bounding box (EPSG:3857): W={west:.1f}  S={south:.1f}  E={east:.1f}  N={north:.1f}")
    print(f"  Tile covers ~{east - west:.1f} m (W–E) × {north - south:.1f} m (S–N)")
    return aerial, transform, west, south, east, north


# =============================================================================
# STAGE 2 — SOIL RASTERISATION
# Adapted from data_preprocessing/soil.py.
# Changes: no intermediate GeoTIFF written; returns array directly; bbox is a
# parameter instead of being read from the config module; no tile-grid loop.
# =============================================================================

def load_soil_data(
    west: float, south: float, east: float, north: float,
    shapefile_path: str,
) -> gpd.GeoDataFrame:
    """Load ASC soil polygons that intersect the AOI, reprojected to EPSG:3857.

    The full NSW soil shapefile is ~384 MB.  To avoid loading the entire file,
    geopandas passes a bbox filter to fiona, which uses the shapefile's spatial
    index to read only the polygons relevant to the AOI.

    The shapefile is natively in EPSG:4283 (GDA94), so the AOI bbox is first
    converted to that CRS before being used as the read filter.

    Args:
        west, south, east, north: AOI bounding box in EPSG:3857.
        shapefile_path: Path to SoilType_ASC_NSW_v4_5_210429.shp.

    Returns:
        GeoDataFrame in EPSG:3857 with an integer 'soil_code' column (uint8).
    """
    if not os.path.exists(shapefile_path):
        raise FileNotFoundError(
            f"Soil shapefile not found at '{shapefile_path}'. "
            f"Place SoilType_ASC_NSW_v4_5_210429.shp (and .dbf, .shx, .prj) "
            f"in ./{ASC_DIR}/ and re-run."
        )

    # Convert AOI corners from EPSG:3857 to EPSG:4283 (GDA94) for the bbox filter.
    to_4283         = Transformer.from_crs(CRS_EPSG, 4283, always_xy=True)
    w_4283, s_4283  = to_4283.transform(west,  south)
    e_4283, n_4283  = to_4283.transform(east,  north)

    print(f"  Reading shapefile with bbox filter (GDA94): "
          f"W={w_4283:.4f}  S={s_4283:.4f}  E={e_4283:.4f}  N={n_4283:.4f}")

    gdf = gpd.read_file(
        shapefile_path,
        bbox=(w_4283, s_4283, e_4283, n_4283),
    )

    # Reproject to Web Mercator so polygon coordinates align with the aerial tile.
    gdf = gdf.to_crs(epsg=CRS_EPSG)

    # Map the ASC_order text column to integer codes.
    gdf["soil_code"] = gdf["ASC_order"].map(ASC_ORDER_MAP)

    n_unknown = gdf["soil_code"].isna().sum()
    if n_unknown > 0:
        unknown_vals = gdf.loc[gdf["soil_code"].isna(), "ASC_order"].unique()
        print(f"  Warning: {n_unknown} feature(s) with unrecognised ASC_order dropped: {unknown_vals}")
    gdf = gdf.dropna(subset=["soil_code"])
    gdf["soil_code"] = gdf["soil_code"].astype(np.uint8)

    if gdf.empty:
        print("  Warning: No soil features found within the AOI. "
              "All soil pixels will be 0 (nodata).")

    print(f"  Loaded {len(gdf)} soil polygon(s).")
    return gdf


def rasterize_soil_tile(
    gdf: gpd.GeoDataFrame,
    west: float, south: float, east: float, north: float,
) -> np.ndarray:
    """Rasterise soil polygons onto the 512×512 tile grid.

    Converts vector soil polygons to a raster of integer codes.  Pixels not
    covered by any polygon receive code 0 (nodata).

    The two-step spatial filter (bbox index → precise intersection) avoids
    iterating over every polygon in the full dataset for each tile.

    'all_touched=True' matches the training pipeline (soil.py line 121), so
    pixels that are only partially covered by a polygon boundary are still assigned
    the polygon's code rather than being left as nodata.

    Args:
        gdf:               Soil GeoDataFrame in EPSG:3857 with 'soil_code' column.
        west, south, east, north: Tile bounding box in EPSG:3857.

    Returns:
        (TILE_SIZE, TILE_SIZE) uint8 array of soil codes; 0 = nodata.
    """
    transform = from_bounds(west, south, east, north, TILE_SIZE, TILE_SIZE)
    tile_box  = box(west, south, east, north)

    # Fast spatial index pre-filter, then precise geometric intersection
    candidate_idx = list(gdf.sindex.intersection((west, south, east, north)))
    tile_gdf      = gdf.iloc[candidate_idx]
    tile_gdf      = tile_gdf[tile_gdf.intersects(tile_box)]

    if tile_gdf.empty:
        return np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.uint8)

    shapes = list(zip(tile_gdf.geometry, tile_gdf["soil_code"]))
    raster = rasterize(
        shapes,
        out_shape=(TILE_SIZE, TILE_SIZE),
        transform=transform,
        fill=0,             # nodata value
        dtype=np.uint8,
        all_touched=True,   # match training pipeline
    )
    return raster


# =============================================================================
# STAGE 3 — LIDAR FEATURE DERIVATION
# Adapted from data_preprocessing/lidar.py.
# Changes: AOI bbox passed as explicit arguments; extraction target changed from
# ../data/raw/lidar/_extracted to ./inputs/_extracted; no output GeoTIFFs written
# (arrays returned directly); write_lidar_tile removed.
# =============================================================================

def extract_elvis_zips(raw_dir: str) -> ElvisData:
    """Extract ELVIS .zip file(s) and return paths to the contained LAZ/LAS files.

    Extraction is skipped if the _extracted/ subdirectory already contains LAZ/LAS
    files, so re-running after a failed first attempt is safe.

    Args:
        raw_dir: Directory containing the ELVIS .zip file (i.e., ./inputs/).

    Returns:
        ElvisData with a list of Path objects pointing to all extracted LAZ/LAS files.
    """
    zip_files = sorted(Path(raw_dir).glob("*.zip"))
    if not zip_files:
        raise RuntimeError(
            f"No .zip file found in ./{raw_dir}/. "
            "Place the ELVIS download package there and re-run."
        )

    extract_dir = Path(EXTRACT_DIR)
    extract_dir.mkdir(parents=True, exist_ok=True)

    # Skip extraction if already done — zip extraction can be slow on large packages.
    existing = list(extract_dir.rglob("*.laz")) + list(extract_dir.rglob("*.las"))
    if existing:
        print(f"  Extraction directory already populated ({len(existing)} file(s)) — skipping.")
    else:
        for zip_path in zip_files:
            print(f"  Extracting {zip_path.name} ...")
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(extract_dir)

    las_files = sorted(extract_dir.rglob("*.laz")) + sorted(extract_dir.rglob("*.las"))
    if not las_files:
        raise RuntimeError(
            "No LAZ/LAS point cloud files found inside the ELVIS zip(s). "
            "Ensure the ELVIS order included point cloud data (not just rasters)."
        )

    print(f"  Found {len(las_files)} LAZ/LAS file(s).")
    return ElvisData(las_files=las_files)


def filter_files_to_aoi(
    elvis: ElvisData,
    west: float, south: float, east: float, north: float,
    fallback_epsg: int,
) -> ElvisData:
    """Remove LAZ/LAS files that do not overlap the AOI (expanded by DEM_BUFFER).

    Reads only file headers (cheap) and converts each file's bounding box to
    EPSG:3857 for the overlap check.  Files whose headers cannot be parsed are
    kept to avoid silently dropping valid data.

    The AOI is expanded by DEM_BUFFER on all sides so that files needed for
    TWI flow-accumulation context beyond the tile edge are not filtered out.

    Args:
        elvis:        Container with all extracted LAZ/LAS paths.
        west, south, east, north: AOI bounding box in EPSG:3857.
        fallback_epsg: EPSG to assume when a file header lacks CRS metadata.

    Returns:
        New ElvisData containing only files that overlap the expanded AOI.
    """
    kept: List[Path] = []

    for las_path in elvis.las_files:
        try:
            with laspy.open(las_path) as reader:
                hdr  = reader.header
                mins = np.array([hdr.x_min, hdr.y_min, hdr.z_min])
                maxs = np.array([hdr.x_max, hdr.y_max, hdr.z_max])

                # Try to read the source CRS from the file's VLR (Variable Length Records).
                src_epsg = fallback_epsg
                try:
                    las_crs = hdr.parse_crs()
                    if las_crs is not None:
                        epsg = las_crs.to_epsg()
                        if epsg is not None:
                            src_epsg = epsg
                except Exception:
                    pass

            # Reproject the file's bounding box corners to EPSG:3857.
            tf = Transformer.from_crs(src_epsg, CRS_EPSG, always_xy=True)
            x_min_3857, y_min_3857 = tf.transform(mins[0], mins[1])
            x_max_3857, y_max_3857 = tf.transform(maxs[0], maxs[1])

            # Overlap test with the DEM-buffered AOI.
            if (x_max_3857 >= west  - DEM_BUFFER and x_min_3857 <= east  + DEM_BUFFER and
                    y_max_3857 >= south - DEM_BUFFER and y_min_3857 <= north + DEM_BUFFER):
                kept.append(las_path)

        except Exception as e:
            # If the header cannot be read at all, keep the file and let laspy
            # raise a more informative error when it is actually read.
            print(f"  Warning: could not read header of {las_path.name} ({e}); keeping it.")
            kept.append(las_path)

    print(f"  Filtered to {len(kept)}/{len(elvis.las_files)} LAZ/LAS file(s) for this AOI.")

    if not kept:
        raise RuntimeError(
            "No LAZ/LAS files overlap the AOI bounding box (including DEM buffer). "
            "Verify that the ELVIS zip covers the same area as the aerial image."
        )

    return ElvisData(las_files=kept)


def generate_dem_from_points(
    las_files: List[Path],
    west: float, south: float, east: float, north: float,
    fallback_epsg: int,
) -> Tuple[np.ndarray, object, float]:
    """Build a Digital Elevation Model from ground-classified LiDAR returns.

    Ground points (ASPRS classes 2 and 8) are reprojected to EPSG:3857 and
    binned into a DEM_RESOLUTION grid that covers the AOI plus DEM_BUFFER on all
    sides.  The minimum Z per cell is used (more robust against residual
    misclassified non-ground points than the mean).  Empty cells are gap-filled
    via nearest-neighbour distance transform, then lightly smoothed with a
    Gaussian filter to remove hard cell-edge discontinuities.

    Note on EPSG:3857 distortion: at NSW latitudes (~34°S) Web Mercator introduces
    a scale factor of ~1.2, so a "1 m cell" in EPSG:3857 corresponds to ~0.83 m
    on the ground.  The training pipeline used this convention throughout, so
    inference must match it.

    Args:
        las_files:    Filtered list of LAZ/LAS file paths.
        west, south, east, north: AOI bounding box in EPSG:3857.
        fallback_epsg: CRS to assume when a file lacks embedded metadata.

    Returns:
        dem_array — 2D float32 ndarray covering the buffered AOI
        transform — Affine transform for dem_array in EPSG:3857
        cell_size — DEM_RESOLUTION (same value throughout; returned for clarity)
    """
    # Build the buffered grid extent.
    grid_w = west  - DEM_BUFFER
    grid_e = east  + DEM_BUFFER
    grid_s = south - DEM_BUFFER
    grid_n = north + DEM_BUFFER

    ncols = int(np.ceil((grid_e - grid_w) / DEM_RESOLUTION))
    nrows = int(np.ceil((grid_n - grid_s) / DEM_RESOLUTION))

    # from_origin(west, north, xsize, ysize): top-left origin, positive-x-right,
    # positive-y-down convention used by rasterio.
    transform = from_origin(grid_w, grid_n, DEM_RESOLUTION, DEM_RESOLUTION)

    print(f"  DEM grid: {ncols}×{nrows} px @ {DEM_RESOLUTION:.1f} m/px "
          f"(AOI + {DEM_BUFFER} m buffer, EPSG:{CRS_EPSG})")

    all_gx: List[np.ndarray] = []
    all_gy: List[np.ndarray] = []
    all_gz: List[np.ndarray] = []

    for las_path in las_files:
        print(f"  Reading ground points from {las_path.name} ...")
        las = laspy.read(las_path)

        # Detect source CRS from VLR; fall back to the zone-derived GDA2020 MGA EPSG.
        src_epsg = fallback_epsg
        try:
            las_crs = las.header.parse_crs()
            if las_crs is not None:
                epsg = las_crs.to_epsg()
                if epsg is not None:
                    src_epsg = epsg
        except Exception:
            pass

        # Select only ASPRS ground classes.
        classification = np.asarray(las.classification, dtype=np.uint8)
        ground_mask    = np.isin(classification, GROUND_CLASSES)
        if not ground_mask.any():
            print(f"    No ground-classified points in {las_path.name}; skipping.")
            continue

        x_src = np.asarray(las.x, dtype=np.float64)[ground_mask]
        y_src = np.asarray(las.y, dtype=np.float64)[ground_mask]
        z_src = np.asarray(las.z, dtype=np.float32)[ground_mask]

        tf = Transformer.from_crs(src_epsg, CRS_EPSG, always_xy=True)
        x_3857, y_3857 = tf.transform(x_src, y_src)
        all_gx.append(np.asarray(x_3857, dtype=np.float64))
        all_gy.append(np.asarray(y_3857, dtype=np.float64))
        all_gz.append(z_src)

    if not all_gx:
        raise RuntimeError(
            "No ground-classified points (ASPRS classes 2 or 8) found in any file. "
            "Ensure the point cloud data has been ground-classified before use."
        )

    gx = np.concatenate(all_gx)
    gy = np.concatenate(all_gy)
    gz = np.concatenate(all_gz)

    # Clip to buffered grid extent.
    in_grid = (gx >= grid_w) & (gx < grid_e) & (gy > grid_s) & (gy <= grid_n)
    gx, gy, gz = gx[in_grid], gy[in_grid], gz[in_grid]

    if len(gx) == 0:
        raise RuntimeError(
            "No ground points fall within the AOI + buffer extent. "
            "Check that the ELVIS zip covers the same area as the aerial image."
        )

    area_km2 = (grid_e - grid_w) * (grid_n - grid_s) / 1e6
    print(f"  {len(gx):,} ground points over {area_km2:.2f} km² "
          f"({len(gx) / (area_km2 * 1e6):.3f} pts/m²)")

    # Bin each ground point to a DEM cell; keep minimum Z per cell.
    col      = np.clip(((gx - grid_w) / DEM_RESOLUTION).astype(np.int32), 0, ncols - 1)
    row      = np.clip(((grid_n - gy) / DEM_RESOLUTION).astype(np.int32), 0, nrows - 1)
    cell_idx = row * ncols + col

    order        = np.argsort(cell_idx)
    sorted_cells = cell_idx[order]
    sorted_z     = gz[order]
    unique_cells, first_occ = np.unique(sorted_cells, return_index=True)

    dem_flat = np.full(nrows * ncols, np.nan, dtype=np.float32)
    dem_flat[unique_cells] = np.minimum.reduceat(sorted_z, first_occ)
    dem = dem_flat.reshape(nrows, ncols)

    # Gap-fill empty cells with their nearest valid neighbour.
    valid_mask = ~np.isnan(dem)
    gap_pct    = 100.0 * (1.0 - valid_mask.sum() / (nrows * ncols))
    print(f"  Gap-filling {gap_pct:.1f}% of DEM cells via nearest-neighbour interpolation ...")
    _, (row_idx, col_idx) = scipy.ndimage.distance_transform_edt(
        ~valid_mask, return_indices=True
    )
    dem = dem[row_idx, col_idx]

    # Light Gaussian smooth removes hard edges introduced by the gap-fill.
    dem = scipy.ndimage.gaussian_filter(dem, sigma=0.5)

    print(f"  DEM complete: {ncols}×{nrows} px @ {DEM_RESOLUTION:.1f} m/px")
    return dem, transform, DEM_RESOLUTION


def load_and_normalize_points(
    las_files: List[Path],
    dem_data: np.ndarray,
    dem_transform,
    west: float, south: float, east: float, north: float,
    fallback_epsg: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load all LiDAR returns, reproject to EPSG:3857, and normalise height against the DEM.

    After normalisation each point's Z value is its height above the local ground
    surface (in metres), not an absolute elevation.  Negative heights (noise or
    edge artefacts) are clipped to zero.

    Args:
        las_files:    Filtered list of LAZ/LAS file paths.
        dem_data:     2D float32 DEM array (output of generate_dem_from_points).
        dem_transform: Affine transform for dem_data.
        west, south, east, north: AOI bounding box in EPSG:3857.
        fallback_epsg: CRS fallback when file header lacks embedded metadata.

    Returns:
        x — easting in EPSG:3857, float64, shape (N,)
        y — northing in EPSG:3857, float64, shape (N,)
        h — height above ground in metres, clipped to ≥0, float32, shape (N,)
    """
    all_x: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    all_z: List[np.ndarray] = []

    for las_path in las_files:
        print(f"  Reading {las_path.name} ...")
        las = laspy.read(las_path)

        src_epsg = fallback_epsg
        try:
            las_crs = las.header.parse_crs()
            if las_crs is not None:
                epsg = las_crs.to_epsg()
                if epsg is not None:
                    src_epsg = epsg
                else:
                    print(f"  Warning: EPSG unresolvable in {las_path.name}; "
                          f"falling back to EPSG:{fallback_epsg}.")
            else:
                print(f"  Warning: No CRS in header of {las_path.name}; "
                      f"falling back to EPSG:{fallback_epsg}.")
        except Exception:
            print(f"  Warning: Could not parse CRS in {las_path.name}; "
                  f"falling back to EPSG:{fallback_epsg}.")

        tf = Transformer.from_crs(src_epsg, CRS_EPSG, always_xy=True)
        x_3857, y_3857 = tf.transform(
            np.asarray(las.x, dtype=np.float64),
            np.asarray(las.y, dtype=np.float64),
        )
        all_x.append(np.asarray(x_3857, dtype=np.float64))
        all_y.append(np.asarray(y_3857, dtype=np.float64))
        all_z.append(np.asarray(las.z,  dtype=np.float32))

    x = np.concatenate(all_x)
    y = np.concatenate(all_y)
    z = np.concatenate(all_z)

    # Discard points outside the AOI (10 m buffer avoids clipping tile-edge returns).
    buf     = 10.0
    in_aoi  = (
        (x >= west  - buf) & (x <= east  + buf) &
        (y >= south - buf) & (y <= north + buf)
    )
    x, y, z = x[in_aoi], y[in_aoi], z[in_aoi]
    print(f"  {in_aoi.sum():,} / {len(in_aoi):,} points within AOI.")

    if len(x) == 0:
        raise RuntimeError(
            "No LiDAR points fall within the AOI bounding box. "
            "Check that the ELVIS zip covers the same area as the aerial image."
        )

    # Look up ground elevation for each point from the DEM.
    rows, cols = rasterio.transform.rowcol(dem_transform, x, y)
    rows   = np.asarray(rows, dtype=np.int32)
    cols   = np.asarray(cols, dtype=np.int32)
    h_dem, w_dem = dem_data.shape

    in_dem  = (rows >= 0) & (rows < h_dem) & (cols >= 0) & (cols < w_dem)
    rows_c  = np.clip(rows, 0, h_dem - 1)
    cols_c  = np.clip(cols, 0, w_dem - 1)

    ground_elev            = dem_data[rows_c, cols_c].astype(np.float32)
    ground_elev[~in_dem]   = np.nan   # mark points outside DEM extent for removal

    h = z - ground_elev
    h = np.clip(h, 0.0, None)         # clip negatives (noise / edge artefacts)

    # Discard points with no valid ground elevation reference.
    valid  = ~np.isnan(ground_elev)
    x, y, h = x[valid], y[valid], h[valid]

    print(f"  Height normalisation complete. {len(h):,} valid points retained.")
    return x, y, h


def compute_lidar_bands(
    x: np.ndarray, y: np.ndarray, h: np.ndarray,
    xmin: float, ymin: float, xmax: float, ymax: float,
) -> np.ndarray:
    """Bin LiDAR returns into a 512×512 grid and compute the six structural feature bands.

    All aggregation is vectorised with np.bincount — there are no Python loops
    over individual cells.

    Per-pixel outputs:
        Band 0: CHM — maximum return height (metres above ground).
                NaN for pixels with zero returns.
        Band 1: Canopy Cover Density — fraction of returns ≥ CANOPY_THRESHOLD (2 m).
                0.0 for pixels with zero returns.
        Band 2: Strata Ground       — fraction of returns 0.0–0.5 m above ground.
        Band 3: Strata Understorey  — fraction of returns 0.5–2.0 m.
        Band 4: Strata Midstorey    — fraction of returns 2.0–10.0 m.
        Band 5: Strata Canopy       — fraction of returns ≥ 10.0 m.

    Args:
        x, y:             Point cloud XY coordinates in EPSG:3857.
        h:                Height above ground (normalised Z) in metres.
        xmin, ymin,
        xmax, ymax:       Tile bounding box in EPSG:3857.

    Returns:
        (6, TILE_SIZE, TILE_SIZE) float32 array.
    """
    flat_size = TILE_SIZE * TILE_SIZE
    pixel_w   = (xmax - xmin) / TILE_SIZE
    pixel_h   = (ymax - ymin) / TILE_SIZE

    # Filter to this tile's extent.
    mask      = (x >= xmin) & (x < xmax) & (y >= ymin) & (y < ymax)
    tx, ty, th = x[mask], y[mask], h[mask]

    if len(tx) == 0:
        # No returns at all — CHM is NaN everywhere, density bands are zero.
        chm   = np.full((TILE_SIZE, TILE_SIZE), np.nan, dtype=np.float32)
        zeros = np.zeros((5, TILE_SIZE, TILE_SIZE), dtype=np.float32)
        return np.concatenate([chm[np.newaxis], zeros], axis=0)

    # Convert point XY to pixel row/col indices.
    # rasterio convention: top-left origin, Y axis points downward.
    col      = np.clip(((tx - xmin) / pixel_w).astype(np.int32), 0, TILE_SIZE - 1)
    row      = np.clip(((ymax - ty) / pixel_h).astype(np.int32), 0, TILE_SIZE - 1)
    cell_idx = row * TILE_SIZE + col

    # Per-cell point count (denominator for all fraction computations).
    counts = np.bincount(cell_idx, minlength=flat_size).astype(np.float32)

    # CHM: maximum height per cell via reduceat on sorted groups.
    order        = np.argsort(cell_idx)
    sorted_cells = cell_idx[order]
    sorted_h     = th[order]
    unique_cells, first_occ = np.unique(sorted_cells, return_index=True)
    chm_flat = np.full(flat_size, np.nan, dtype=np.float32)
    chm_flat[unique_cells] = np.maximum.reduceat(sorted_h, first_occ)

    # Canopy cover density.
    above      = (th >= CANOPY_THRESHOLD).astype(np.float32)
    cover_flat = np.zeros(flat_size, dtype=np.float32)
    np.divide(
        np.bincount(cell_idx, weights=above, minlength=flat_size),
        counts, out=cover_flat, where=counts > 0,
    )

    # Strata density (one band per height bin).
    strata_flat = np.zeros((4, flat_size), dtype=np.float32)
    for i, (lo, hi) in enumerate(STRATA_BINS):
        in_band = ((th >= lo) & (th < hi)).astype(np.float32)
        np.divide(
            np.bincount(cell_idx, weights=in_band, minlength=flat_size),
            counts, out=strata_flat[i], where=counts > 0,
        )

    bands = np.stack(
        [chm_flat, cover_flat,
         strata_flat[0], strata_flat[1], strata_flat[2], strata_flat[3]],
        axis=0,
    ).reshape(6, TILE_SIZE, TILE_SIZE)

    return bands.astype(np.float32)


@numba.njit
def _d8_flow_accumulation(dem_flat: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """D8 flow direction and accumulation, JIT-compiled with numba for performance.

    D8 routing assigns each cell to the single steepest downslope neighbour among
    its 8 connected neighbours (diagonal neighbours are penalised by √2 distance).
    Flow accumulation counts the number of cells that drain through each cell
    (including itself).  The result feeds into the specific catchment area term
    of the TWI formula.

    Cells are processed from highest to lowest elevation so that every cell's
    upstream contributions have already been accumulated by the time its own
    contribution is passed to its receiver.

    Args:
        dem_flat: 1D float32 array of elevation values, shape (rows*cols,).
                  NaN cells are treated as nodata and do not accumulate flow.
        rows, cols: Dimensions of the original 2D DEM grid.

    Returns:
        flow_acc: 1D float32 array, shape (rows*cols,).
                  Value at each cell = number of upstream cells including itself.
    """
    flat_size = rows * cols
    flow_acc  = np.ones(flat_size, dtype=np.float32)   # each cell starts with 1 (itself)
    flow_dir  = np.full(flat_size, -1, dtype=np.int32)  # -1 = no downslope receiver

    # Row/column offsets and distance factors for the 8 connected neighbours.
    n_dr   = (-1, -1, -1,  0,  0,  1,  1,  1)
    n_dc   = (-1,  0,  1, -1,  1, -1,  0,  1)
    n_dist = (1.41421356, 1.0, 1.41421356, 1.0, 1.0, 1.41421356, 1.0, 1.41421356)

    # Assign each cell the steepest downslope neighbour as its flow receiver.
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            if np.isnan(dem_flat[idx]):
                continue
            max_slope = 0.0
            best      = -1
            for k in range(8):
                nr = r + n_dr[k]
                nc = c + n_dc[k]
                if 0 <= nr < rows and 0 <= nc < cols:
                    nidx = nr * cols + nc
                    if not np.isnan(dem_flat[nidx]):
                        drop = (dem_flat[idx] - dem_flat[nidx]) / n_dist[k]
                        if drop > max_slope:
                            max_slope = drop
                            best      = nidx
            flow_dir[idx] = best

    # Replace NaN elevations with a sentinel value so they sort after all valid cells.
    dem_sort = dem_flat.copy()
    for i in range(flat_size):
        if np.isnan(dem_sort[i]):
            dem_sort[i] = -1e38

    # Process cells from highest to lowest elevation.
    sorted_order = np.argsort(-dem_sort)
    for i in range(flat_size):
        idx      = sorted_order[i]
        receiver = flow_dir[idx]
        if receiver >= 0:
            flow_acc[receiver] += flow_acc[idx]

    return flow_acc


def compute_twi(
    dem_data: np.ndarray,
    dem_transform,
    cell_size: float,
    xmin: float, ymin: float, xmax: float, ymax: float,
) -> np.ndarray:
    """Compute the Topographic Wetness Index (TWI) for the tile.

    TWI = ln(SCA / tan(slope))
        where SCA = Specific Catchment Area = flow_accumulation × cell_size
        and slope is the local terrain gradient in radians.

    A buffer of DEM_BUFFER pixels is extracted around the tile before running
    flow accumulation to give upstream cells context beyond the tile boundary,
    reducing edge artefacts.  The buffer is trimmed off again before returning.

    Flat terrain (slope ≈ 0) is clipped to 0.001 rad to avoid division by zero.

    If the DEM resolution produces a tile excerpt that is not exactly TILE_SIZE×TILE_SIZE
    (possible due to rounding), the result is resampled with bilinear interpolation.

    Args:
        dem_data:    Full buffered DEM array (from generate_dem_from_points).
        dem_transform: Affine transform for dem_data (EPSG:3857).
        cell_size:   DEM cell size in EPSG:3857 units (= DEM_RESOLUTION).
        xmin, ymin,
        xmax, ymax:  Tile bounding box in EPSG:3857.

    Returns:
        (TILE_SIZE, TILE_SIZE) float32 TWI raster.
    """
    buf_px = max(50, int(DEM_BUFFER / cell_size))

    # Find the tile extent in DEM pixel coordinates.
    row_n, col_w = rasterio.transform.rowcol(dem_transform, xmin, ymax)   # NW corner
    row_s, col_e = rasterio.transform.rowcol(dem_transform, xmax, ymin)   # SE corner

    h_dem, w_dem = dem_data.shape

    # Expand window by buf_px on all sides for hydrological context.
    win_r0 = max(0, min(row_n, row_s) - buf_px)
    win_r1 = min(h_dem, max(row_n, row_s) + buf_px + 1)
    win_c0 = max(0, min(col_w, col_e) - buf_px)
    win_c1 = min(w_dem, max(col_w, col_e) + buf_px + 1)

    dem_window      = dem_data[win_r0:win_r1, win_c0:win_c1].copy()
    win_h, win_w    = dem_window.shape

    # Compute slope magnitude from the DEM gradient.
    # np.gradient returns [dz/drow, dz/dcol]; we divide by cell_size to get rad/m.
    gy_arr, gx_arr  = np.gradient(dem_window, cell_size, cell_size)
    slope           = np.arctan(np.sqrt(gx_arr**2 + gy_arr**2))
    slope           = np.clip(slope, 0.001, None)   # avoid ln(inf) at flat terrain

    # D8 flow accumulation on the windowed DEM.
    flat     = np.ascontiguousarray(dem_window.ravel(), dtype=np.float32)
    flow_acc = _d8_flow_accumulation(flat, win_h, win_w).reshape(win_h, win_w)

    # TWI formula.
    sca = flow_acc * cell_size
    twi = np.log(sca / np.tan(slope)).astype(np.float32)

    # Trim back to just the tile extent within the windowed result.
    tr0       = max(0, min(row_n, row_s) - win_r0)
    tr1       = min(win_h, max(row_n, row_s) - win_r0 + 1)
    tc0       = max(0, min(col_w, col_e) - win_c0)
    tc1       = min(win_w, max(col_w, col_e) - win_c0 + 1)
    twi_tile  = twi[tr0:tr1, tc0:tc1]

    # Resample to exactly TILE_SIZE × TILE_SIZE if needed.
    if twi_tile.shape != (TILE_SIZE, TILE_SIZE):
        zoom_r   = TILE_SIZE / max(twi_tile.shape[0], 1)
        zoom_c   = TILE_SIZE / max(twi_tile.shape[1], 1)
        twi_tile = scipy.ndimage.zoom(twi_tile, (zoom_r, zoom_c), order=1)

    return twi_tile[:TILE_SIZE, :TILE_SIZE].astype(np.float32)


# =============================================================================
# STAGE 4 — MERGE MODALITIES AND WRITE OUTPUT GEOTIFF
# =============================================================================

def merge_and_write(
    aerial: np.ndarray,
    soil_raster: np.ndarray,
    lidar_bands: np.ndarray,
    transform,
    output_path: str,
) -> None:
    """Stack the three modalities into an 11-band float32 GeoTIFF and write to disk.

    Band order is fixed by the training pipeline (data_preprocessing/merge.py) and
    must not be changed — the model's weight tensors were optimised against this
    specific ordering.

    Aerial RGB bands are already float32.  The soil raster is uint8 and is cast
    to float32 before stacking, matching what merge.py does (line 64).  NaN values
    in the LiDAR bands are preserved, not replaced; downstream normalisation in the
    model handles them.

    The output GeoTIFF uses the same transform as the input aerial, so it is
    spatially co-registered with it.

    Args:
        aerial:       (3, 512, 512) float32 RGB array; bands [Red, Green, Blue].
        soil_raster:  (512, 512) uint8 array of ASC soil codes.
        lidar_bands:  (7, 512, 512) float32 array; bands [CHM, Cover, 4×Strata, TWI].
        transform:    Affine transform from the aerial GeoTIFF (EPSG:3857).
        output_path:  Destination file path for the merged GeoTIFF.
    """
    # Cast soil uint8 → float32 and add the channel dimension.
    soil_f32 = soil_raster.astype(np.float32)[np.newaxis]   # (1, 512, 512)

    # Concatenate in training band order: RGB | soil | LiDAR.
    stack = np.concatenate([aerial, soil_f32, lidar_bands], axis=0)   # (11, 512, 512)

    assert stack.shape == (N_BANDS, TILE_SIZE, TILE_SIZE), (
        f"Unexpected stack shape {stack.shape}; "
        f"expected ({N_BANDS}, {TILE_SIZE}, {TILE_SIZE})."
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with rasterio.open(
        output_path, "w",
        driver="GTiff",
        height=TILE_SIZE,
        width=TILE_SIZE,
        count=N_BANDS,
        dtype=np.float32,
        crs=CRS.from_epsg(CRS_EPSG),
        transform=transform,
        compress="lzw",      # lossless; matches data_preprocessing/merge.py
    ) as dst:
        dst.write(stack)
        dst.update_tags(bands=BAND_TAGS)

    print(f"\n[Output] {output_path}")
    print(f"  Shape: {stack.shape}  dtype: float32  CRS: EPSG:{CRS_EPSG}")
    print(f"  Bands: {BAND_TAGS}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """Run the full single-tile inference preprocessing pipeline.

    Expected directory layout (relative to inference/):
        inputs/   — one aerial .tif (512×512, EPSG:3857) + one ELVIS .zip
        asc/      — SoilType_ASC_NSW_v4_5_210429.shp and companion files
        outputs/  — created if absent; output .tif written here
    """
    print("=" * 60)
    print("NSW Vegetation Classifier — Inference Preprocessing")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # Stage 1: Read aerial image and derive the geographic bounding box.
    # -----------------------------------------------------------------------
    print("\n[Stage 1] Reading aerial image ...")
    aerial, transform, west, south, east, north = read_aerial(INPUTS_DIR)

    # Determine the most likely MGA zone for the AOI so we have a sensible
    # fallback CRS for LAS files that are missing embedded projection metadata.
    fallback_epsg = _mga_fallback_epsg(west, east, south, north)
    print(f"  LiDAR CRS fallback: EPSG:{fallback_epsg} (GDA2020 MGA zone "
          f"{fallback_epsg - 7800})")

    # -----------------------------------------------------------------------
    # Stage 2: Rasterise soil polygons onto the tile grid.
    # -----------------------------------------------------------------------
    print("\n[Stage 2] Rasterising soil data ...")
    soil_gdf    = load_soil_data(west, south, east, north, SOIL_SHAPEFILE)
    soil_raster = rasterize_soil_tile(soil_gdf, west, south, east, north)
    print(f"  Soil codes present in tile: {np.unique(soil_raster).tolist()}")

    # -----------------------------------------------------------------------
    # Stage 3a: Extract the ELVIS zip.
    # -----------------------------------------------------------------------
    print("\n[Stage 3a] Extracting ELVIS zip ...")
    elvis = extract_elvis_zips(INPUTS_DIR)

    # -----------------------------------------------------------------------
    # Stage 3b: Filter to LAZ/LAS files that overlap this AOI.
    # -----------------------------------------------------------------------
    print("\n[Stage 3b] Filtering point cloud files to AOI ...")
    elvis = filter_files_to_aoi(elvis, west, south, east, north, fallback_epsg)

    # -----------------------------------------------------------------------
    # Stage 3c: Generate DEM from ground-classified returns.
    # -----------------------------------------------------------------------
    print("\n[Stage 3c] Generating DEM from ground points ...")
    dem_data, dem_transform, cell_size = generate_dem_from_points(
        elvis.las_files, west, south, east, north, fallback_epsg
    )

    # -----------------------------------------------------------------------
    # Stage 3d: Load the full point cloud and normalise heights against DEM.
    # -----------------------------------------------------------------------
    print("\n[Stage 3d] Loading and normalising full point cloud ...")
    px, py, ph = load_and_normalize_points(
        elvis.las_files, dem_data, dem_transform,
        west, south, east, north, fallback_epsg
    )

    # -----------------------------------------------------------------------
    # Numba JIT warmup.
    # The first call to _d8_flow_accumulation triggers compilation (~5–10 s).
    # Running a trivial dummy call here causes that compile to happen now,
    # at a clearly labelled step, rather than silently mid-computation.
    # -----------------------------------------------------------------------
    print("\n  Warming up numba JIT compiler (first-run compilation) ...")
    _d8_flow_accumulation(np.ones(4, dtype=np.float32), 2, 2)
    print("  JIT ready.")

    # -----------------------------------------------------------------------
    # Stage 3e: Compute the six point-cloud structural feature bands.
    # -----------------------------------------------------------------------
    print("\n[Stage 3e] Computing point-cloud feature bands ...")
    pc_bands = compute_lidar_bands(px, py, ph, west, south, east, north)
    print(f"  Output shape: {pc_bands.shape}  "
          f"(bands: CHM, CanopyCover, StrataGround, StrataUnderstorey, "
          f"StrataMidstorey, StrataCanopy)")

    # -----------------------------------------------------------------------
    # Stage 3f: Compute Topographic Wetness Index from the DEM.
    # -----------------------------------------------------------------------
    print("\n[Stage 3f] Computing TWI ...")
    twi = compute_twi(dem_data, dem_transform, cell_size, west, south, east, north)
    print(f"  TWI shape: {twi.shape}")

    # Concatenate the 6 point-cloud bands and 1 TWI band → (7, 512, 512).
    lidar_bands = np.concatenate([pc_bands, twi[np.newaxis]], axis=0)
    print(f"  Combined LiDAR bands shape: {lidar_bands.shape}")

    # -----------------------------------------------------------------------
    # Stage 4: Stack all modalities and write the output GeoTIFF.
    # -----------------------------------------------------------------------
    print("\n[Stage 4] Merging modalities and writing output ...")
    aerial_stem = Path(list(Path(INPUTS_DIR).glob("*.tif"))[0]).stem
    output_path = os.path.join(OUTPUTS_DIR, f"{aerial_stem}.tif")
    merge_and_write(aerial, soil_raster, lidar_bands, transform, output_path)

    print("\n" + "=" * 60)
    print("Preprocessing complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
