"""Constants for the NSW vegetation GUI inference package."""

from __future__ import annotations

PATCH_SIZE = 512
INPUT_CHANNELS = 11

RGB_IDXS = [0, 1, 2]
AUX_IDXS = [3, 4, 5, 6, 7, 8, 9, 10]
SOIL_IDX = 3
CHM_IDX = 4
BINARY_IDXS = [5, 6, 7, 8, 9]
TWI_IDX = 10
ZSCORE_IDXS = [CHM_IDX, TWI_IDX]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

BINARY_CLASS_NAMES = [
    "Other / non-target land cover",
    "Vegetation classes 1-16",
]

GROUP_CLASS_NAMES = [
    "Alpine",
    "Arid Shrublands",
    "Dry Sclerophyll Forests",
    "Wetlands",
    "Grass / Grassy Woodlands",
    "Heathlands",
    "Rainforests",
    "Semi-arid Woodlands",
    "Wet Sclerophyll Forests",
]

FINAL_CLASS_VALUES = {
    "nodata": 0,
    "other": 17,
    "groups": {
        1: "Alpine",
        2: "Arid Shrublands",
        3: "Dry Sclerophyll Forests",
        4: "Wetlands",
        5: "Grass / Grassy Woodlands",
        6: "Heathlands",
        7: "Rainforests",
        8: "Semi-arid Woodlands",
        9: "Wet Sclerophyll Forests",
    },
}

RAW_TO_GROUP = {
    1: 0,
    2: 1,
    3: 1,
    4: 2,
    5: 2,
    6: 3,
    7: 3,
    12: 3,
    8: 4,
    9: 4,
    10: 5,
    11: 6,
    13: 7,
    14: 7,
    15: 8,
    16: 8,
}
