# ASC Soil Assets

This folder is where the backend expects the NSW ASC soil shapefile used by the
raw aerial + ELVIS inference workflow.

Required files:

- SoilType_ASC_NSW_v4_5_210429.shp
- SoilType_ASC_NSW_v4_5_210429.dbf
- SoilType_ASC_NSW_v4_5_210429.shx
- SoilType_ASC_NSW_v4_5_210429.prj

The shapefile is intentionally ignored by Git because it is a large geospatial
asset. To keep it somewhere else, set:

```bash
export VEGEMAP_ASC_DIR=/absolute/path/to/asc
```
