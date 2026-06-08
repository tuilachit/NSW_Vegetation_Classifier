# Model Assets

Large binary assets are kept out of normal Git history.

## Required Model Files

Place these files in `backend/model_assets/`:

```text
binary_veg_other_p2_best.weights.h5
group9_after_binary_p2_best.weights.h5
binary_run_metadata.json
group9_run_metadata.json
```

Optional evaluation summaries:

```text
binary_eval_summary.csv
group9_eval_summary.csv
group9_p2_best_test_per_class_iou.csv
```

## Required Soil Files

Raw RGB + ELVIS inference also needs the ASC shapefile in `backend/soil_assets/asc/`:

```text
SoilType_ASC_NSW_v4_5_210429.shp
SoilType_ASC_NSW_v4_5_210429.dbf
SoilType_ASC_NSW_v4_5_210429.shx
SoilType_ASC_NSW_v4_5_210429.prj
```

To use a different folder:

```bash
export VEGEMAP_ASC_DIR=/absolute/path/to/asc
```

## GitHub Options

For a public portfolio repo, keep the source code in Git and provide model assets through one of these channels:

- GitHub Releases for downloadable weight archives.
- Git LFS if the repository quota allows it.
- A private course-submission archive when the project is being marked.

The `.gitignore` is configured to avoid accidentally committing large model files.
